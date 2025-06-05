#!/usr/bin/env python3
"""
Real-Time MJX-Robot Integration System

Provides bidirectional data flow between MuJoCo MJX simulation and real robot
with sub-10ms latency for real-time robot control and policy validation.

Features:
- Real-time sensor data streaming
- Synchronized MJX simulation
- Bidirectional state validation
- Domain adaptation
- Policy execution framework
- Performance monitoring
"""

import asyncio
import time
import threading
import queue
import json
import numpy as np
import jax
import jax.numpy as jp
from mujoco import mjx
import cv2
import pyrealsense2 as rs
import websockets
import zmq
from dataclasses import dataclass, asdict
from typing import Optional, Tuple, Dict, Any, Callable
from concurrent.futures import ThreadPoolExecutor
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class IntegrationConfig:
    """Configuration for MJX-Robot integration"""
    # Robot connection
    robot_ip: str = "10.20.0.46"
    robot_port: int = 5555
    
    # Sensor configuration
    depth_width: int = 640
    depth_height: int = 480
    depth_fps: int = 30
    
    # MJX configuration
    mjcf_path: str = "robot_scene.xml"
    num_worlds: int = 1
    render_width: int = 64
    render_height: int = 64
    
    # Performance thresholds
    max_latency_ms: float = 10.0
    target_fps: float = 30.0
    max_domain_gap: float = 0.05
    
    # Validation settings
    validation_frequency: int = 10  # Every N frames
    state_sync_tolerance: float = 0.01

class RealTimeSensorInterface:
    """Real-time interface to robot sensors"""
    
    def __init__(self, config: IntegrationConfig):
        self.config = config
        self.pipeline = None
        self.depth_queue = queue.Queue(maxsize=5)
        self.rgb_queue = queue.Queue(maxsize=5)
        self.running = False
        self.thread = None
        
    def start(self):
        """Start sensor streaming"""
        try:
            self.pipeline = rs.pipeline()
            rs_config = rs.config()
            
            # Configure depth and RGB streams
            rs_config.enable_stream(rs.stream.depth, 
                                  self.config.depth_width, 
                                  self.config.depth_height, 
                                  rs.format.z16, 
                                  self.config.depth_fps)
            rs_config.enable_stream(rs.stream.color,
                                  self.config.depth_width,
                                  self.config.depth_height,
                                  rs.format.bgr8,
                                  self.config.depth_fps)
            
            # Start streaming
            profile = self.pipeline.start(rs_config)
            
            # Get camera intrinsics for MJX synchronization
            depth_profile = rs.video_stream_profile(profile.get_stream(rs.stream.depth))
            self.intrinsics = depth_profile.get_intrinsics()
            
            self.running = True
            self.thread = threading.Thread(target=self._capture_loop)
            self.thread.start()
            
            logger.info(f"Sensor interface started: {self.config.depth_width}x{self.config.depth_height}@{self.config.depth_fps}fps")
            return True
            
        except Exception as e:
            logger.error(f"Failed to start sensor interface: {e}")
            return False
    
    def _capture_loop(self):
        """Main sensor capture loop"""
        align = rs.align(rs.stream.color)
        
        while self.running:
            try:
                frames = self.pipeline.wait_for_frames(timeout_ms=100)
                aligned_frames = align.process(frames)
                
                depth_frame = aligned_frames.get_depth_frame()
                color_frame = aligned_frames.get_color_frame()
                
                if depth_frame and color_frame:
                    timestamp = time.time()
                    
                    # Convert to numpy arrays
                    depth_image = np.asanyarray(depth_frame.get_data())
                    color_image = np.asanyarray(color_frame.get_data())
                    
                    # Add to queues (non-blocking)
                    try:
                        self.depth_queue.put_nowait({
                            'data': depth_image,
                            'timestamp': timestamp,
                            'frame_number': depth_frame.get_frame_number()
                        })
                        self.rgb_queue.put_nowait({
                            'data': color_image,
                            'timestamp': timestamp,
                            'frame_number': color_frame.get_frame_number()
                        })
                    except queue.Full:
                        # Drop frames if queues are full (maintain real-time)
                        pass
                        
            except RuntimeError:
                # Timeout - continue
                continue
            except Exception as e:
                logger.error(f"Sensor capture error: {e}")
                break
    
    def get_latest_depth(self) -> Optional[Dict[str, Any]]:
        """Get latest depth frame (non-blocking)"""
        try:
            return self.depth_queue.get_nowait()
        except queue.Empty:
            return None
    
    def get_latest_rgb(self) -> Optional[Dict[str, Any]]:
        """Get latest RGB frame (non-blocking)"""
        try:
            return self.rgb_queue.get_nowait()
        except queue.Empty:
            return None
    
    def stop(self):
        """Stop sensor streaming"""
        self.running = False
        if self.thread:
            self.thread.join()
        if self.pipeline:
            self.pipeline.stop()
        logger.info("Sensor interface stopped")

class MJXSimulationEngine:
    """MJX simulation engine with real-time rendering"""
    
    def __init__(self, config: IntegrationConfig):
        self.config = config
        self.model = None
        self.mjx_model = None
        self.mjx_data = None
        self.renderer = None
        self.render_token = None
        self.step_fn = None
        self.render_fn = None
        
    def initialize(self) -> bool:
        """Initialize MJX simulation"""
        try:
            from madrona_mjx.renderer import BatchRenderer
            from madrona_mjx.wrapper import load_model, _identity_randomization_fn
            
            # Load model
            self.model = load_model(self.config.mjcf_path)
            self.mjx_model = mjx.put_model(self.model)
            
            # Initialize renderer
            self.renderer = BatchRenderer(
                self.mjx_model,
                gpu_id=0,
                num_worlds=self.config.num_worlds,
                batch_render_view_width=self.config.render_width,
                batch_render_view_height=self.config.render_height,
                camera_ids=np.array([0]),  # Use first camera
                add_cam_debug_geo=False,
                use_rasterizer=True
            )
            
            # Setup vectorized model for batch processing
            v_mjx_model, v_in_axes = _identity_randomization_fn(
                self.mjx_model, self.config.num_worlds
            )
            
            # Initialize simulation state
            @jax.jit
            def init_simulation(rng, model):
                def init_single(rng, model):
                    data = mjx.make_data(model)
                    data = mjx.forward(model, data)
                    render_token, rgb, depth = self.renderer.init(data, model)
                    return data, render_token, rgb, depth
                
                return jax.vmap(init_single, in_axes=[0, v_in_axes])(rng, model)
            
            # Initialize with random seed
            rng = jax.random.PRNGKey(42)
            rng, *keys = jax.random.split(rng, self.config.num_worlds + 1)
            self.mjx_data, self.render_token, initial_rgb, initial_depth = init_simulation(
                jp.asarray(keys), v_mjx_model
            )
            
            # Compile step and render functions
            @jax.jit
            def step_simulation(data, actions):
                def step_single(data, action):
                    data = data.replace(ctrl=action)
                    return mjx.step(self.mjx_model, data)
                
                return jax.vmap(step_single)(data, actions)
            
            @jax.jit
            def render_simulation(render_token, data):
                def render_single(token, data):
                    _, rgb, depth = self.renderer.render(token, data)
                    return rgb, depth
                
                return jax.vmap(render_single)(render_token, data)
            
            self.step_fn = step_simulation
            self.render_fn = render_simulation
            
            logger.info(f"MJX simulation initialized: {self.config.num_worlds} worlds")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize MJX simulation: {e}")
            return False
    
    def step(self, actions: jp.ndarray) -> Tuple[jp.ndarray, jp.ndarray]:
        """Step simulation and render"""
        if self.step_fn is None or self.render_fn is None:
            raise RuntimeError("Simulation not initialized")
        
        # Step physics
        self.mjx_data = self.step_fn(self.mjx_data, actions)
        
        # Render observations
        rgb, depth = self.render_fn(self.render_token, self.mjx_data)
        
        return rgb, depth
    
    def get_state(self) -> Dict[str, Any]:
        """Get current simulation state"""
        if self.mjx_data is None:
            return {}
        
        return {
            'qpos': np.array(self.mjx_data.qpos[0]),  # First world
            'qvel': np.array(self.mjx_data.qvel[0]),
            'time': float(self.mjx_data.time[0])
        }
    
    def set_state(self, qpos: np.ndarray, qvel: np.ndarray):
        """Set simulation state"""
        if self.mjx_data is None:
            return
        
        # Update first world (can extend to all worlds)
        new_qpos = self.mjx_data.qpos.at[0].set(qpos)
        new_qvel = self.mjx_data.qvel.at[0].set(qvel)
        
        self.mjx_data = self.mjx_data.replace(qpos=new_qpos, qvel=new_qvel)
        
        # Forward kinematics update
        self.mjx_data = mjx.forward(self.mjx_model, self.mjx_data)

class StateSynchronizer:
    """Synchronizes state between real robot and MJX simulation"""
    
    def __init__(self, config: IntegrationConfig):
        self.config = config
        self.robot_interface = None
        self.last_sync_time = 0
        
    def setup_robot_interface(self, robot_ip: str, robot_port: int):
        """Setup connection to real robot"""
        try:
            # Setup ZMQ connection for robot state
            context = zmq.Context()
            self.robot_socket = context.socket(zmq.REQ)
            self.robot_socket.connect(f"tcp://{robot_ip}:{robot_port}")
            self.robot_socket.setsockopt(zmq.RCVTIMEO, 100)  # 100ms timeout
            
            logger.info(f"Robot interface connected: {robot_ip}:{robot_port}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to setup robot interface: {e}")
            return False
    
    def get_robot_state(self) -> Optional[Dict[str, Any]]:
        """Get current robot state"""
        try:
            # Request robot state
            self.robot_socket.send_json({"command": "get_state"})
            response = self.robot_socket.recv_json()
            
            return {
                'qpos': np.array(response['joint_positions']),
                'qvel': np.array(response['joint_velocities']),
                'timestamp': response['timestamp']
            }
            
        except zmq.Again:
            # Timeout
            return None
        except Exception as e:
            logger.error(f"Failed to get robot state: {e}")
            return None
    
    def send_robot_command(self, actions: np.ndarray) -> bool:
        """Send command to real robot"""
        try:
            command = {
                "command": "set_actions",
                "actions": actions.tolist(),
                "timestamp": time.time()
            }
            
            self.robot_socket.send_json(command)
            response = self.robot_socket.recv_json()
            
            return response.get('success', False)
            
        except Exception as e:
            logger.error(f"Failed to send robot command: {e}")
            return False
    
    def sync_states(self, mjx_engine: MJXSimulationEngine) -> Dict[str, float]:
        """Synchronize robot and simulation states"""
        robot_state = self.get_robot_state()
        if robot_state is None:
            return {'sync_error': float('inf')}
        
        sim_state = mjx_engine.get_state()
        if not sim_state:
            return {'sync_error': float('inf')}
        
        # Calculate state differences
        qpos_error = np.linalg.norm(robot_state['qpos'] - sim_state['qpos'])
        qvel_error = np.linalg.norm(robot_state['qvel'] - sim_state['qvel'])
        
        # Update simulation if error is large
        if qpos_error > self.config.state_sync_tolerance:
            mjx_engine.set_state(robot_state['qpos'], robot_state['qvel'])
            logger.debug(f"Synchronized sim state: qpos_error={qpos_error:.4f}")
        
        return {
            'qpos_error': qpos_error,
            'qvel_error': qvel_error,
            'sync_time': time.time() - robot_state['timestamp']
        }

class DomainAdapter:
    """Adapts observations between simulation and real robot"""
    
    def __init__(self, config: IntegrationConfig):
        self.config = config
        self.domain_stats = {
            'sim_mean': None,
            'sim_std': None,
            'real_mean': None,
            'real_std': None
        }
        self.adaptation_history = []
    
    def update_statistics(self, sim_obs: np.ndarray, real_obs: np.ndarray):
        """Update domain adaptation statistics"""
        # Update running statistics
        if self.domain_stats['sim_mean'] is None:
            self.domain_stats['sim_mean'] = np.mean(sim_obs)
            self.domain_stats['sim_std'] = np.std(sim_obs)
            self.domain_stats['real_mean'] = np.mean(real_obs)
            self.domain_stats['real_std'] = np.std(real_obs)
        else:
            # Exponential moving average
            alpha = 0.1
            self.domain_stats['sim_mean'] = (1-alpha) * self.domain_stats['sim_mean'] + alpha * np.mean(sim_obs)
            self.domain_stats['sim_std'] = (1-alpha) * self.domain_stats['sim_std'] + alpha * np.std(sim_obs)
            self.domain_stats['real_mean'] = (1-alpha) * self.domain_stats['real_mean'] + alpha * np.mean(real_obs)
            self.domain_stats['real_std'] = (1-alpha) * self.domain_stats['real_std'] + alpha * np.std(real_obs)
    
    def adapt_sim_to_real(self, sim_obs: np.ndarray) -> np.ndarray:
        """Adapt simulation observation for real robot policy"""
        if any(stat is None for stat in self.domain_stats.values()):
            return sim_obs
        
        # Normalize sim observation and denormalize to real domain
        normalized = (sim_obs - self.domain_stats['sim_mean']) / (self.domain_stats['sim_std'] + 1e-8)
        adapted = normalized * self.domain_stats['real_std'] + self.domain_stats['real_mean']
        
        return adapted
    
    def adapt_real_to_sim(self, real_obs: np.ndarray) -> np.ndarray:
        """Adapt real observation for sim-trained policy"""
        if any(stat is None for stat in self.domain_stats.values()):
            return real_obs
        
        # Normalize real observation and denormalize to sim domain
        normalized = (real_obs - self.domain_stats['real_mean']) / (self.domain_stats['real_std'] + 1e-8)
        adapted = normalized * self.domain_stats['sim_std'] + self.domain_stats['sim_mean']
        
        return adapted
    
    def get_domain_gap(self) -> float:
        """Calculate current domain gap"""
        if any(stat is None for stat in self.domain_stats.values()):
            return float('inf')
        
        mean_diff = abs(self.domain_stats['sim_mean'] - self.domain_stats['real_mean'])
        std_diff = abs(self.domain_stats['sim_std'] - self.domain_stats['real_std'])
        
        return (mean_diff + std_diff) / 2

class PolicyExecutor:
    """Executes policies on both simulation and real robot"""
    
    def __init__(self, config: IntegrationConfig):
        self.config = config
        self.policy_fn = None
        self.execution_stats = {
            'sim_actions': [],
            'real_actions': [],
            'latencies': [],
            'domain_gaps': []
        }
    
    def load_policy(self, policy_fn: Callable):
        """Load policy function"""
        self.policy_fn = policy_fn
        logger.info("Policy loaded")
    
    def execute_step(self, 
                    sim_obs: np.ndarray, 
                    real_obs: np.ndarray,
                    adapter: DomainAdapter) -> Tuple[np.ndarray, np.ndarray, Dict[str, float]]:
        """Execute one policy step on both sim and real"""
        if self.policy_fn is None:
            raise RuntimeError("No policy loaded")
        
        start_time = time.time()
        
        # Get actions from policy
        sim_action = self.policy_fn(sim_obs)
        real_action = self.policy_fn(adapter.adapt_real_to_sim(real_obs))
        
        # Adapt real action back to real domain
        real_action_adapted = adapter.adapt_sim_to_real(real_action)
        
        execution_time = (time.time() - start_time) * 1000  # ms
        
        # Update statistics
        self.execution_stats['sim_actions'].append(sim_action)
        self.execution_stats['real_actions'].append(real_action_adapted)
        self.execution_stats['latencies'].append(execution_time)
        self.execution_stats['domain_gaps'].append(adapter.get_domain_gap())
        
        # Keep only recent history
        max_history = 1000
        for key in self.execution_stats:
            if len(self.execution_stats[key]) > max_history:
                self.execution_stats[key] = self.execution_stats[key][-max_history:]
        
        metrics = {
            'execution_latency_ms': execution_time,
            'action_difference': np.linalg.norm(sim_action - real_action_adapted),
            'domain_gap': adapter.get_domain_gap()
        }
        
        return sim_action, real_action_adapted, metrics

class PerformanceMonitor:
    """Monitors system performance and validates real-time constraints"""
    
    def __init__(self, config: IntegrationConfig):
        self.config = config
        self.metrics = {
            'frame_times': [],
            'latencies': [],
            'sync_errors': [],
            'domain_gaps': [],
            'dropped_frames': 0,
            'total_frames': 0
        }
        
    def update_metrics(self, frame_time: float, latency: float, 
                      sync_error: float, domain_gap: float):
        """Update performance metrics"""
        self.metrics['frame_times'].append(frame_time)
        self.metrics['latencies'].append(latency)
        self.metrics['sync_errors'].append(sync_error)
        self.metrics['domain_gaps'].append(domain_gap)
        self.metrics['total_frames'] += 1
        
        # Check if frame was dropped due to latency
        if latency > self.config.max_latency_ms:
            self.metrics['dropped_frames'] += 1
        
        # Keep only recent history
        max_history = 1000
        for key in ['frame_times', 'latencies', 'sync_errors', 'domain_gaps']:
            if len(self.metrics[key]) > max_history:
                self.metrics[key] = self.metrics[key][-max_history:]
    
    def get_performance_report(self) -> Dict[str, Any]:
        """Get comprehensive performance report"""
        if not self.metrics['frame_times']:
            return {'status': 'no_data'}
        
        report = {
            'avg_fps': 1.0 / np.mean(self.metrics['frame_times']) if self.metrics['frame_times'] else 0,
            'avg_latency_ms': np.mean(self.metrics['latencies']),
            'max_latency_ms': np.max(self.metrics['latencies']),
            'avg_sync_error': np.mean(self.metrics['sync_errors']),
            'avg_domain_gap': np.mean(self.metrics['domain_gaps']),
            'frame_drop_rate': self.metrics['dropped_frames'] / max(self.metrics['total_frames'], 1),
            'real_time_capable': np.mean(self.metrics['latencies']) < self.config.max_latency_ms,
            'domain_gap_acceptable': np.mean(self.metrics['domain_gaps']) < self.config.max_domain_gap
        }
        
        return report

class RealTimeMJXRobotIntegration:
    """Main integration system"""
    
    def __init__(self, config: IntegrationConfig):
        self.config = config
        self.sensor_interface = RealTimeSensorInterface(config)
        self.mjx_engine = MJXSimulationEngine(config)
        self.state_synchronizer = StateSynchronizer(config)
        self.domain_adapter = DomainAdapter(config)
        self.policy_executor = PolicyExecutor(config)
        self.performance_monitor = PerformanceMonitor(config)
        
        self.running = False
        self.integration_thread = None
        
    async def initialize(self) -> bool:
        """Initialize all system components"""
        logger.info("Initializing real-time MJX-Robot integration...")
        
        # Initialize sensor interface
        if not self.sensor_interface.start():
            return False
        
        # Initialize MJX simulation
        if not self.mjx_engine.initialize():
            return False
        
        # Setup robot interface
        if not self.state_synchronizer.setup_robot_interface(
            self.config.robot_ip, self.config.robot_port):
            logger.warning("Robot interface not available - running in sim-only mode")
        
        logger.info("System initialization complete")
        return True
    
    def start_integration(self, policy_fn: Callable):
        """Start real-time integration loop"""
        self.policy_executor.load_policy(policy_fn)
        self.running = True
        self.integration_thread = threading.Thread(target=self._integration_loop)
        self.integration_thread.start()
        logger.info("Real-time integration started")
    
    def _integration_loop(self):
        """Main integration loop"""
        frame_count = 0
        last_validation = 0
        
        while self.running:
            loop_start = time.time()
            
            try:
                # Get real sensor data
                depth_data = self.sensor_interface.get_latest_depth()
                if depth_data is None:
                    time.sleep(0.001)  # 1ms sleep
                    continue
                
                # Process depth data
                real_depth = depth_data['data'].astype(np.float32) / 1000.0  # Convert to meters
                real_depth_resized = cv2.resize(real_depth, 
                                              (self.config.render_width, self.config.render_height))
                
                # Get simulation observation
                dummy_actions = jp.zeros((self.config.num_worlds, self.mjx_engine.model.nu))
                sim_rgb, sim_depth = self.mjx_engine.step(dummy_actions)
                sim_depth_np = np.array(sim_depth[0])  # First world
                
                # Update domain adaptation
                self.domain_adapter.update_statistics(sim_depth_np, real_depth_resized)
                
                # Execute policy
                sim_action, real_action, exec_metrics = self.policy_executor.execute_step(
                    sim_depth_np, real_depth_resized, self.domain_adapter
                )
                
                # Send action to real robot
                robot_success = self.state_synchronizer.send_robot_command(real_action)
                
                # Periodic state synchronization
                sync_metrics = {'qpos_error': 0, 'qvel_error': 0, 'sync_time': 0}
                if frame_count % self.config.validation_frequency == 0:
                    sync_metrics = self.state_synchronizer.sync_states(self.mjx_engine)
                
                # Calculate performance metrics
                loop_time = time.time() - loop_start
                latency_ms = exec_metrics['execution_latency_ms']
                sync_error = sync_metrics.get('qpos_error', 0)
                domain_gap = exec_metrics['domain_gap']
                
                # Update performance monitor
                self.performance_monitor.update_metrics(
                    loop_time, latency_ms, sync_error, domain_gap
                )
                
                # Log periodic status
                if frame_count % 30 == 0:  # Every 30 frames
                    report = self.performance_monitor.get_performance_report()
                    logger.info(f"Frame {frame_count}: FPS={report['avg_fps']:.1f}, "
                              f"Latency={report['avg_latency_ms']:.1f}ms, "
                              f"Domain_gap={report['avg_domain_gap']:.3f}")
                
                frame_count += 1
                
                # Maintain target framerate
                target_loop_time = 1.0 / self.config.target_fps
                sleep_time = target_loop_time - loop_time
                if sleep_time > 0:
                    time.sleep(sleep_time)
                
            except Exception as e:
                logger.error(f"Integration loop error: {e}")
                time.sleep(0.01)  # Brief pause before retry
    
    def stop_integration(self):
        """Stop integration and cleanup"""
        self.running = False
        if self.integration_thread:
            self.integration_thread.join()
        
        self.sensor_interface.stop()
        logger.info("Integration stopped")
    
    def get_status(self) -> Dict[str, Any]:
        """Get current system status"""
        performance_report = self.performance_monitor.get_performance_report()
        
        return {
            'running': self.running,
            'performance': performance_report,
            'config': asdict(self.config)
        }

# Example usage and policy
def example_policy(observation: np.ndarray) -> np.ndarray:
    """Example policy - replace with your trained policy"""
    # Simple policy: move based on depth gradients
    obs_flat = observation.flatten()
    action_dim = 6  # Example 6-DOF robot
    
    # Generate some action based on observation
    action = np.tanh(obs_flat[:action_dim]) * 0.1  # Small movements
    
    return action

async def main():
    """Main execution function"""
    # Configuration
    config = IntegrationConfig(
        robot_ip="10.20.0.46",
        mjcf_path="humanoid_robot.xml",  # Your robot MJCF file
        num_worlds=1,
        render_width=64,
        render_height=64,
        max_latency_ms=8.0,
        target_fps=30.0
    )
    
    # Create integration system
    integration = RealTimeMJXRobotIntegration(config)
    
    try:
        # Initialize system
        if not await integration.initialize():
            logger.error("Failed to initialize integration system")
            return
        
        # Start integration with example policy
        integration.start_integration(example_policy)
        
        # Run for demo period
        logger.info("Running integration demo for 30 seconds...")
        await asyncio.sleep(30)
        
        # Get final status
        status = integration.get_status()
        logger.info(f"Final status: {json.dumps(status, indent=2)}")
        
    finally:
        # Cleanup
        integration.stop_integration()

if __name__ == "__main__":
    asyncio.run(main())