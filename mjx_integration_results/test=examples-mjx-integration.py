#!/usr/bin/env python3
"""
Complete MJX-Robot Integration Example

Demonstrates full usage of the real-time MJX-robot integration system
with policy loading, execution, monitoring, and validation.

Usage:
    python complete_example.py --policy policy.pkl --mjcf robot.xml --duration 300
"""

import argparse
import asyncio
import time
import pickle
import numpy as np
import json
import signal
import sys
from pathlib import Path
from typing import Callable, Dict, Any

# Import our integration system
from mjx_real_robot_integration import (
    RealTimeMJXRobotIntegration, IntegrationConfig, 
    PerformanceMonitor, logger
)

class PolicyLoader:
    """Loads and manages different types of policies"""
    
    @staticmethod
    def load_pickle_policy(policy_path: str) -> Callable:
        """Load policy from pickle file"""
        with open(policy_path, 'rb') as f:
            policy = pickle.load(f)
        
        if hasattr(policy, 'predict'):
            # Stable-baselines3 or similar
            return lambda obs: policy.predict(obs, deterministic=True)[0]
        elif hasattr(policy, 'forward'):
            # PyTorch model
            import torch
            policy.eval()
            return lambda obs: policy.forward(torch.from_numpy(obs).float()).detach().numpy()
        elif callable(policy):
            # Direct callable
            return policy
        else:
            raise ValueError(f"Unknown policy type: {type(policy)}")
    
    @staticmethod
    def load_onnx_policy(policy_path: str) -> Callable:
        """Load policy from ONNX model"""
        import onnxruntime as ort
        
        session = ort.InferenceSession(policy_path)
        input_name = session.get_inputs()[0].name
        
        def policy_fn(obs):
            obs_batch = obs.reshape(1, -1).astype(np.float32)
            result = session.run(None, {input_name: obs_batch})
            return result[0].squeeze()
        
        return policy_fn
    
    @staticmethod
    def create_example_policy() -> Callable:
        """Create example policy for demonstration"""
        def example_policy(observation: np.ndarray) -> np.ndarray:
            """
            Example policy that generates smooth walking motions
            Replace with your trained policy
            """
            # Flatten observation
            obs_flat = observation.flatten()
            
            # Simple sinusoidal walking pattern
            t = time.time()
            
            # Generate actions for humanoid robot (23 DOF example)
            actions = np.zeros(23)
            
            # Leg movements (simplified walking)
            leg_freq = 1.0  # Hz
            leg_amplitude = 0.3
            
            # Left leg (joints 3-8)
            actions[3] = leg_amplitude * np.sin(2 * np.pi * leg_freq * t)      # hip_pitch
            actions[6] = leg_amplitude * np.sin(2 * np.pi * leg_freq * t + np.pi/2)  # knee
            
            # Right leg (joints 9-14)  
            actions[9] = leg_amplitude * np.sin(2 * np.pi * leg_freq * t + np.pi)    # hip_pitch
            actions[12] = leg_amplitude * np.sin(2 * np.pi * leg_freq * t + 3*np.pi/2) # knee
            
            # Add small torso stabilization based on observation
            if len(obs_flat) > 10:
                torso_correction = np.clip(obs_flat[:3] * 0.1, -0.1, 0.1)
                actions[:3] = torso_correction
            
            # Arm swinging
            arm_freq = 2.0
            arm_amplitude = 0.2
            actions[15] = arm_amplitude * np.sin(2 * np.pi * arm_freq * t)     # left_shoulder
            actions[19] = arm_amplitude * np.sin(2 * np.pi * arm_freq * t + np.pi) # right_shoulder
            
            return actions
        
        return example_policy

class ExperimentLogger:
    """Logs experiment data and results"""
    
    def __init__(self, experiment_name: str):
        self.experiment_name = experiment_name
        self.log_dir = Path(f"experiments/{experiment_name}_{int(time.time())}")
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        self.data = {
            'performance_history': [],
            'domain_adaptation_history': [],
            'safety_events': [],
            'config': {},
            'start_time': time.time()
        }
        
        logger.info(f"Experiment logging to: {self.log_dir}")
    
    def log_performance(self, performance_data: Dict[str, Any]):
        """Log performance metrics"""
        entry = {
            'timestamp': time.time(),
            **performance_data
        }
        self.data['performance_history'].append(entry)
    
    def log_safety_event(self, event_type: str, details: Dict[str, Any]):
        """Log safety events"""
        entry = {
            'timestamp': time.time(),
            'event_type': event_type,
            'details': details
        }
        self.data['safety_events'].append(entry)
        logger.warning(f"Safety event: {event_type} - {details}")
    
    def save_experiment(self):
        """Save experiment data"""
        experiment_file = self.log_dir / "experiment_data.json"
        with open(experiment_file, 'w') as f:
            json.dump(self.data, f, indent=2, default=str)
        
        logger.info(f"Experiment data saved: {experiment_file}")

class IntegrationController:
    """Main controller for the integration system"""
    
    def __init__(self, args):
        self.args = args
        self.integration = None
        self.experiment_logger = None
        self.running = False
        
        # Setup signal handlers
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
    
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals"""
        logger.info(f"Received signal {signum}, shutting down...")
        self.shutdown()
        sys.exit(0)
    
    async def initialize(self) -> bool:
        """Initialize the system"""
        logger.info("Initializing MJX-Robot Integration System")
        logger.info("=" * 60)
        
        # Create configuration
        config = IntegrationConfig(
            robot_ip=self.args.robot_ip,
            robot_port=self.args.robot_port,
            mjcf_path=self.args.mjcf,
            depth_width=self.args.depth_width,
            depth_height=self.args.depth_height,
            render_width=self.args.render_width,
            render_height=self.args.render_height,
            max_latency_ms=self.args.max_latency,
            target_fps=self.args.target_fps,
            num_worlds=self.args.num_worlds
        )
        
        # Initialize experiment logging
        self.experiment_logger = ExperimentLogger(self.args.experiment_name)
        self.experiment_logger.data['config'] = config.__dict__
        
        # Create integration system
        self.integration = RealTimeMJXRobotIntegration(config)
        
        # Initialize system
        if not await self.integration.initialize():
            logger.error("Failed to initialize integration system")
            return False
        
        logger.info("System initialization complete!")
        return True
    
    def load_policy(self) -> Callable:
        """Load the policy"""
        if self.args.policy:
            policy_path = Path(self.args.policy)
            if not policy_path.exists():
                raise FileNotFoundError(f"Policy file not found: {policy_path}")
            
            if policy_path.suffix == '.pkl':
                logger.info(f"Loading pickle policy: {policy_path}")
                return PolicyLoader.load_pickle_policy(str(policy_path))
            elif policy_path.suffix == '.onnx':
                logger.info(f"Loading ONNX policy: {policy_path}")
                return PolicyLoader.load_onnx_policy(str(policy_path))
            else:
                raise ValueError(f"Unsupported policy format: {policy_path.suffix}")
        else:
            logger.info("Using example policy")
            return PolicyLoader.create_example_policy()
    
    async def run_experiment(self):
        """Run the main experiment"""
        logger.info(f"Starting experiment: {self.args.experiment_name}")
        logger.info(f"Duration: {self.args.duration} seconds")
        
        # Load policy
        policy = self.load_policy()
        
        # Start integration
        self.integration.start_integration(policy)
        self.running = True
        
        # Monitoring loop
        start_time = time.time()
        last_log_time = start_time
        last_safety_check = start_time
        
        try:
            while self.running and (time.time() - start_time) < self.args.duration:
                current_time = time.time()
                
                # Log performance every 5 seconds
                if current_time - last_log_time >= 5.0:
                    status = self.integration.get_status()
                    performance = status['performance']
                    
                    self.experiment_logger.log_performance(performance)
                    
                    logger.info(f"Time: {current_time - start_time:.1f}s | "
                              f"FPS: {performance['avg_fps']:.1f} | "
                              f"Latency: {performance['avg_latency_ms']:.1f}ms | "
                              f"Domain Gap: {performance['avg_domain_gap']:.3f} | "
                              f"RT Capable: {performance['real_time_capable']}")
                    
                    last_log_time = current_time
                
                # Safety monitoring every second
                if current_time - last_safety_check >= 1.0:
                    await self._check_safety()
                    last_safety_check = current_time
                
                # Check for early termination conditions
                if self._should_terminate():
                    logger.warning("Early termination triggered")
                    break
                
                await asyncio.sleep(0.1)  # 100ms monitoring loop
                
        except Exception as e:
            logger.error(f"Experiment error: {e}")
            self.experiment_logger.log_safety_event("experiment_error", {"error": str(e)})
        
        finally:
            # Clean shutdown
            self.shutdown()
    
    async def _check_safety(self):
        """Check system safety conditions"""
        status = self.integration.get_status()
        performance = status['performance']
        
        # Check real-time capability
        if not performance['real_time_capable']:
            self.experiment_logger.log_safety_event(
                "real_time_violation",
                {"avg_latency_ms": performance['avg_latency_ms']}
            )
        
        # Check domain gap
        if performance['avg_domain_gap'] > self.integration.config.max_domain_gap:
            self.experiment_logger.log_safety_event(
                "domain_gap_exceeded",
                {"domain_gap": performance['avg_domain_gap']}
            )
        
        # Check frame drop rate
        if performance['frame_drop_rate'] > 0.1:  # 10% threshold
            self.experiment_logger.log_safety_event(
                "high_frame_drops",
                {"drop_rate": performance['frame_drop_rate']}
            )
    
    def _should_terminate(self) -> bool:
        """Check if experiment should terminate early"""
        status = self.integration.get_status()
        performance = status['performance']
        
        # Terminate if system becomes non-real-time consistently
        if (len(self.experiment_logger.data['performance_history']) > 10 and
            all(not entry['real_time_capable'] 
                for entry in self.experiment_logger.data['performance_history'][-10:])):
            return True
        
        # Terminate if too many safety events
        if len(self.experiment_logger.data['safety_events']) > 50:
            return True
        
        return False
    
    def shutdown(self):
        """Shutdown the system"""
        if self.running:
            logger.info("Shutting down integration system...")
            self.running = False
            
            if self.integration:
                self.integration.stop_integration()
            
            if self.experiment_logger:
                self.experiment_logger.save_experiment()
                self._generate_report()
    
    def _generate_report(self):
        """Generate experiment report"""
        if not self.experiment_logger.data['performance_history']:
            return
        
        # Calculate summary statistics
        performance_data = self.experiment_logger.data['performance_history']
        
        avg_fps = np.mean([entry['avg_fps'] for entry in performance_data])
        avg_latency = np.mean([entry['avg_latency_ms'] for entry in performance_data])
        avg_domain_gap = np.mean([entry['avg_domain_gap'] for entry in performance_data])
        real_time_percent = np.mean([entry['real_time_capable'] for entry in performance_data]) * 100
        
        # Generate report
        report = f"""
Experiment Report: {self.args.experiment_name}
{'=' * 60}
Duration: {time.time() - self.experiment_logger.data['start_time']:.1f} seconds
Policy: {self.args.policy or 'example_policy'}
Robot Model: {self.args.mjcf}

Performance Summary:
• Average FPS: {avg_fps:.1f}
• Average Latency: {avg_latency:.1f} ms
• Average Domain Gap: {avg_domain_gap:.3f}
• Real-time Capable: {real_time_percent:.1f}%

Safety Events: {len(self.experiment_logger.data['safety_events'])}
Data Points: {len(performance_data)}

Results: {'SUCCESS' if real_time_percent > 90 else 'NEEDS_OPTIMIZATION'}
"""
        
        # Save report
        report_file = self.experiment_logger.log_dir / "experiment_report.txt"
        with open(report_file, 'w') as f:
            f.write(report)
        
        print(report)
        logger.info(f"Experiment report saved: {report_file}")

def main():
    """Main execution function"""
    parser = argparse.ArgumentParser(description="MJX-Robot Integration Example")
    
    # Required arguments
    parser.add_argument('--mjcf', type=str, required=True,
                       help='Path to robot MJCF file')
    
    # Optional arguments
    parser.add_argument('--policy', type=str, default=None,
                       help='Path to policy file (.pkl or .onnx)')
    parser.add_argument('--duration', type=int, default=300,
                       help='Experiment duration in seconds')
    parser.add_argument('--experiment-name', type=str, default='mjx_robot_test',
                       help='Name for this experiment')
    
    # System configuration
    parser.add_argument('--robot-ip', type=str, default='10.20.0.46',
                       help='Robot computer IP address')
    parser.add_argument('--robot-port', type=int, default=5555,
                       help='Robot communication port')
    parser.add_argument('--depth-width', type=int, default=640,
                       help='Camera depth width')
    parser.add_argument('--depth-height', type=int, default=480,
                       help='Camera depth height')
    parser.add_argument('--render-width', type=int, default=64,
                       help='MJX render width')
    parser.add_argument('--render-height', type=int, default=64,
                       help='MJX render height')
    parser.add_argument('--max-latency', type=float, default=8.0,
                       help='Maximum acceptable latency (ms)')
    parser.add_argument('--target-fps', type=float, default=30.0,
                       help='Target control frequency')
    parser.add_argument('--num-worlds', type=int, default=1,
                       help='Number of MJX simulation worlds')
    
    args = parser.parse_args()
    
    # Validate arguments
    if not Path(args.mjcf).exists():
        print(f"Error: MJCF file not found: {args.mjcf}")
        sys.exit(1)
    
    if args.policy and not Path(args.policy).exists():
        print(f"Error: Policy file not found: {args.policy}")
        sys.exit(1)
    
    # Print configuration
    print("MJX-Robot Integration System")
    print("=" * 60)
    print(f"Experiment: {args.experiment_name}")
    print(f"Duration: {args.duration}s")
    print(f"Robot Model: {args.mjcf}")
    print(f"Policy: {args.policy or 'example_policy'}")
    print(f"Robot IP: {args.robot_ip}:{args.robot_port}")
    print(f"Target Performance: {args.target_fps} FPS, <{args.max_latency}ms latency")
    print("=" * 60)
    
    # Run experiment
    controller = IntegrationController(args)
    
    async def run():
        if await controller.initialize():
            await controller.run_experiment()
        else:
            print("System initialization failed")
            sys.exit(1)
    
    try:
        asyncio.run(run())
        print("Experiment completed successfully")
    except KeyboardInterrupt:
        print("\nExperiment interrupted by user")
    except Exception as e:
        print(f"Experiment failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()