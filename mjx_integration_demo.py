#!/usr/bin/env python3
"""
MJX Integration Demo - Simulated Pipeline
Demonstrates the MJX depth integration without requiring actual MJX dependencies
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import cv2
import time
import json
from datetime import datetime
import os
from typing import Dict, List, Tuple

# Import our core processors
from mjx_depth_integration import MJXDepthProcessor
from simplified_depth_processing import SimplifiedDepthProcessor, DepthProcessingConfig

class MJXIntegrationDemo:
    """Demo of MJX integration using simulated data"""
    
    def __init__(self):
        self.target_resolution = (48, 32)
        self.depth_processor = MJXDepthProcessor(self.target_resolution)
        self.simplified_processor = SimplifiedDepthProcessor()
        
        print("🎮 MJX Integration Demo initialized")
        print(f"   Target resolution: {self.target_resolution}")
    
    def create_synthetic_robot_data(self) -> List[Dict]:
        """Create synthetic robot depth data for demo"""
        
        print("🤖 Creating synthetic robot depth data...")
        
        synthetic_data = []
        
        for i in range(8):
            # Create realistic depth pattern
            height, width = 480, 640
            depth = np.zeros((height, width))
            
            # Add ground plane
            y_coords, x_coords = np.ogrid[:height, :width]
            ground_depth = 1.2 + (y_coords - height//2) * 0.001
            depth = np.maximum(depth, ground_depth)
            
            # Add some objects
            for obj_id in range(3):
                center_y = height//2 + np.random.randint(-100, 100)
                center_x = width//2 + np.random.randint(-200, 200)
                radius = 30 + obj_id * 15
                
                object_mask = ((y_coords - center_y) ** 2 + (x_coords - center_x) ** 2) < radius ** 2
                object_depth = 0.8 + obj_id * 0.3 + i * 0.1
                depth[object_mask] = object_depth
            
            # Add noise
            noise = np.random.normal(0, 0.02, (height, width))
            depth += noise
            depth = np.clip(depth, 0, 3.0)
            
            # Set some pixels to 0 (no depth)
            mask = np.random.random((height, width)) < 0.2
            depth[mask] = 0
            
            synthetic_data.append({
                'data': depth,
                'filename': f'synthetic_robot_frame_{i+1}.npy',
                'source': 'synthetic'
            })
        
        print(f"   ✅ Created {len(synthetic_data)} synthetic depth frames")
        return synthetic_data
    
    def simulate_mjx_worlds(self, num_worlds: int = 4) -> List[np.ndarray]:
        """Simulate MJX rendering multiple worlds"""
        
        print(f"🎨 Simulating {num_worlds} MJX worlds...")
        
        world_depths = []
        
        for world_id in range(num_worlds):
            # Create simulated world depth
            height, width = 320, 240
            depth = np.zeros((height, width))
            
            # Create different scene for each world
            y_coords, x_coords = np.ogrid[:height, :width]
            
            # Central objects with different depths per world
            center_y, center_x = height // 2, width // 2
            
            # Primary object
            radius1 = 40 + world_id * 5
            object1_mask = ((y_coords - center_y) ** 2 + (x_coords - center_x) ** 2) < radius1 ** 2
            depth[object1_mask] = 0.6 + world_id * 0.2
            
            # Secondary object
            offset_x = 60 + world_id * 10
            object2_mask = ((y_coords - center_y + 30) ** 2 + (x_coords - center_x + offset_x) ** 2) < 25 ** 2
            depth[object2_mask] = 1.0 + world_id * 0.15
            
            # Background
            background_mask = depth == 0
            depth[background_mask] = 1.8 + world_id * 0.05
            
            # Add realistic noise
            noise = np.random.normal(0, 0.01, (height, width))
            depth += noise
            depth = np.clip(depth, 0.3, 2.0)
            
            world_depths.append(depth)
        
        return world_depths
    
    def run_integration_step(self, robot_data: List[Dict], step_id: int) -> Dict:
        """Run one integration step"""
        
        print(f"\n📊 Running integration step {step_id}")
        
        step_start_time = time.time()
        
        # Select robot data for this step
        num_worlds = min(4, len(robot_data))
        selected_data = robot_data[:num_worlds]
        
        # Simulate MJX worlds
        mjx_worlds = self.simulate_mjx_worlds(num_worlds)
        
        # Process each world
        world_results = []
        
        for world_id in range(num_worlds):
            # Get robot data and MJX simulation
            robot_frame = selected_data[world_id]['data']
            mjx_depth = mjx_worlds[world_id]
            
            # Process both through our pipeline
            processed_robot = self.depth_processor.process_for_training(robot_frame)
            processed_mjx = self.depth_processor.process_for_training(mjx_depth)
            
            # Calculate metrics
            robot_coverage = 100 * np.sum(processed_robot < 2.0) / processed_robot.size
            mjx_coverage = 100 * np.sum(processed_mjx < 2.0) / processed_mjx.size
            
            world_result = {
                'world_id': world_id,
                'robot_source': selected_data[world_id]['source'],
                'robot_coverage': float(robot_coverage),
                'mjx_coverage': float(mjx_coverage),
                'robot_mean_depth': float(np.mean(processed_robot[processed_robot < 2.0])),
                'mjx_mean_depth': float(np.mean(processed_mjx[processed_mjx < 2.0])),
                'original_robot_shape': robot_frame.shape,
                'processed_shape': processed_robot.shape
            }
            
            world_results.append(world_result)
            
            print(f"   World {world_id}: Robot coverage {robot_coverage:.1f}%, MJX coverage {mjx_coverage:.1f}%")
        
        step_time = time.time() - step_start_time
        
        # Calculate step statistics
        avg_coverage_loss = np.mean([
            100 - wr['robot_coverage'] for wr in world_results
        ])
        
        step_result = {
            'step': step_id,
            'processing_time': step_time,
            'num_worlds': num_worlds,
            'world_results': world_results,
            'avg_coverage_loss': float(avg_coverage_loss),
            'avg_robot_coverage': float(np.mean([wr['robot_coverage'] for wr in world_results])),
            'avg_mjx_coverage': float(np.mean([wr['mjx_coverage'] for wr in world_results]))
        }
        
        print(f"   ✅ Step {step_id} complete: {step_time:.3f}s, avg coverage loss: {avg_coverage_loss:.1f}%")
        
        return step_result
    
    def run_complete_demo(self) -> Dict:
        """Run the complete MJX integration demo"""
        
        print("🚀 STARTING MJX INTEGRATION DEMO")
        print("=" * 60)
        
        total_start_time = time.time()
        
        # Create synthetic robot data
        robot_data = self.create_synthetic_robot_data()
        print(f"📊 Using {len(robot_data)} synthetic robot depth frames")
        
        # Run integration steps
        results = []
        
        for step in range(2):  # Run 2 simulation steps
            step_result = self.run_integration_step(robot_data, step + 1)
            results.append(step_result)
        
        total_time = time.time() - total_start_time
        
        # Calculate summary statistics
        total_worlds = sum(r['num_worlds'] for r in results)
        avg_step_time = np.mean([r['processing_time'] for r in results])
        avg_coverage_loss = np.mean([r['avg_coverage_loss'] for r in results])
        
        # Prepare summary
        demo_summary = {
            'timestamp': datetime.now().strftime("%Y%m%d_%H%M%S"),
            'total_processing_time': total_time,
            'steps': len(results),
            'total_worlds': total_worlds,
            'avg_step_time': avg_step_time,
            'avg_coverage_loss': avg_coverage_loss,
            'robot_data_sources': [d['source'] for d in robot_data],
            'step_results': results,
            'performance_ready': avg_step_time < 0.1,
            'status': 'demo_complete'
        }
        
        # Save summary
        os.makedirs('mjx_integration_results', exist_ok=True)
        summary_filename = f'mjx_integration_results/mjx_integration_demo_summary_{demo_summary["timestamp"]}.json'
        
        with open(summary_filename, 'w') as f:
            json.dump(demo_summary, f, indent=2)
        
        print(f"\n🎉 MJX INTEGRATION DEMO COMPLETE!")
        print("=" * 60)
        print(f"📊 Processed {len(results)} steps across {total_worlds} worlds")
        print(f"⚡ Average step time: {avg_step_time:.3f}s")
        print(f"📉 Average coverage loss: {avg_coverage_loss:.1f}%")
        print(f"📝 Summary: {summary_filename}")
        print(f"✅ Ready for RL training: {'Yes' if demo_summary['performance_ready'] else 'Needs optimization'}")
        print("=" * 60)
        
        return demo_summary

if __name__ == "__main__":
    # Create and run demo
    demo = MJXIntegrationDemo()
    results = demo.run_complete_demo() 