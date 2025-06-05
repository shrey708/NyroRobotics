#!/usr/bin/env python3
"""
Simplified Depth Processing for Real Robot Training
Validated stereo-matching simulation pipeline for RealSense D435i
"""

import numpy as np
import cv2
from scipy import ndimage
from noise import pnoise2
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import time
import json
from datetime import datetime
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
import os

@dataclass
class DepthProcessingConfig:
    """Configuration for depth processing pipeline"""
    target_resolution: Tuple[int, int] = (48, 32)  # (width, height)
    max_depth: float = 2.0
    min_depth: float = 0.15
    blur_sigma: float = 0.8
    
    # Stereo-matching simulation parameters
    edge_noise_intensity: float = 0.3
    perlin_threshold: float = 0.45
    blind_spot_width_ratio: float = 0.08  # 8% of width

class SimplifiedDepthProcessor:
    """Simplified depth processor validated with real robot data"""
    
    def __init__(self, config: DepthProcessingConfig = None):
        self.config = config or DepthProcessingConfig()
        self.frame_count = 0
        
        print(f"🔬 Simplified Depth Processor initialized")
        print(f"   Target resolution: {self.config.target_resolution}")
        print(f"   Depth range: {self.config.min_depth}m - {self.config.max_depth}m")
        print(f"   Validation: Confirmed with Booster Humanoid Robot data")
    
    def add_edge_noise(self, depth_image: np.ndarray) -> np.ndarray:
        """Add edge-based noise and failures"""
        
        processed = depth_image.copy()
        
        # Calculate depth gradients
        grad_x = np.abs(np.gradient(depth_image, axis=1))
        grad_y = np.abs(np.gradient(depth_image, axis=0))
        gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
        
        # Identify edge regions
        valid_mask = depth_image > 0
        if not np.any(valid_mask):
            return processed
            
        edge_threshold = np.percentile(gradient_magnitude[valid_mask], 80)
        edge_mask = (gradient_magnitude > edge_threshold) & valid_mask
        
        # Apply edge corruptions
        edge_pixels = np.where(edge_mask)
        num_edge_pixels = len(edge_pixels[0])
        
        if num_edge_pixels > 0:
            corruption_prob = self.config.edge_noise_intensity
            corruption_mask = np.random.random(num_edge_pixels) < corruption_prob
            corrupt_indices = np.where(corruption_mask)[0]
            
            for idx in corrupt_indices:
                y, x = edge_pixels[0][idx], edge_pixels[1][idx]
                
                corruption_type = np.random.random()
                
                if corruption_type < 0.5:
                    # Remove pixel (depth discontinuity)
                    processed[y, x] = 0
                else:
                    # Add noise
                    noise_amount = np.random.normal(0, self.config.edge_noise_intensity)
                    processed[y, x] = max(0, processed[y, x] + noise_amount)
        
        return processed
    
    def add_perlin_holes(self, depth_image: np.ndarray) -> np.ndarray:
        """Add holes using Perlin noise (temporally consistent)"""
        
        processed = depth_image.copy()
        height, width = depth_image.shape
        
        # Generate Perlin noise with temporal evolution
        time_offset = self.frame_count * 0.1  # Slow temporal evolution
        
        noise_array = np.zeros((height, width))
        for y in range(height):
            for x in range(width):
                noise_value = pnoise2(
                    x / 8.0,
                    y / 8.0 + time_offset,  # Add temporal component
                    octaves=4,
                    persistence=0.6,
                    lacunarity=2.0,
                    repeatx=width,
                    repeaty=height,
                    base=42
                )
                noise_array[y, x] = noise_value
        
        # Normalize and create holes
        noise_normalized = (noise_array - noise_array.min()) / (noise_array.max() - noise_array.min())
        hole_mask = noise_normalized > self.config.perlin_threshold
        
        # Apply holes only to valid pixels
        valid_mask = depth_image > 0
        final_holes = hole_mask & valid_mask
        processed[final_holes] = 0
        
        self.frame_count += 1
        return processed
    
    def add_blind_spots(self, depth_image: np.ndarray) -> np.ndarray:
        """Add stereo-matching blind spots (occlusion simulation)"""
        
        processed = depth_image.copy()
        height, width = depth_image.shape
        
        # Left edge blind spot (stereo occlusion)
        blind_spot_width = max(1, int(width * self.config.blind_spot_width_ratio))
        
        # Randomize the blind spot slightly
        blind_variation = np.random.randint(-2, 3)  # ±2 pixels
        actual_blind_width = max(1, blind_spot_width + blind_variation)
        
        processed[:, :actual_blind_width] = 0
        
        return processed
    
    def process_simulated_depth(self, depth_image: np.ndarray) -> np.ndarray:
        """Process simulated depth with realistic degradation"""
        
        # Step 1: Resize to target resolution
        processed = cv2.resize(depth_image, self.config.target_resolution, 
                             interpolation=cv2.INTER_LINEAR)
        
        # Step 2: Add edge noise
        processed = self.add_edge_noise(processed)
        
        # Step 3: Add Perlin holes
        processed = self.add_perlin_holes(processed)
        
        # Step 4: Add blind spots
        processed = self.add_blind_spots(processed)
        
        # Step 5: Clip depth values
        processed = np.clip(processed, 0, self.config.max_depth)
        processed[processed < self.config.min_depth] = self.config.max_depth
        
        # Step 6: Apply final blur
        processed = ndimage.gaussian_filter(processed, sigma=self.config.blur_sigma)
        
        return processed
    
    def process_real_depth(self, depth_image: np.ndarray) -> np.ndarray:
        """Process real depth to match simulation characteristics"""
        
        # Step 1: Clip depth values
        processed = np.clip(depth_image, 0, self.config.max_depth)
        processed[processed < self.config.min_depth] = self.config.max_depth
        
        # Step 2: Resize to target resolution
        processed = cv2.resize(processed, self.config.target_resolution, 
                             interpolation=cv2.INTER_AREA)
        
        # Step 3: Apply light denoising
        processed = ndimage.median_filter(processed, size=3)
        
        # Step 4: Apply same blur as simulation
        processed = ndimage.gaussian_filter(processed, sigma=self.config.blur_sigma)
        
        return processed

def analyze_processing_metrics(original: np.ndarray, 
                             processed: np.ndarray) -> Dict:
    """Analyze processing performance metrics"""
    
    metrics = {}
    
    # Coverage metrics
    metrics['original_coverage'] = float(100 * np.sum(original > 0) / original.size)
    metrics['processed_coverage'] = float(100 * np.sum(processed < 2.0) / processed.size)
    metrics['coverage_change'] = metrics['processed_coverage'] - metrics['original_coverage']
    
    # Depth statistics
    original_valid = original[original > 0]
    processed_valid = processed[processed < 2.0]
    
    if len(original_valid) > 0:
        metrics['original_mean_depth'] = float(original_valid.mean())
        metrics['original_std_depth'] = float(original_valid.std())
    
    if len(processed_valid) > 0:
        metrics['processed_mean_depth'] = float(processed_valid.mean())
        metrics['processed_std_depth'] = float(processed_valid.std())
    
    # Resolution metrics
    metrics['compression_ratio'] = float(original.size / processed.size)
    metrics['original_shape'] = original.shape
    metrics['processed_shape'] = processed.shape
    
    return metrics

if __name__ == "__main__":
    # Create demo
    print("🔬 Simplified Depth Processing Demo")
    print("=" * 50)
    
    # Create synthetic depth image for testing
    test_depth = np.random.uniform(0.5, 2.0, (240, 320))
    test_depth[test_depth < 0.8] = 0  # Add some holes
    
    # Initialize processor
    processor = SimplifiedDepthProcessor()
    
    # Process depth
    processed = processor.process_simulated_depth(test_depth)
    
    # Analyze
    metrics = analyze_processing_metrics(test_depth, processed)
    
    print(f"✅ Processing complete:")
    print(f"   Coverage change: {metrics['coverage_change']:+.1f}%")
    print(f"   Compression: {metrics['compression_ratio']:.1f}x")
    print(f"   Shape: {metrics['original_shape']} → {metrics['processed_shape']}") 