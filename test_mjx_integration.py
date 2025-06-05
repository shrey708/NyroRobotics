#!/usr/bin/env python3
"""
Test Suite for MJX Depth Integration
Tests the core functionality without requiring MJX dependencies
"""

import unittest
import numpy as np
import tempfile
import os
import shutil

# Import our modules
from mjx_depth_integration import MJXDepthProcessor, MJXBatchRenderer
from simplified_depth_processing import SimplifiedDepthProcessor, DepthProcessingConfig
from mjx_integration_demo import MJXIntegrationDemo

class TestMJXDepthProcessor(unittest.TestCase):
    """Test cases for MJX Depth Processor"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.processor = MJXDepthProcessor()
        self.test_depth = np.random.uniform(0.5, 2.0, (240, 320))
        self.test_depth[self.test_depth < 0.8] = 0  # Add some holes
    
    def test_initialization(self):
        """Test processor initialization"""
        self.assertEqual(self.processor.target_resolution, (48, 32))
        self.assertEqual(self.processor.max_depth, 2.0)
        self.assertEqual(self.processor.min_depth, 0.15)
    
    def test_process_for_training(self):
        """Test the complete processing pipeline"""
        processed = self.processor.process_for_training(self.test_depth)
        
        # Check output shape
        self.assertEqual(processed.shape, self.processor.target_resolution)
        
        # Check depth range
        valid_pixels = processed[processed < self.processor.max_depth]
        if len(valid_pixels) > 0:
            self.assertGreaterEqual(np.min(valid_pixels), 0)
            self.assertLessEqual(np.max(valid_pixels), self.processor.max_depth)

class TestSimplifiedDepthProcessor(unittest.TestCase):
    """Test cases for Simplified Depth Processor"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.config = DepthProcessingConfig()
        self.processor = SimplifiedDepthProcessor(self.config)
        self.test_depth = np.random.uniform(0.5, 2.0, (240, 320))
    
    def test_initialization(self):
        """Test processor initialization"""
        self.assertEqual(self.processor.config.target_resolution, (48, 32))
        self.assertEqual(self.processor.frame_count, 0)
    
    def test_process_simulated_depth(self):
        """Test complete simulated depth processing"""
        processed = self.processor.process_simulated_depth(self.test_depth)
        
        # Check output shape
        self.assertEqual(processed.shape, self.config.target_resolution)
        
        # Check that processing occurred (coverage should be reduced)
        processed_coverage = np.sum(processed < self.config.max_depth) / processed.size
        self.assertLess(processed_coverage, 1.0)  # Should have some holes/degradation

class TestMJXIntegrationDemo(unittest.TestCase):
    """Test cases for MJX Integration Demo"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.demo = MJXIntegrationDemo()
    
    def test_initialization(self):
        """Test demo initialization"""
        self.assertEqual(self.demo.target_resolution, (48, 32))
        self.assertIsNotNone(self.demo.depth_processor)
    
    def test_create_synthetic_robot_data(self):
        """Test synthetic robot data creation"""
        synthetic_data = self.demo.create_synthetic_robot_data()
        
        self.assertEqual(len(synthetic_data), 8)
        
        for data in synthetic_data:
            self.assertIn('data', data)
            self.assertIn('filename', data)
            self.assertEqual(data['source'], 'synthetic')
            self.assertEqual(len(data['data'].shape), 2)  # 2D depth image

def run_tests():
    """Run all tests"""
    print("🧪 Running MJX Integration Test Suite")
    print("=" * 50)
    
    # Create test suite
    test_suite = unittest.TestSuite()
    
    # Add test classes
    test_classes = [
        TestMJXDepthProcessor,
        TestSimplifiedDepthProcessor,
        TestMJXIntegrationDemo
    ]
    
    for test_class in test_classes:
        tests = unittest.TestLoader().loadTestsFromTestCase(test_class)
        test_suite.addTests(tests)
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)
    
    # Print summary
    print("\n📊 Test Results Summary")
    print("=" * 30)
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    
    if result.wasSuccessful():
        print("\n✅ All tests passed!")
        return True
    else:
        print("\n❌ Some tests failed!")
        return False

if __name__ == "__main__":
    success = run_tests()
    exit(0 if success else 1) 