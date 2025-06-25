#!/usr/bin/env python3
"""
Test script for dual detection method (position + color histogram)
"""

import cv2
import numpy as np
from car_detector import CarDetector

def test_dual_detection():
    """Test the dual detection method with a sample image"""
    
    # Initialize detector
    detector = CarDetector("yolov9c.pt", show_timer=True)
    
    # Create a test image (or load from file)
    test_image = np.zeros((480, 640, 3), dtype=np.uint8)
    
    # Simulate a car bounding box
    bbox1 = [100, 100, 200, 150]  # First position
    bbox2 = [105, 105, 205, 155]  # Slightly moved (should be stationary)
    bbox3 = [150, 150, 250, 200]  # Moved significantly (should be moving)
    
    # Add some color to the test image
    cv2.rectangle(test_image, (bbox1[0], bbox1[1]), (bbox1[2], bbox1[3]), (0, 255, 0), -1)
    
    print("Testing dual detection method...")
    print("=" * 50)
    
    # Test 1: First detection (new car)
    print("Test 1: First detection")
    detector.update_car_timer(1, bbox1, None, test_image)
    print(f"Car 1 added to timers: {1 in detector.car_timers}")
    
    # Test 2: Slight movement (should be stationary)
    print("\nTest 2: Slight movement")
    detector.update_car_timer(1, bbox2, None, test_image)
    car_data = detector.car_timers[1]
    print(f"Stationary frames: {car_data['stationary_frames']}")
    print(f"Is parked: {car_data['is_parked']}")
    
    # Test 3: Significant movement (should reset timer)
    print("\nTest 3: Significant movement")
    detector.update_car_timer(1, bbox3, None, test_image)
    car_data = detector.car_timers[1]
    print(f"Stationary frames: {car_data['stationary_frames']}")
    print(f"Is parked: {car_data['is_parked']}")
    
    # Test detection method info
    print("\nTest 4: Detection method info")
    info = detector.get_detection_method_info(1, test_image, bbox2)
    if isinstance(info, dict):
        print(f"Position distance: {info['position_distance']:.2f}px")
        print(f"Histogram similarity: {info['histogram_similarity']:.3f}")
        print(f"Position stationary: {info['position_stationary']}")
        print(f"Histogram stationary: {info['histogram_stationary']}")
        print(f"Combined stationary: {info['combined_stationary']}")
    
    print("\nDual detection test completed!")

if __name__ == "__main__":
    test_dual_detection() 