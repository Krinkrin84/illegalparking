"""
Configuration file for Car Detection and Parking Timer System

This file contains all configuration variables and argument parsing logic.
"""

import argparse
import cv2

# ===== CORE PARKING SYSTEM CONFIGURATION =====
# These parameters directly affect the car parking detection system

# Color variation detection (HSV histogram analysis)
COLOR_CHANGE_THRESHOLD = 0.15  # Lower = more sensitive to color changes
COLOR_POSITION_RATIO = 0.4     # Ratio of color vs position detection (0.0-1.0)
COLOR_CHANGE_SMOOTHING = 0.9   # Higher = more smoothing

# Parking timer settings
ALLOWED_PARKING_TIME = 20      # seconds - time limit before illegal parking
POSITION_THRESHOLD = 20        # Max movement in pixels to consider a car stationary
PARKING_THRESHOLD = 30         # frames to wait before counting as parked
OCCLUSION_TOLERANCE_FRAMES = 60 # frames to keep timer data for occluded cars

# ===== OTHER CONFIGURATION =====
# These parameters affect detection and tracking but not core parking logic

# YOLO configuration
CONFIDENCE_THRESHOLD = 0.35    # minimum confidence for car detection
IOU_THRESHOLD = 0.7            # IoU threshold for NMS
DEFAULT_MODEL = "yolov9c.pt"   # default YOLO model file

# ByteTracker configuration
TRACK_THRESHOLD = 0.5          # tracking confidence threshold
TRACK_BUFFER = 120             # frames to keep lost tracks
MATCH_THRESHOLD = 0.8          # matching threshold for tracking
MIN_BOX_AREA = 10              # minimum box area to track

# Tracking stability configuration
TRACK_STABILITY_BUFFER = 30    # frames to maintain track before assigning stable ID
TRACK_CONFIRMATION_FRAMES = 5  # minimum frames to confirm a track
TRACK_REAPPEAR_THRESHOLD = 0.7 # similarity threshold for track reappearance
TRACK_POSITION_SMOOTHING = 0.8 # smoothing factor for track position
TRACK_SIZE_SMOOTHING = 0.9     # smoothing factor for track size

# Color histogram configuration
HISTOGRAM_THRESHOLD = 0.85     # Similarity threshold for color histogram comparison
HISTOGRAM_BINS = 32            # Number of bins for color histogram
HISTOGRAM_METHOD = cv2.HISTCMP_CORREL  # Histogram comparison method

# Purpose: Whether to use last frame's position for histogram comparison to isolate color changes
USE_LAST_POSITION_FOR_HISTOGRAM = False  # True = compare same position, False = compare current position


class TrackerArgs:
    """Simple args class for ByteTracker"""
    def __init__(self, track_thresh=TRACK_THRESHOLD, track_buffer=TRACK_BUFFER, 
                 match_thresh=MATCH_THRESHOLD, mot20=False):
        self.track_thresh = track_thresh
        self.track_buffer = track_buffer
        self.match_thresh = match_thresh
        self.mot20 = mot20
        # These are not used by the tracker but are required by the demo code
        self.aspect_ratio_thresh = 1.6
        self.min_box_area = MIN_BOX_AREA


def parse_arguments():
    """Parse command line arguments for car detection"""
    parser = argparse.ArgumentParser(description="Car Detection using YOLO with ByteTracker")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL, help="YOLO model file")
    parser.add_argument("--input", type=str, required=True, help="Input image or video")
    parser.add_argument("--output", type=str, default=None, help="Output file path")
    parser.add_argument("--conf", type=float, default=CONFIDENCE_THRESHOLD, help="Confidence threshold")
    parser.add_argument("--iou", type=float, default=IOU_THRESHOLD, help="IoU threshold")
    parser.add_argument("--parking-threshold", type=int, default=PARKING_THRESHOLD, help="Frames to wait before counting as parked")
    parser.add_argument("--track-buffer", type=int, default=TRACK_BUFFER, help="Number of frames to keep a lost track.")
    parser.add_argument("--no-color-hist", action="store_true", help="Disable color histogram comparison")
    
    return parser.parse_args()


def print_configuration():
    """Print current configuration settings"""
    print(f"================Configuration===================")
    print(f"Core Parking System Configuration:")
    print(f"  Color change threshold: {COLOR_CHANGE_THRESHOLD:.3f}")
    print(f"  Color/Position ratio: {COLOR_POSITION_RATIO:.2f}")
    print(f"  Color smoothing: {COLOR_CHANGE_SMOOTHING:.2f}")
    print(f"  Allowed parking time: {ALLOWED_PARKING_TIME} seconds")
    print(f"  Position threshold: {POSITION_THRESHOLD} pixels")
    print(f"  Parking threshold: {PARKING_THRESHOLD} frames")
    print(f"  Occlusion tolerance: {OCCLUSION_TOLERANCE_FRAMES} frames")
    print(f"")
    print(f"Other Configuration:")
    print(f"  YOLO confidence threshold: {CONFIDENCE_THRESHOLD}")
    print(f"  YOLO IoU threshold: {IOU_THRESHOLD}")
    print(f"  Default model: {DEFAULT_MODEL}")
    print(f"  Track buffer: {TRACK_BUFFER} frames")
    print(f"  Track stability buffer: {TRACK_STABILITY_BUFFER} frames")
    print(f"  Histogram bins: {HISTOGRAM_BINS}")
    print(f"================================================") 
