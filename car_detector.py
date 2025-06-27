"""
Car Detection using YOLO with ByteTracker

Example usage:
    python car_detector.py --input videos/carpark2.mp4 --output .\output\videos\2.mp4

To disable color histogram comparison entirely:
    python car_detector.py --input video.mp4 --no-color-hist

Note: Tracking and parking timer are always enabled and required for car detection.
"""

import cv2
import torch
import numpy as np
import os
from pathlib import Path
from tqdm import tqdm
from datetime import datetime, timedelta
from ultralytics import YOLO

# Import configuration
from config import *

# Import a custom ByteTracker implementation
try:
    from track import BYTETracker
except ImportError:
    print("Warning: track.py not found. Tracking will be disabled.")
    BYTETracker = None

class CarDetector:
    def __init__(self, model_path, conf_threshold=None, iou_threshold=None, 
                 parking_threshold=None, track_buffer=None, no_color_hist=False):
        """Initialize car detector with YOLO model and ByteTracker"""
        self.model_path = model_path
        
        # Core parking system configuration (from global variables)
        self.color_threshold = COLOR_CHANGE_THRESHOLD
        self.color_weight = COLOR_POSITION_RATIO
        self.position_weight = 1.0 - COLOR_POSITION_RATIO
        self.color_smoothing = COLOR_CHANGE_SMOOTHING
        self.allowed_parking_time = ALLOWED_PARKING_TIME
        self.position_threshold = POSITION_THRESHOLD
        self.parking_threshold = parking_threshold or PARKING_THRESHOLD
        self.occlusion_tolerance_frames = OCCLUSION_TOLERANCE_FRAMES
        
        # Other configuration (from global variables)
        self.conf_threshold = conf_threshold or CONFIDENCE_THRESHOLD
        self.iou_threshold = iou_threshold or IOU_THRESHOLD
        
        # Tracking stability parameters (from global variables)
        self.track_stability_buffer = TRACK_STABILITY_BUFFER
        self.track_confirmation_frames = TRACK_CONFIRMATION_FRAMES
        self.track_reappear_threshold = TRACK_REAPPEAR_THRESHOLD
        self.track_position_smoothing = TRACK_POSITION_SMOOTHING
        self.track_size_smoothing = TRACK_SIZE_SMOOTHING
        
        # Color histogram parameters (from global variables)
        self.histogram_threshold = HISTOGRAM_THRESHOLD
        self.histogram_bins = HISTOGRAM_BINS
        self.histogram_method = HISTOGRAM_METHOD
        self.use_last_position_for_histogram = USE_LAST_POSITION_FOR_HISTOGRAM
        
        self.model = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.no_color_hist = no_color_hist
        
        # ByteTracker initialization (always enabled)
        self.tracker = None
        self.track_buffer = track_buffer or TRACK_BUFFER
        if BYTETracker is None:
            print("Error: ByteTracker not found in track.py, cannot enable tracking.")
            raise ImportError("track.py is required for car detection")
        else:
            self.tracker = BYTETracker(track_buffer=self.track_buffer)
        
        # Parking timer data
        self.car_timers = {}  # track_id -> car_timer_data
        self.frame_count = 0
        self.illegal_parking_violations = []  # Track illegal parking violations
        self.last_detections = [] # Store last known detections
        
        # Track stability data
        self.stable_tracks = {}  # stable_id -> track_data
        self.track_history = {}  # track_id -> history_data
        self.next_stable_id = 1
        
        self.load_model()
        
    def load_model(self):
        """Load YOLO model from file"""
        print_configuration()
        print(f"Loading model: {self.model_path}")
        print(f"Device: {self.device}")
        print(f"Tracking: {'On (ByteTracker from track.py)' if self.tracker else 'Off'}")
        print(f"Parking timer: On")
        print(f"Color histogram: {'Disabled' if self.no_color_hist else 'Enabled'}")
        if not self.no_color_hist:
            print(f"Color/Position ratio: {COLOR_POSITION_RATIO:.2f} (Color:{self.color_weight:.2f}, Position:{self.position_weight:.2f})")
            print(f"Histogram position: {'Last frame' if self.use_last_position_for_histogram else 'Current frame'}")
            print(f"Fixed parking histogram: Enabled (uses parking start position)")
        print(f"Track stability: {self.track_stability_buffer} frames, {self.track_confirmation_frames} min frames")
        print(f"Track reappear threshold: {self.track_reappear_threshold:.2f}")
        print(f"Occlusion tolerance: {self.occlusion_tolerance_frames} frames (2x for illegal cars)")
        print(f"================================================")
        try:
            self.model = YOLO(self.model_path)
            print("Model loaded successfully!")
        except Exception as e:
            print(f"Error loading model: {e}")
            raise
    
    def calculate_color_histogram(self, image, bbox):
        """Calculate color histogram for a bounding box region"""
        x1, y1, x2, y2 = bbox
        # Ensure coordinates are within image bounds
        h, w = image.shape[:2]
        x1 = max(0, min(x1, w-1))
        y1 = max(0, min(y1, h-1))
        x2 = max(0, min(x2, w-1))
        y2 = max(0, min(y2, h-1))
        
        if x2 <= x1 or y2 <= y1:
            return None
            
        # Extract the region of interest
        roi = image[y1:y2, x1:x2]
        if roi.size == 0:
            return None
            
        # Convert to HSV color space for better color representation
        hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        
        # Calculate histogram for H and S channels (ignore V for lighting invariance)
        hist = cv2.calcHist([hsv_roi], [0, 1], None, [HISTOGRAM_BINS, HISTOGRAM_BINS], [0, 180, 0, 256])
        
        # Normalize histogram
        cv2.normalize(hist, hist, 0, 1, cv2.NORM_MINMAX)
        
        return hist
    
    def compare_histograms(self, hist1, hist2):
        """Compare two histograms and return similarity score"""
        if hist1 is None or hist2 is None:
            return 0.0
        
        # Compare histograms using the specified method
        similarity = cv2.compareHist(hist1, hist2, HISTOGRAM_METHOD)
        
        # For correlation method, values range from -1 to 1, where 1 is perfect match
        # For other methods, higher values indicate more similarity
        if HISTOGRAM_METHOD == cv2.HISTCMP_CORREL:
            return max(0, similarity)  # Convert to 0-1 range
        else:
            return similarity
    
    def is_car_stationary(self, track_id, new_bbox, image=None):
        """Check if car has moved significantly using both position and color histogram with configurable thresholds"""
        if track_id not in self.car_timers or self.car_timers[track_id]['last_position'] is None:
            return False
        
        # Method 1: Geometric position-based detection
        last_pos = self.car_timers[track_id]['last_position']
        new_center = ((new_bbox[0] + new_bbox[2]) // 2, (new_bbox[1] + new_bbox[3]) // 2)
        last_center = ((last_pos[0] + last_pos[2]) // 2, (last_pos[1] + last_pos[3]) // 2)
        
        position_distance = np.sqrt((new_center[0] - last_center[0])**2 + (new_center[1] - last_center[1])**2)
        position_stationary = position_distance <= self.position_threshold
        
        # If color histogram comparison is disabled, only use position-based detection
        if self.no_color_hist:
            return position_stationary
        
        # Method 2: Color histogram-based detection (if image is provided)
        color_stationary = True  # Default to True if no image provided
        color_similarity = 1.0  # Default similarity score
        
        if image is not None:
            color_similarity = self.get_color_similarity_score(track_id, image, new_bbox, self.use_last_position_for_histogram)
            # Use configurable threshold for color change detection
            color_stationary = color_similarity >= (1.0 - self.color_threshold)
        
        # Combined decision with configurable weights
        if self.color_weight > 0 and self.position_weight > 0:
            # Weighted combination of both methods
            combined_score = (self.position_weight * float(position_stationary) + 
                            self.color_weight * float(color_stationary))
            return combined_score >= 0.5  # At least 50% confidence
        elif self.color_weight == 0:
            # Only position-based detection
            return position_stationary
        elif self.position_weight == 0:
            # Only color-based detection
            return color_stationary
        else:
            # Fallback to original logic: Both methods must agree
            return position_stationary and color_stationary
    
    def get_color_similarity_score(self, track_id, image, bbox, use_last_position=False):
        """Get color similarity score for a track"""
        # If color histogram comparison is disabled, return default similarity
        if self.no_color_hist:
            return 1.0
        
        if track_id not in self.car_timers:
            return 1.0  # No car data, assume no change
        
        car_data = self.car_timers[track_id]
        
        # Only use color histogram detection AFTER car has started parking
        # Before parking starts, return default similarity (no color-based movement detection)
        if not car_data.get('is_parked', False):
            return 1.0  # Car not parked yet, no color change detection
        
        # Car is parked, use the fixed parking histogram for comparison
        if car_data.get('parking_histogram') is not None:
            # Use the fixed bbox position when parking started
            parking_bbox = car_data['parking_histogram_bbox']
            if parking_bbox is not None:
                # Calculate histogram at the fixed parking position
                current_hist = self.calculate_color_histogram(image, parking_bbox)
                if current_hist is not None:
                    # Compare with the fixed parking histogram (initial state)
                    parking_hist = car_data['parking_histogram']
                    similarity = self.compare_histograms(current_hist, parking_hist)
                    
                    # Apply smoothing if previous similarity exists
                    if 'last_color_similarity' in car_data:
                        prev_similarity = car_data['last_color_similarity']
                        similarity = (self.color_smoothing * prev_similarity + 
                                     (1 - self.color_smoothing) * similarity)
                    
                    # Store for next frame
                    car_data['last_color_similarity'] = similarity
                    return similarity
        
        # Fallback: if no parking histogram available, return default
        return 1.0
    
    def update_car_timer(self, track_id, bbox, timestamp, image=None):
        """Update car's parking timer with illegal parking detection"""
        if track_id not in self.car_timers:
            # New car detected
            self.car_timers[track_id] = {
                'track_id': track_id,
                'first_seen': timestamp,
                'parking_start': None,
                'current_parking_time': 0,
                'is_parked': False,
                'is_illegal': False,
                'last_position': bbox,
                'stationary_frames': 0,
                'was_illegal_before': False,  # Track if car was illegal before disappearing
                'last_illegal_time': None,    # Store the last illegal parking time
                'frames_not_detected': 0,     # Count frames since last detection
                'last_detected_frame': self.frame_count,  # Track when car was last detected
                'parking_histogram_bbox': None,  # Fixed bbox for histogram comparison when parking starts
                'parking_histogram': None      # Fixed histogram for comparison when parking starts
            }
        else:
            # Update existing car
            car_data = self.car_timers[track_id]
            car_data['last_seen'] = timestamp
            car_data['frames_not_detected'] = 0  # Reset occlusion counter
            car_data['last_detected_frame'] = self.frame_count

            # Check if car is stationary using both methods
            if self.is_car_stationary(track_id, bbox, image):
                car_data['stationary_frames'] += 1
                
                # Start parking timer if stationary for threshold frames
                if car_data['stationary_frames'] >= self.parking_threshold and not car_data['is_parked']:
                    car_data['is_parked'] = True
                    car_data['parking_start'] = timestamp
                    
                    # Record fixed bbox and histogram for parking color comparison
                    if not self.no_color_hist and image is not None:
                        car_data['parking_histogram_bbox'] = bbox.copy()  # Store the bbox when parking starts
                        car_data['parking_histogram'] = self.calculate_color_histogram(image, bbox)
                        print(f"Car #{track_id} started parking at {timestamp} - Fixed histogram bbox: {bbox}")
                    else:
                        print(f"Car #{track_id} started parking at {timestamp}")
                
            else:
                # Car moved, but preserve illegal status if it was illegal before
                if car_data['is_parked']:
                    last_pos = car_data['last_position']
                    new_center = ((bbox[0] + bbox[2]) // 2, (bbox[1] + bbox[3]) // 2)
                    last_center = ((last_pos[0] + last_pos[2]) // 2, (last_pos[1] + last_pos[3]) // 2)
                    position_distance = np.sqrt((new_center[0] - last_center[0])**2 + (new_center[1] - last_center[1])**2)
                    print(f"Car #{track_id} moved ({position_distance:.0f}px) after parking for {car_data['current_parking_time']:.1f}s. Resetting timer.")
                
                # Store illegal status before resetting
                if car_data['is_illegal']:
                    car_data['was_illegal_before'] = True
                    car_data['last_illegal_time'] = car_data['current_parking_time']
                
                self.reset_car_timer(car_data, track_id)
            
            # Update position
            car_data['last_position'] = bbox
    
    def reset_car_timer(self, car_data, track_id):
        """Helper to reset a car's parking state while preserving illegal status"""
        # Preserve illegal status if car was illegal before
        was_illegal = car_data.get('was_illegal_before', False)
        last_illegal_time = car_data.get('last_illegal_time', None)
        
        car_data['is_parked'] = False
        car_data['is_illegal'] = False
        car_data['parking_start'] = None
        car_data['current_parking_time'] = 0
        car_data['stationary_frames'] = 0
        
        # Clear fixed parking histogram data when resetting
        car_data['parking_histogram_bbox'] = None
        car_data['parking_histogram'] = None
        
        # Restore illegal status if car was illegal before and hasn't moved significantly
        if was_illegal and last_illegal_time is not None:
            car_data['was_illegal_before'] = True
            car_data['last_illegal_time'] = last_illegal_time
            print(f"Car #{track_id} was illegal before (parked for {last_illegal_time:.1f}s), preserving status")
    
    def update_all_timers_continuously(self):
        """On every frame, update the timers for all parked cars."""
        timestamp = datetime.now()
        for track_id, car_data in self.car_timers.items():
            if car_data['is_parked']:
                # Recalculate parking time based on current time
                car_data['current_parking_time'] = (timestamp - car_data['parking_start']).total_seconds()
                
                # Check for new illegal parking violations
                if car_data['current_parking_time'] >= self.allowed_parking_time and not car_data['is_illegal']:
                    car_data['is_illegal'] = True
                    violation = {
                        'track_id': track_id,
                        'parking_start': car_data['parking_start'],
                        'violation_time': timestamp,
                        'parking_duration': car_data['current_parking_time']
                    }
                    self.illegal_parking_violations.append(violation)
                    print(f"⚠️  ILLEGAL PARKING: Car #{track_id} exceeded {self.allowed_parking_time}s limit!")
            
            # Check if car was illegal before and has started parking again
            elif car_data.get('was_illegal_before', False) and car_data.get('last_illegal_time', 0) > 0:
                # If car was illegal before and has been stationary for a short time, restore illegal status
                if car_data['stationary_frames'] >= 5:  # Quick restoration threshold
                    car_data['is_parked'] = True
                    car_data['is_illegal'] = True
                    car_data['parking_start'] = timestamp - timedelta(seconds=car_data['last_illegal_time'])
                    car_data['current_parking_time'] = car_data['last_illegal_time']
                    print(f"🔄 RESTORED ILLEGAL STATUS: Car #{track_id} was illegal before, restored after brief interruption")
            
            # Update parking time for occluded cars that were parked
            elif car_data.get('is_parked', False) and car_data.get('frames_not_detected', 0) > 0:
                # Continue counting parking time for occluded cars
                if car_data['parking_start'] is not None:
                    car_data['current_parking_time'] = (timestamp - car_data['parking_start']).total_seconds()
                    
                    # Check for new illegal parking violations for occluded cars
                    if car_data['current_parking_time'] >= self.allowed_parking_time and not car_data['is_illegal']:
                        car_data['is_illegal'] = True
                        violation = {
                            'track_id': track_id,
                            'parking_start': car_data['parking_start'],
                            'violation_time': timestamp,
                            'parking_duration': car_data['current_parking_time']
                        }
                        self.illegal_parking_violations.append(violation)
                        print(f"⚠️  ILLEGAL PARKING (OCCLUDED): Car #{track_id} exceeded {self.allowed_parking_time}s limit!")
    
    def get_parking_time_str(self, track_id):
        """Get formatted parking time string for a car"""
        if track_id not in self.car_timers or not self.car_timers[track_id]['is_parked']:
            return ""
        
        seconds = int(self.car_timers[track_id]['current_parking_time'])
        minutes = seconds // 60
        seconds = seconds % 60
        
        if minutes > 0:
            return f"Parked: {minutes}m {seconds}s"
        else:
            return f"Parked: {seconds}s"
    
    def get_parking_status_str(self, track_id):
        """Get parking status string including illegal parking"""
        if track_id not in self.car_timers or not self.car_timers[track_id]['is_parked']:
            return ""
        
        car_data = self.car_timers[track_id]
        parking_time = self.get_parking_time_str(track_id)
        
        if car_data['is_illegal']:
            return f"ILLEGAL: {parking_time}"
        else:
            return parking_time
    
    def get_parking_report(self):
        """Get parking report sorted by parking time in descending order"""
        parking_data = []
        
        for track_id, car_data in self.car_timers.items():
            if car_data['is_parked']:
                parking_data.append({
                    'track_id': track_id,
                    'parking_time': car_data['current_parking_time'],
                    'parking_start': car_data['parking_start'],
                    'is_illegal': car_data['is_illegal']
                })
        
        # Sort by parking time in descending order
        parking_data.sort(key=lambda x: x['parking_time'], reverse=True)
        return parking_data
    
    def print_parking_results(self):
        """Print parking results sorted by parking time including illegal parking"""
        print("\n" + "="*70)
        print("PARKING RESULTS (Sorted by Parking Time)")
        print("="*70)
        
        parking_data = self.get_parking_report()
        
        if not parking_data:
            print("No cars are currently parked.")
        else:
            print(f"{'Car ID':<8} {'Status':<12} {'Parking Time':<15} {'Start Time':<20}")
            print("-" * 70)
            
            for i, data in enumerate(parking_data, 1):
                track_id = data['track_id']
                parking_time = data['parking_time']
                start_time = data['parking_start']
                is_illegal = data['is_illegal']
                
                # Format parking time
                minutes = int(parking_time) // 60
                seconds = int(parking_time) % 60
                if minutes > 0:
                    time_str = f"{minutes}m {seconds}s"
                else:
                    time_str = f"{seconds}s"
                
                # Format start time
                start_str = start_time.strftime("%H:%M:%S") if start_time else "N/A"
                
                # Status
                status = "ILLEGAL" if is_illegal else "Parked"
                
                print(f"#{track_id:<7} {status:<12} {time_str:<15} {start_str:<20}")
            
            print("-" * 70)
            print(f"Total parked cars: {len(parking_data)}")
            
            # Show summary statistics
            if parking_data:
                total_time = sum(data['parking_time'] for data in parking_data)
                avg_time = total_time / len(parking_data)
                max_time = parking_data[0]['parking_time']
                illegal_count = sum(1 for data in parking_data if data['is_illegal'])
                
                print(f"Total parking time: {int(total_time//60)}m {int(total_time%60)}s")
                print(f"Average parking time: {int(avg_time//60)}m {int(avg_time%60)}s")
                print(f"Longest parking time: {int(max_time//60)}m {int(max_time%60)}s")
                print(f"Illegal parking violations: {illegal_count}")
        
        # Print historical illegal parking violations
        if self.illegal_parking_violations:
            print("\n" + "="*70)
            print("HISTORICAL ILLEGAL PARKING VIOLATIONS")
            print("="*70)
            print(f"{'Car ID':<8} {'Duration':<12} {'Start Time':<20} {'Violation Time':<20}")
            print("-" * 70)
            
            for violation in self.illegal_parking_violations:
                track_id = violation['track_id']
                duration = violation['parking_duration']
                start_time = violation['parking_start']
                violation_time = violation['violation_time']
                
                # Format duration
                minutes = int(duration) // 60
                seconds = int(duration) % 60
                if minutes > 0:
                    duration_str = f"{minutes}m {seconds}s"
                else:
                    duration_str = f"{seconds}s"
                
                # Format times
                start_str = start_time.strftime("%H:%M:%S") if start_time else "N/A"
                violation_str = violation_time.strftime("%H:%M:%S") if violation_time else "N/A"
                
                print(f"#{track_id:<7} {duration_str:<12} {start_str:<20} {violation_str:<20}")
            
            print("-" * 70)
            print(f"Total violations: {len(self.illegal_parking_violations)}")
        
        print("="*70)
    
    def cleanup_old_timers(self, current_track_ids):
        """Remove timers for cars that are no longer visible, with occlusion tolerance"""
        current_ids = set(current_track_ids)
        
        # Update occlusion counters for all cars not currently detected
        for track_id in self.car_timers:
            if track_id not in current_ids:
                self.car_timers[track_id]['frames_not_detected'] += 1
        
        # Only remove cars that have been occluded for too long
        cars_to_remove = []
        for track_id, car_data in self.car_timers.items():
            if track_id not in current_ids:
                frames_occluded = car_data['frames_not_detected']
                
                # If car was illegally parked, give it more tolerance
                if car_data.get('is_illegal', False) or car_data.get('was_illegal_before', False):
                    max_occlusion_frames = self.occlusion_tolerance_frames * 2  # Double tolerance for illegal cars
                else:
                    max_occlusion_frames = self.occlusion_tolerance_frames
                
                if frames_occluded >= max_occlusion_frames:
                    cars_to_remove.append(track_id)
                    if car_data['is_parked']:
                        parking_time = car_data['current_parking_time']
                        if car_data['is_illegal']:
                            print(f"🚗 Car #{track_id} left while ILLEGALLY parked after {parking_time:.1f} seconds (occluded for {frames_occluded} frames)")
                        else:
                            print(f"Car #{track_id} left while parked after {parking_time:.1f} seconds (occluded for {frames_occluded} frames)")
                else:
                    # Car is temporarily occluded, keep its timer data
                    if frames_occluded == 1:  # Just started being occluded
                        if car_data.get('is_illegal', False):
                            print(f"🔄 Car #{track_id} temporarily occluded (illegal parking preserved)")
                        elif car_data.get('is_parked', False):
                            print(f"🔄 Car #{track_id} temporarily occluded (parking timer preserved)")
        
        # Remove cars that have been occluded for too long
        for track_id in cars_to_remove:
            del self.car_timers[track_id]
    
    def detect_cars(self, image):
        """Detect cars in image with optional tracking and parking timer"""
        if self.model is None:
            raise ValueError("Model not loaded!")
        
        try:
            detections = []
            current_track_ids = []
            timestamp = datetime.now()

            # --- Get Detections from YOLO ---
            yolo_results = self.model.predict(image, conf=self.conf_threshold, iou=self.iou_threshold, verbose=False, show=False)

            # --- Process Detections for Tracking ---
            # 1. Prepare YOLO detections for the tracker
            boxes_for_tracker = []
            scores_for_tracker = []
            classes_for_tracker = []
            
            for result in yolo_results:
                boxes = result.boxes
                if boxes is not None:
                    for i in range(len(boxes)):
                        box = boxes[i]
                        cls = int(box.cls[0].cpu().numpy())
                        if cls in [2, 3, 5, 7]: # Filter for cars, motorcycles, buses, and trucks
                            conf = box.conf[0].cpu().numpy()
                            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                            boxes_for_tracker.append([x1, y1, x2, y2])
                            scores_for_tracker.append(conf)
                            classes_for_tracker.append(cls)

            # 2. Update ByteTracker with YOLO detections
            if self.tracker is not None:
                if len(boxes_for_tracker) > 0:
                    tracked_objects = self.tracker.update(
                        np.array(boxes_for_tracker),
                        np.array(scores_for_tracker),
                        np.array(classes_for_tracker)
                    )
                    
                    # 3. Process tracked results for parking timer
                    for i in range(len(tracked_objects)):
                        # Output format from track.py: [x1, y1, x2, y2, track_id, score, cls, idx]
                        # tracked_objects is a numpy array where each row is [x1, y1, x2, y2, track_id, score, cls, idx]
                        obj = tracked_objects[i]
                        if len(obj) >= 8:
                            x1, y1, x2, y2, track_id, score, _, _ = obj
                            bbox = [int(x1), int(y1), int(x2), int(y2)]
                            original_track_id = int(track_id)
                            
                            # Get stable track ID
                            stable_track_id = self.get_stable_track_id(original_track_id, bbox, image)
                            
                            self.update_car_timer(stable_track_id, bbox, timestamp, image)
                            current_track_ids.append(stable_track_id)
                            
                            detections.append({
                                'bbox': bbox,
                                'confidence': float(score),
                                'track_id': stable_track_id
                            })
                else:
                    # No detections, pass empty arrays with correct dimensions
                    tracked_objects = self.tracker.update(
                        np.empty((0, 4), dtype=np.float32),
                        np.empty((0,), dtype=np.float32),
                        np.empty((0,), dtype=np.float32)
                    )
            
            # Cleanup timers for cars that are no longer visible
            self.cleanup_old_timers(current_track_ids)
            
            self.frame_count += 1
            self.last_detections = detections
            return detections
            
        except Exception as e:
            print(f"Detection error: {e}")
            import traceback
            traceback.print_exc()
            return []
    
    def draw_detections(self, image, detections):
        """Draw car bounding boxes, labels, and parking timers on image"""
        result_image = image.copy()
        
        # Display the allowed parking time rule on the top-left corner
        rule_text = f"Allowed Parking Time: {self.allowed_parking_time}s"
        cv2.putText(result_image, rule_text, (20, 40), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2, cv2.LINE_AA)

        for detection in detections:
            bbox = detection['bbox']
            confidence = detection['confidence']
            track_id = detection['track_id']
            
            x1, y1, x2, y2 = bbox
            
            # ===== HSV HISTOGRAM DISPLAY (COMMENT OUT WHEN NOT NEEDED) =====
            # Uncomment the following lines to show HSV histograms for each detection
            # if track_id >= 0:  # Only for tracked objects
            #     self.show_hsv_histogram(result_image, bbox, track_id)
            #     # Uncomment the next line to also show comparison with previous histogram
            #     # self.show_comparison_histograms(result_image, bbox, track_id)
            # ===== END HSV HISTOGRAM DISPLAY =====
            
            # Choose color based on parking status
            if track_id >= 0 and track_id in self.car_timers:
                car_data = self.car_timers[track_id]
                if car_data['is_parked']:
                    if car_data['is_illegal']:
                        color = (0, 0, 255)  # Red for illegal parking
                    else:
                        color = (0, 255, 255)  # Yellow for legal parking
                else:
                    color = (0, 255, 0)  # Green for moving cars
            else:
                color = (0, 255, 0)  # Default green
            
            # Draw bounding box
            cv2.rectangle(result_image, (x1, y1), (x2, y2), color, 2)
            
            # Create integrated label with timer
            if track_id >= 0:
                label = f"Car #{track_id}: {confidence:.2f}"
            else:
                label = f"Car: {confidence:.2f}"
            
            # Add timer information to the same label
            if track_id >= 0 and track_id in self.car_timers:
                car_data = self.car_timers[track_id]
                
                if car_data['is_parked']:
                    # Show parking status and timer
                    parking_status = self.get_parking_status_str(track_id)
                    label += f" | {parking_status}"
                else:
                    # Show stationary counter and detection method info
                    stationary_frames = car_data['stationary_frames']
                    if stationary_frames > 0:
                        label += f" | Stationary: {stationary_frames}/{self.parking_threshold}"
                        
                        # Add detection method info for debugging
                        if hasattr(self, 'get_detection_method_info'):
                            try:
                                info = self.get_detection_method_info(track_id, result_image, bbox)
                                if isinstance(info, dict):
                                    pos_dist = info.get('position_distance', 0)
                                    color_sim = info.get('color_similarity', 0)
                                    using_fixed = info.get('using_fixed_histogram', False)
                                    label += f" | Pos:{pos_dist:.0f}px Color:{color_sim:.2f}"
                                    if using_fixed:
                                        label += " [Fixed]"
                            except:
                                pass
                    else:
                        label += " | Moving"
            
            # Draw integrated label background and text
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
            cv2.rectangle(result_image, (x1, y1 - label_size[1] - 10), 
                         (x1 + label_size[0], y1), color, -1)
            cv2.putText(result_image, label, (x1, y1 - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
        
        return result_image
    
    def process_image(self, image_path, output_path=None):
        """Process single image for car detection"""
        print(f"Processing image: {image_path}")
        
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Cannot read image: {image_path}")
        
        detections = self.detect_cars(image)
        print(f"Found {len(detections)} cars")
        
        result_image = self.draw_detections(image, detections)
        
        if output_path:
            cv2.imwrite(output_path, result_image)
            print(f"Saved to: {output_path}")
        else:
            cv2.imshow('Car Detection', result_image)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        
        return detections
    
    def process_video(self, video_path, output_path=None, show_progress=True):
        """Process video for car detection with progress tracking"""
        print(f"Processing video: {video_path}")
        
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")
        
        # Get video properties
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        print(f"Video: {total_frames} frames, {fps} FPS, {width}x{height}")
        
        # Setup video writer
        writer = None
        if output_path:
            fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
            writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        # Process frames
        frame_count = 0
        total_cars = 0
        unique_tracks = set()
        
        if show_progress:
            pbar = tqdm(total=total_frames, desc="Processing")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # --- Detection on every frame ---
            self.last_detections = self.detect_cars(frame)

            # --- Continuous Timer Update (Every Frame) ---
            self.update_all_timers_continuously()

            # --- Drawing ---
            detections_to_draw = self.last_detections
            total_cars = len(detections_to_draw)
            
            # Count unique tracks from the timer data
            unique_tracks = set(self.car_timers.keys())
            
            result_frame = self.draw_detections(frame, detections_to_draw)
            
            if writer:
                writer.write(result_frame)
            
            frame_count += 1
            
            if show_progress:
                pbar.update(1)
                pbar.set_postfix({'Cars': total_cars, 'Tracks': len(unique_tracks)})
        
        # Cleanup
        cap.release()
        if writer:
            writer.release()
        if show_progress:
            pbar.close()
        
        print(f"Complete: {frame_count} frames, {total_cars} cars, {len(unique_tracks)} tracks")
        if output_path:
            print(f"Saved to: {output_path}")
        
        # Print parking results after processing
        self.print_parking_results()
        
        return total_cars, len(unique_tracks)

    def get_detection_method_info(self, track_id, image, bbox):
        """Get detailed information about detection methods for debugging"""
        if track_id not in self.car_timers:
            return None
        
        # Position-based detection info
        last_pos = self.car_timers[track_id]['last_position']
        new_center = ((bbox[0] + bbox[2]) // 2, (bbox[1] + bbox[3]) // 2)
        last_center = ((last_pos[0] + last_pos[2]) // 2, (last_pos[1] + last_pos[3]) // 2)
        position_distance = np.sqrt((new_center[0] - last_center[0])**2 + (new_center[1] - last_center[1])**2)
        position_stationary = position_distance <= self.position_threshold
        
        # If color histogram is disabled, return position-only info
        if self.no_color_hist:
            info = {
                'track_id': track_id,
                'position_distance': position_distance,
                'position_threshold': self.position_threshold,
                'position_stationary': position_stationary,
                'color_similarity': 1.0,      # Default when disabled
                'color_threshold': 1.0,       # Default when disabled
                'color_stationary': True,     # Default when disabled
                'color_weight': 0.0,          # Disabled
                'position_weight': 1.0,       # Only position detection
                'color_position_ratio': 0.0,  # Position only
                'combined_score': float(position_stationary),
                'combined_stationary': position_stationary,
                'using_fixed_histogram': False,
                'parking_histogram_bbox': None
            }
            return info
        
        # Color histogram-based detection info (only for parked cars)
        color_similarity = 1.0
        color_stationary = True
        car_data = self.car_timers[track_id]
        
        # Only use color detection if car is parked and has fixed histogram
        if car_data.get('is_parked', False) and car_data.get('parking_histogram') is not None:
            color_similarity = self.get_color_similarity_score(track_id, image, bbox, False)
            color_stationary = color_similarity >= (1.0 - self.color_threshold)
        
        # Combined result
        if self.color_weight > 0 and self.position_weight > 0:
            combined_score = (self.position_weight * float(position_stationary) + 
                            self.color_weight * float(color_stationary))
            combined_stationary = combined_score >= 0.5
        else:
            combined_stationary = position_stationary and color_stationary
        
        info = {
            'track_id': track_id,
            'position_distance': position_distance,
            'position_threshold': self.position_threshold,
            'position_stationary': position_stationary,
            'color_similarity': color_similarity,
            'color_threshold': 1.0 - self.color_threshold,
            'color_stationary': color_stationary,
            'color_weight': self.color_weight,
            'position_weight': self.position_weight,
            'color_position_ratio': COLOR_POSITION_RATIO,
            'combined_score': combined_score if self.color_weight > 0 and self.position_weight > 0 else None,
            'combined_stationary': combined_stationary,
            'using_fixed_histogram': car_data.get('is_parked', False) and car_data.get('parking_histogram') is not None,
            'parking_histogram_bbox': car_data.get('parking_histogram_bbox', None)
        }
        
        return info
    
    def show_hsv_histogram(self, image, bbox, track_id=None, window_name=None):
        """
        Display HSV histogram for a bounding box region
        This function can be commented out when not needed
        
        Args:
            image: Input image
            bbox: Bounding box [x1, y1, x2, y2]
            track_id: Track ID for labeling (optional)
            window_name: Custom window name (optional)
        """
        try:
            # Extract ROI
            x1, y1, x2, y2 = bbox
            h, w = image.shape[:2]
            x1 = max(0, min(x1, w-1))
            y1 = max(0, min(y1, h-1))
            x2 = max(0, min(x2, w-1))
            y2 = max(0, min(y2, h-1))
            
            if x2 <= x1 or y2 <= y1:
                print(f"Invalid bbox for histogram: {bbox}")
                return
            
            roi = image[y1:y2, x1:x2]
            if roi.size == 0:
                print(f"Empty ROI for bbox: {bbox}")
                return
            
            # Convert to HSV
            hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
            
            # Calculate histograms for each channel
            h_hist = cv2.calcHist([hsv_roi], [0], None, [180], [0, 180])
            s_hist = cv2.calcHist([hsv_roi], [1], None, [256], [0, 256])
            v_hist = cv2.calcHist([hsv_roi], [2], None, [256], [0, 256])
            
            # Normalize histograms
            cv2.normalize(h_hist, h_hist, 0, 255, cv2.NORM_MINMAX)
            cv2.normalize(s_hist, s_hist, 0, 255, cv2.NORM_MINMAX)
            cv2.normalize(v_hist, v_hist, 0, 255, cv2.NORM_MINMAX)
            
            # Create histogram visualization
            hist_width = 400
            hist_height = 300
            hist_img = np.zeros((hist_height, hist_width, 3), dtype=np.uint8)
            
            # Draw H channel (Hue) - Red
            h_bins = np.linspace(0, hist_width, 180)
            for i in range(179):
                x1_hist = int(h_bins[i])
                x2_hist = int(h_bins[i+1])
                y1_hist = hist_height - int(h_hist[i] * hist_height / 255)
                y2_hist = hist_height
                cv2.rectangle(hist_img, (x1_hist, y1_hist), (x2_hist, y2_hist), (0, 0, 255), -1)
            
            # Draw S channel (Saturation) - Green
            s_bins = np.linspace(0, hist_width, 256)
            for i in range(255):
                x1_hist = int(s_bins[i])
                x2_hist = int(s_bins[i+1])
                y1_hist = hist_height - int(s_hist[i] * hist_height / 255)
                y2_hist = hist_height
                cv2.rectangle(hist_img, (x1_hist, y1_hist), (x2_hist, y2_hist), (0, 255, 0), -1)
            
            # Draw V channel (Value) - Blue
            v_bins = np.linspace(0, hist_width, 256)
            for i in range(255):
                x1_hist = int(v_bins[i])
                x2_hist = int(v_bins[i+1])
                y1_hist = hist_height - int(v_hist[i] * hist_height / 255)
                y2_hist = hist_height
                cv2.rectangle(hist_img, (x1_hist, y1_hist), (x2_hist, y2_hist), (255, 0, 0), -1)
            
            # Add labels
            cv2.putText(hist_img, "H (Red)", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
            cv2.putText(hist_img, "S (Green)", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            cv2.putText(hist_img, "V (Blue)", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
            
            # Add track ID if provided
            if track_id is not None:
                cv2.putText(hist_img, f"Track #{track_id}", (10, hist_height - 10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            # Show ROI and histogram
            window_title = window_name or f"HSV Histogram - Track {track_id}" if track_id else "HSV Histogram"
            
            # Resize ROI to match histogram height for proper concatenation
            roi_height, roi_width = roi.shape[:2]
            if roi_height > 0 and roi_width > 0:
                # Calculate new width to maintain aspect ratio
                new_width = int(roi_width * hist_height / roi_height)
                roi_resized = cv2.resize(roi, (new_width, hist_height))
                
                # Create combined display
                combined_img = np.zeros((hist_height, hist_width + new_width, 3), dtype=np.uint8)
                combined_img[:, :hist_width] = hist_img
                combined_img[:, hist_width:] = roi_resized
                
                # Add separator line
                cv2.line(combined_img, (hist_width, 0), (hist_width, hist_height), (255, 255, 255), 2)
                
                # Show the combined image
                cv2.imshow(window_title, combined_img)
                cv2.waitKey(1)  # Non-blocking display
            else:
                # If ROI is too small, just show histogram
                cv2.imshow(window_title, hist_img)
                cv2.waitKey(1)
            
            # Print histogram statistics
            print(f"Track #{track_id} HSV Stats - H: {np.mean(hsv_roi[:,:,0]):.1f}, "
                  f"S: {np.mean(hsv_roi[:,:,1]):.1f}, V: {np.mean(hsv_roi[:,:,2]):.1f}")
            
        except Exception as e:
            print(f"Error showing HSV histogram: {e}")
    
    def calculate_track_similarity(self, track1_data, track2_data):
        """Calculate similarity between two tracks based on position and size"""
        if track1_data is None or track2_data is None:
            return 0.0
        
        # Position similarity (Euclidean distance)
        pos1 = track1_data.get('center', (0, 0))
        pos2 = track2_data.get('center', (0, 0))
        position_distance = np.sqrt((pos1[0] - pos2[0])**2 + (pos1[1] - pos2[1])**2)
        
        # Normalize position distance (assume max distance of 1000 pixels)
        max_position_distance = 1000
        position_similarity = max(0, 1 - position_distance / max_position_distance)
        
        # Size similarity (area ratio)
        area1 = track1_data.get('area', 1)
        area2 = track2_data.get('area', 1)
        if area1 > 0 and area2 > 0:
            area_ratio = min(area1, area2) / max(area1, area2)
        else:
            area_ratio = 0
        
        # Combined similarity (weighted average)
        # Give more weight to position similarity
        position_weight = 0.7
        size_weight = 0.3
        
        combined_similarity = (position_weight * position_similarity + 
                             size_weight * area_ratio)
        
        return combined_similarity
    
    def update_track_history(self, track_id, bbox, image=None):
        """Update track history with position, size, and features"""
        if track_id is None or bbox is None or len(bbox) != 4:
            return
        
        if track_id not in self.track_history:
            self.track_history[track_id] = {
                'first_seen': self.frame_count,
                'last_seen': self.frame_count,
                'frames_seen': 1,
                'positions': [],
                'sizes': [],
                'stable_id': None
            }
        
        history = self.track_history[track_id]
        history['last_seen'] = self.frame_count
        history['frames_seen'] += 1
        
        # Calculate center and area
        center = ((bbox[0] + bbox[2]) // 2, (bbox[1] + bbox[3]) // 2)
        area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
        
        # Validate area
        if area <= 0:
            return
        
        # Apply smoothing to position and size
        if len(history['positions']) > 0:
            last_center = history['positions'][-1]
            smoothed_center = (
                int(self.track_position_smoothing * last_center[0] + (1 - self.track_position_smoothing) * center[0]),
                int(self.track_position_smoothing * last_center[1] + (1 - self.track_position_smoothing) * center[1])
            )
            history['positions'].append(smoothed_center)
        else:
            history['positions'].append(center)
        
        if len(history['sizes']) > 0:
            last_area = history['sizes'][-1]
            smoothed_area = self.track_size_smoothing * last_area + (1 - self.track_size_smoothing) * area
            history['sizes'].append(smoothed_area)
        else:
            history['sizes'].append(area)
        
        # Keep only recent history
        max_history = 30
        if len(history['positions']) > max_history:
            history['positions'] = history['positions'][-max_history:]
            history['sizes'] = history['sizes'][-max_history:]
    
    def assign_stable_id(self, track_id):
        """Assign a stable ID to a track if it meets stability criteria"""
        if track_id not in self.track_history:
            return None
        
        history = self.track_history[track_id]
        
        # Check if track is confirmed (seen for minimum frames)
        if history['frames_seen'] < self.track_confirmation_frames:
            return None
        
        # Check if track already has a stable ID
        if history['stable_id'] is not None:
            return history['stable_id']
        
        # Check if track has been stable for enough frames
        if history['frames_seen'] >= self.track_stability_buffer:
            # Assign new stable ID
            stable_id = self.next_stable_id
            self.next_stable_id += 1
            history['stable_id'] = stable_id
            
            # Store track data
            self.stable_tracks[stable_id] = {
                'track_id': track_id,
                'center': history['positions'][-1],
                'area': history['sizes'][-1],
                'first_seen': history['first_seen'],
                'last_seen': history['last_seen']
            }
            
            return stable_id
        
        return None
    
    def find_reappearing_track(self, bbox, image=None):
        """Find if a track is reappearing based on similarity to stable tracks"""
        # Validate inputs
        if bbox is None or len(bbox) != 4 or len(self.stable_tracks) == 0:
            return None
        
        # Calculate current track data
        center = ((bbox[0] + bbox[2]) // 2, (bbox[1] + bbox[3]) // 2)
        area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
        
        # Validate area
        if area <= 0:
            return None
        
        current_data = {
            'center': center,
            'area': area
        }
        
        # Find best matching stable track
        best_match = None
        best_similarity = 0
        
        for stable_id, stable_data in self.stable_tracks.items():
            similarity = self.calculate_track_similarity(current_data, stable_data)
            if similarity > best_similarity and similarity >= self.track_reappear_threshold:
                best_similarity = similarity
                best_match = stable_id
        
        return best_match
    
    def get_stable_track_id(self, track_id, bbox, image=None):
        """Get stable track ID for a given track"""
        # Validate inputs
        if track_id is None or bbox is None or len(bbox) != 4:
            return track_id
        
        # Update track history
        self.update_track_history(track_id, bbox, image)
        
        # Try to assign stable ID
        stable_id = self.assign_stable_id(track_id)
        if stable_id is not None:
            return stable_id
        
        # Check if this might be a reappearing track
        reappearing_id = self.find_reappearing_track(bbox, image)
        if reappearing_id is not None:
            # Update the stable track data
            self.stable_tracks[reappearing_id]['last_seen'] = self.frame_count
            return reappearing_id
        
        # Return original track ID if no stable ID available
        return track_id

def create_output_dirs():
    """Create output directories for results"""
    output_dir = Path("output")
    output_dir.mkdir(exist_ok=True)
    (output_dir / "images").mkdir(exist_ok=True)
    (output_dir / "videos").mkdir(exist_ok=True)
    return output_dir

def main():
    """Main function to run car detection"""
    from config import parse_arguments
    args = parse_arguments()
    
    # Create output directories
    output_dir = create_output_dirs()
    
    # Initialize detector
    detector = CarDetector(args.model, args.conf, args.iou, 
                         parking_threshold=args.parking_threshold, track_buffer=args.track_buffer, no_color_hist=args.no_color_hist)
    
    # Check input file
    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Error: Input file {args.input} not found!")
        return
    
    # Generate output path
    if args.output is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        if input_path.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp']:
            output_path = output_dir / "images" / f"cars_{timestamp}_{input_path.name}"
        else:
            output_path = output_dir / "videos" / f"cars_{timestamp}_{input_path.name}"
    else:
        output_path = args.output
    
    # Process input
    try:
        if input_path.suffix.lower() in ['.jpg', '.jpeg', '.png', 'bmp']:
            detector.process_image(str(input_path), str(output_path))
        else:
            detector.process_video(str(input_path), str(output_path))
            
    except Exception as e:
        print(f"Error processing {args.input}: {e}")

if __name__ == "__main__":
    main() 
