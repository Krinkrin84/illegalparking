"""
Car Detection using YOLO with ByteTracker

Example usage:
    python car_detector.py --input videos/carpark2.mp4 --output .\output\videos\2.mp4 --model yolov9c.pt --show-timer

"""

#Configuration
ALLOWED_PARKING_TIME = 20  # seconds - time limit before illegal parking
POSITION_THRESHOLD = 20   # Max movement in pixels to consider a car stationary. If it moves more, the timer resets.
# Purpose: How many frames a car must stay stationary before starting the parking timer
PARKING_THRESHOLD = 30  # frames to wait before counting as parked 

# Yolo configuration
CONFIDENCE_THRESHOLD = 0.35  # minimum confidence for car detection
IOU_THRESHOLD = 0.7  # IoU threshold for NMS
DEFAULT_MODEL = "yolov9c.pt"  # default YOLO model file

# ByteTracker configuration
TRACK_THRESHOLD = 0.5  # tracking confidence threshold
TRACK_BUFFER = 120  # frames to keep lost tracks
MATCH_THRESHOLD = 0.8  # matching threshold for tracking
MIN_BOX_AREA = 10  # minimum box area to track

# Tracker configuration
INFERENCE_INTERVAL = 0  # Seconds between YOLO inferences

import cv2
import torch
import numpy as np
import os
import argparse
from pathlib import Path
from tqdm import tqdm
from datetime import datetime, timedelta
from ultralytics import YOLO

# Import a custom ByteTracker implementation
try:
    from track import BYTETracker
except ImportError:
    print("Warning: track.py not found. Tracking will be disabled.")
    BYTETracker = None


# Create a simple args class for ByteTracker
class TrackerArgs:
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
    parser.add_argument("--no-track", action="store_true", help="Disable tracking")
    parser.add_argument("--parking-threshold", type=int, default=PARKING_THRESHOLD, help="Frames to wait before counting as parked")
    parser.add_argument("--show-timer", action="store_true", help="Display parking time on video")
    parser.add_argument("--track-buffer", type=int, default=TRACK_BUFFER, help="Number of frames to keep a lost track.")
    parser.add_argument("--inference-interval", type=float, default=INFERENCE_INTERVAL, help="Seconds between YOLO inferences.")
    return parser.parse_args()

class CarDetector:
    def __init__(self, model_path, conf_threshold=None, iou_threshold=None, track_mode=True, 
                 parking_threshold=None, show_timer=False, track_buffer=None, inference_interval=None):
        """Initialize car detector with YOLO model and ByteTracker"""
        self.model_path = model_path
        self.conf_threshold = conf_threshold or CONFIDENCE_THRESHOLD
        self.iou_threshold = iou_threshold or IOU_THRESHOLD
        self.track_mode = track_mode
        self.parking_threshold = parking_threshold or PARKING_THRESHOLD
        self.position_threshold = POSITION_THRESHOLD  # Use global variable
        self.show_timer = show_timer
        self.allowed_parking_time = ALLOWED_PARKING_TIME  # Use global variable
        self.model = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.inference_interval = inference_interval or INFERENCE_INTERVAL
        
        # ByteTracker initialization
        self.tracker = None
        self.track_buffer = track_buffer or TRACK_BUFFER
        if self.track_mode:
            if BYTETracker is None:
                print("Error: ByteTracker not found in track.py, cannot enable tracking.")
                self.track_mode = False
            else:
                self.tracker = BYTETracker(track_buffer=self.track_buffer)
        
        # Parking timer data
        self.car_timers = {}  # track_id -> car_timer_data
        self.frame_count = 0
        self.illegal_parking_violations = []  # Track illegal parking violations
        self.last_detections = [] # Store last known detections
        
        self.load_model()
        
    def load_model(self):
        """Load YOLO model from file"""
        print(f"================Configuration===================")
        print(f"Loading model: {self.model_path}")
        print(f"Device: {self.device}")
        print(f"Tracking: {'On (ByteTracker from track.py)' if self.track_mode else 'Off'}")
        print(f"Parking timer: {'On' if self.show_timer else 'Off'}")
        print(f"================================================")
        try:
            self.model = YOLO(self.model_path)
            print("Model loaded successfully!")
        except Exception as e:
            print(f"Error loading model: {e}")
            raise
    
    def is_car_stationary(self, track_id, new_bbox):
        """Check if car has moved significantly from last position"""
        if track_id not in self.car_timers or self.car_timers[track_id]['last_position'] is None:
            return False
        
        last_pos = self.car_timers[track_id]['last_position']
        new_center = ((new_bbox[0] + new_bbox[2]) // 2, (new_bbox[1] + new_bbox[3]) // 2)
        last_center = ((last_pos[0] + last_pos[2]) // 2, (last_pos[1] + last_pos[3]) // 2)
        
        distance = np.sqrt((new_center[0] - last_center[0])**2 + (new_center[1] - last_center[1])**2)
        return distance <= self.position_threshold
    
    def update_car_timer(self, track_id, bbox, timestamp):
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
                'stationary_frames': 0
            }
        else:
            # Update existing car
            car_data = self.car_timers[track_id]
            car_data['last_seen'] = timestamp

            # Check if car is stationary
            if self.is_car_stationary(track_id, bbox):
                car_data['stationary_frames'] += 1
                
                # Start parking timer if stationary for threshold frames
                if car_data['stationary_frames'] >= self.parking_threshold and not car_data['is_parked']:
                    car_data['is_parked'] = True
                    car_data['parking_start'] = timestamp
                    print(f"Car #{track_id} started parking at {timestamp}")
                
            else:
                # Car moved, reset parking state
                if car_data['is_parked']:
                    last_pos = car_data['last_position']
                    new_center = ((bbox[0] + bbox[2]) // 2, (bbox[1] + bbox[3]) // 2)
                    last_center = ((last_pos[0] + last_pos[2]) // 2, (last_pos[1] + last_pos[3]) // 2)
                    distance = np.sqrt((new_center[0] - last_center[0])**2 + (new_center[1] - last_center[1])**2)
                    print(f"Car #{track_id} moved ({distance:.0f}px) after parking for {car_data['current_parking_time']:.1f}s. Resetting timer.")
                
                self.reset_car_timer(car_data, track_id)
            
            car_data['last_position'] = bbox
    
    def reset_car_timer(self, car_data, track_id):
        """Helper to reset a car's parking state."""
        car_data['is_parked'] = False
        car_data['is_illegal'] = False
        car_data['parking_start'] = None
        car_data['current_parking_time'] = 0
        car_data['stationary_frames'] = 0
    
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
        """Remove timers for cars that are no longer visible"""
        current_ids = set(current_track_ids)
        old_ids = set(self.car_timers.keys()) - current_ids
        
        for track_id in old_ids:
            if self.car_timers[track_id]['is_parked']:
                parking_time = self.car_timers[track_id]['current_parking_time']
                print(f"Car #{track_id} left while parked after {parking_time:.1f} seconds")
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

            # --- Process Detections for Tracking or Standalone Use ---
            if not self.track_mode:
                # --- Detection-only mode ---
                for result in yolo_results:
                    boxes = result.boxes
                    if boxes is not None:
                        for i in range(len(boxes)):
                            box = boxes[i]
                            if int(box.cls[0].cpu().numpy()) == 2: # car class
                                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                                conf = box.conf[0].cpu().numpy()
                                detections.append({
                                    'bbox': [int(x1), int(y1), int(x2), int(y2)],
                                    'confidence': float(conf),
                                    'track_id': -1
                                })
            else:
                # --- Tracking mode with ByteTracker from track.py ---
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
                    tracked_objects = self.tracker.update(
                        np.array(boxes_for_tracker),
                        np.array(scores_for_tracker),
                        np.array(classes_for_tracker)
                    )
                    
                    # 3. Process tracked results for parking timer
                    for obj in tracked_objects:
                        # Output format from track.py: [x1, y1, x2, y2, track_id, score, cls, idx]
                        x1, y1, x2, y2, track_id, score, _, _ = obj
                        bbox = [int(x1), int(y1), int(x2), int(y2)]
                        track_id = int(track_id)
                        
                        self.update_car_timer(track_id, bbox, timestamp)
                        current_track_ids.append(track_id)
                        
                        detections.append({
                            'bbox': bbox,
                            'confidence': float(score),
                            'track_id': track_id
                        })
            
            # Cleanup timers for cars that are no longer visible
            if self.track_mode:
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
            
            # Choose color based on parking status
            if self.show_timer and track_id >= 0 and track_id in self.car_timers:
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
            if self.show_timer and track_id >= 0 and track_id in self.car_timers:
                car_data = self.car_timers[track_id]
                
                if car_data['is_parked']:
                    # Show parking status and timer
                    parking_status = self.get_parking_status_str(track_id)
                    label += f" | {parking_status}"
                else:
                    # Show stationary counter
                    stationary_frames = car_data['stationary_frames']
                    if stationary_frames > 0:
                        label += f" | Stationary: {stationary_frames}/{self.parking_threshold}"
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
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        last_inference_time = -self.inference_interval # Ensure the first frame is processed
        
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
            
            # --- Periodic Inference ---
            current_time_sec = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
            if current_time_sec - last_inference_time >= self.inference_interval:
                last_inference_time = current_time_sec
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
        if self.show_timer:
            self.print_parking_results()
        
        return total_cars, len(unique_tracks)

def create_output_dirs():
    """Create output directories for results"""
    output_dir = Path("output")
    output_dir.mkdir(exist_ok=True)
    (output_dir / "images").mkdir(exist_ok=True)
    (output_dir / "videos").mkdir(exist_ok=True)
    return output_dir

def main():
    """Main function to run car detection"""
    args = parse_arguments()
    
    # Create output directories
    output_dir = create_output_dirs()
    
    # Initialize detector
    detector = CarDetector(args.model, args.conf, args.iou, track_mode=not args.no_track, 
                         parking_threshold=args.parking_threshold, show_timer=args.show_timer,
                         track_buffer=args.track_buffer, inference_interval=args.inference_interval)
    
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