import cv2
import torch
import numpy as np
import os
import argparse
from pathlib import Path
from tqdm import tqdm
import time
from datetime import datetime

class ObjectDetector:
    def __init__(self, model_path, target_classes=None, conf_threshold=0.5, iou_threshold=0.45):
        """
        Initialize the object detector with YOLO model
        
        Args:
            model_path (str): Path to the YOLO .pt model file
            target_classes (list): List of class IDs to detect (None = all classes)
            conf_threshold (float): Confidence threshold for detections
            iou_threshold (float): IoU threshold for NMS
        """
        self.model_path = model_path
        self.target_classes = target_classes
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.model = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # COCO dataset class names (you can modify this for your custom dataset)
        self.class_names = {
            0: 'person', 1: 'bicycle', 2: 'car', 3: 'motorcycle', 4: 'airplane', 5: 'bus',
            6: 'train', 7: 'truck', 8: 'boat', 9: 'traffic light', 10: 'fire hydrant',
            11: 'stop sign', 12: 'parking meter', 13: 'bench', 14: 'bird', 15: 'cat',
            16: 'dog', 17: 'horse', 18: 'sheep', 19: 'cow', 20: 'elephant',
            21: 'bear', 22: 'zebra', 23: 'giraffe', 24: 'backpack', 25: 'umbrella',
            26: 'handbag', 27: 'tie', 28: 'suitcase', 29: 'frisbee', 30: 'skis',
            31: 'snowboard', 32: 'sports ball', 33: 'kite', 34: 'baseball bat', 35: 'baseball glove',
            36: 'skateboard', 37: 'surfboard', 38: 'tennis racket', 39: 'bottle', 40: 'wine glass',
            41: 'cup', 42: 'fork', 43: 'knife', 44: 'spoon', 45: 'bowl',
            46: 'banana', 47: 'apple', 48: 'sandwich', 49: 'orange', 50: 'broccoli',
            51: 'carrot', 52: 'hot dog', 53: 'pizza', 54: 'donut', 55: 'cake',
            56: 'chair', 57: 'couch', 58: 'potted plant', 59: 'bed', 60: 'dining table',
            61: 'toilet', 62: 'tv', 63: 'laptop', 64: 'mouse', 65: 'remote',
            66: 'keyboard', 67: 'cell phone', 68: 'microwave', 69: 'oven', 70: 'toaster',
            71: 'sink', 72: 'refrigerator', 73: 'book', 74: 'clock', 75: 'vase',
            76: 'scissors', 77: 'teddy bear', 78: 'hair drier', 79: 'toothbrush'
        }
        
        self.load_model()
        
    def load_model(self):
        """Load the YOLO model from .pt file"""
        print(f"Loading YOLO model from: {self.model_path}")
        print(f"Using device: {self.device}")
        
        try:
            from ultralytics import YOLO
            self.model = YOLO(self.model_path)
            print("Model loaded successfully!")
            
        except Exception as e:
            print(f"Error loading model: {e}")
            raise
    
    def get_class_name(self, class_id):
        """Get class name from class ID"""
        return self.class_names.get(class_id, f"class_{class_id}")
    
    def detect_objects(self, image):
        """
        Detect objects in the given image
        
        Args:
            image: Input image (numpy array)
            
        Returns:
            list: List of detections with bounding boxes, confidence scores, and class labels
        """
        if self.model is None:
            raise ValueError("Model not loaded!")
        
        try:
            results = self.model.predict(
                image, 
                conf=self.conf_threshold, 
                iou=self.iou_threshold,
                verbose=False
            )
            
            detections = []
            for result in results:
                boxes = result.boxes
                if boxes is not None:
                    for box in boxes:
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        conf = box.conf[0].cpu().numpy()
                        cls = int(box.cls[0].cpu().numpy())
                        
                        # Filter by target classes if specified
                        if self.target_classes is None or cls in self.target_classes:
                            detections.append({
                                'bbox': [int(x1), int(y1), int(x2), int(y2)],
                                'confidence': float(conf),
                                'class': cls,
                                'class_name': self.get_class_name(cls)
                            })
            
            return detections
            
        except Exception as e:
            print(f"Error during detection: {e}")
            return []
    
    def draw_detections(self, image, detections):
        """
        Draw bounding boxes and labels on the image
        
        Args:
            image: Input image
            detections: List of detections
            
        Returns:
            numpy.ndarray: Image with bounding boxes drawn
        """
        result_image = image.copy()
        
        # Define colors for different classes (you can customize this)
        colors = {
            'person': (255, 0, 0),      # Red
            'car': (0, 255, 0),         # Green
            'truck': (0, 255, 255),     # Yellow
            'bus': (255, 0, 255),       # Magenta
            'motorcycle': (255, 255, 0), # Cyan
            'bicycle': (128, 0, 128),   # Purple
            'dog': (0, 128, 255),       # Orange
            'cat': (255, 128, 0),       # Blue
        }
        
        for detection in detections:
            bbox = detection['bbox']
            confidence = detection['confidence']
            class_name = detection['class_name']
            
            x1, y1, x2, y2 = bbox
            
            # Get color for this class
            color = colors.get(class_name, (0, 255, 0))  # Default to green
            
            # Draw bounding box
            cv2.rectangle(result_image, (x1, y1), (x2, y2), color, 2)
            
            # Draw label
            label = f"{class_name}: {confidence:.2f}"
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
            
            # Draw label background
            cv2.rectangle(result_image, (x1, y1 - label_size[1] - 10), 
                         (x1 + label_size[0], y1), color, -1)
            
            # Draw label text
            cv2.putText(result_image, label, (x1, y1 - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
        
        return result_image
    
    def process_image(self, image_path, output_path=None):
        """
        Process a single image for object detection
        
        Args:
            image_path (str): Path to input image
            output_path (str): Path to save output image (optional)
        """
        print(f"Processing image: {image_path}")
        
        # Read image
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not read image: {image_path}")
        
        # Detect objects
        detections = self.detect_objects(image)
        print(f"Found {len(detections)} objects")
        
        # Print detection details
        for det in detections:
            print(f"  - {det['class_name']}: {det['confidence']:.2f}")
        
        # Draw detections
        result_image = self.draw_detections(image, detections)
        
        # Save or display result
        if output_path:
            cv2.imwrite(output_path, result_image)
            print(f"Result saved to: {output_path}")
        else:
            # Display image
            cv2.imshow('Object Detection', result_image)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        
        return detections
    
    def process_video(self, video_path, output_path=None, show_progress=True):
        """
        Process a video for object detection with progress tracking
        
        Args:
            video_path (str): Path to input video
            output_path (str): Path to save output video (optional)
            show_progress (bool): Whether to show progress bar
        """
        print(f"Processing video: {video_path}")
        
        # Open video
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")
        
        # Get video properties
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        print(f"Video properties: {total_frames} frames, {fps} FPS, {width}x{height}")
        
        # Setup video writer if output path is provided
        writer = None
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        # Process frames
        frame_count = 0
        total_objects = 0
        
        if show_progress:
            pbar = tqdm(total=total_frames, desc="Processing video")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Detect objects in frame
            detections = self.detect_objects(frame)
            total_objects += len(detections)
            
            # Draw detections
            result_frame = self.draw_detections(frame, detections)
            
            # Write frame if output path is provided
            if writer:
                writer.write(result_frame)
            
            frame_count += 1
            
            if show_progress:
                pbar.update(1)
                pbar.set_postfix({'Objects detected': total_objects})
        
        # Cleanup
        cap.release()
        if writer:
            writer.release()
        if show_progress:
            pbar.close()
        
        print(f"Video processing complete!")
        print(f"Total frames processed: {frame_count}")
        print(f"Total objects detected: {total_objects}")
        
        if output_path:
            print(f"Output video saved to: {output_path}")
        
        return total_objects

def create_output_directories():
    """Create output directories for saving results"""
    output_dir = Path("output")
    output_dir.mkdir(exist_ok=True)
    
    # Create subdirectories
    (output_dir / "images").mkdir(exist_ok=True)
    (output_dir / "videos").mkdir(exist_ok=True)
    
    return output_dir

def parse_class_list(class_string):
    """Parse class list from string (e.g., '2,3,5,7' -> [2,3,5,7])"""
    if not class_string:
        return None
    return [int(x.strip()) for x in class_string.split(',')]

def main():
    parser = argparse.ArgumentParser(description="Object Detection using YOLO")
    parser.add_argument("--model", type=str, default="yolov9c.pt",
                       help="Path to YOLO model file")
    parser.add_argument("--input", type=str, required=True,
                       help="Path to input image or video")
    parser.add_argument("--output", type=str, default=None,
                       help="Path to output file (optional)")
    parser.add_argument("--classes", type=str, default=None,
                       help="Comma-separated list of class IDs to detect (e.g., '2,3,5,7'). Leave empty for all classes.")
    parser.add_argument("--conf", type=float, default=0.5,
                       help="Confidence threshold")
    parser.add_argument("--iou", type=float, default=0.45,
                       help="IoU threshold")
    
    args = parser.parse_args()
    
    # Create output directories
    output_dir = create_output_directories()
    
    # Parse target classes
    target_classes = parse_class_list(args.classes)
    
    # Initialize detector
    detector = ObjectDetector(args.model, target_classes, args.conf, args.iou)
    
    # Determine if input is image or video
    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Error: Input file {args.input} does not exist!")
        return
    
    # Generate output path if not provided
    if args.output is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        if input_path.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp']:
            output_path = output_dir / "images" / f"detected_{timestamp}_{input_path.name}"
        else:
            output_path = output_dir / "videos" / f"detected_{timestamp}_{input_path.name}"
    else:
        output_path = args.output
    
    # Process input
    try:
        if input_path.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp']:
            # Process image
            detector.process_image(str(input_path), str(output_path))
        else:
            # Process video
            detector.process_video(str(input_path), str(output_path))
            
    except Exception as e:
        print(f"Error processing {args.input}: {e}")

if __name__ == "__main__":
    main() 