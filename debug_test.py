import cv2
import torch
from ultralytics import YOLO
import numpy as np

def test_detection():
    print("Testing basic YOLO detection...")
    
    # Load model
    model = YOLO("yolov9c.pt")
    print("Model loaded successfully!")
    
    # Load test image
    image = cv2.imread("test_image.jpg")
    if image is None:
        print("Error: Cannot read test_image.jpg")
        return
    
    print(f"Image shape: {image.shape}")
    
    # Run detection with lower confidence
    results = model.predict(image, conf=0.1, iou=0.7, verbose=False, show=False)
    
    # Extract all detections
    all_detections = []
    car_detections = []
    for result in results:
        boxes = result.boxes
        if boxes is not None:
            print(f"Found {len(boxes)} total detections")
            for i in range(len(boxes)):
                box = boxes[i]
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                conf = box.conf[0].cpu().numpy()
                cls = int(box.cls[0].cpu().numpy())
                
                print(f"Detection {i}: class={cls}, conf={conf:.3f}, bbox=[{x1:.1f}, {y1:.1f}, {x2:.1f}, {y2:.1f}]")
                all_detections.append([x1, y1, x2, y2, conf, cls])
                
                # Only detect cars (class 2 in COCO)
                if cls == 2:
                    car_detections.append([x1, y1, x2, y2, conf])
    
    print(f"Found {len(car_detections)} cars out of {len(all_detections)} total detections")
    
    # Draw all detections
    result_image = image.copy()
    for det in all_detections:
        x1, y1, x2, y2, conf, cls = det
        bbox = [int(x1), int(y1), int(x2), int(y2)]
        
        # Choose color based on class
        if cls == 2:  # Car
            color = (0, 255, 0)  # Green
            label = f"Car: {conf:.2f}"
        else:
            color = (255, 0, 0)  # Red
            label = f"Class{cls}: {conf:.2f}"
        
        # Draw bounding box
        cv2.rectangle(result_image, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, 2)
        
        # Draw label
        cv2.putText(result_image, label, (bbox[0], bbox[1] - 10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    
    # Save result
    cv2.imwrite("debug_output.jpg", result_image)
    print("Saved debug_output.jpg")

if __name__ == "__main__":
    test_detection() 