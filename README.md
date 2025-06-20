# Car Detection and Parking Timer System

A comprehensive car detection and parking monitoring system using YOLO with real-time parking timer and illegal parking detection.

## Features

### 🚗 **Car Detection**
- Real-time car detection using YOLO models
- Support for YOLOv8, YOLOv9, and other YOLO variants
- Configurable confidence and IoU thresholds
- GPU acceleration support

### ⏱️ **Parking Timer System**
- Real-time parking duration tracking
- Individual timers for each car ID
- Visual indicators for parked vs. moving cars
- Configurable parking thresholds

### 🚨 **Illegal Parking Detection**
- Automatic detection of cars exceeding allowed parking time
- Visual alerts with red bounding boxes for illegal parking
- Historical violation tracking
- Configurable time limits

### 🎯 **Object Tracking**
- Persistent car tracking across video frames
- Unique ID assignment for each car
- Track continuity even with temporary occlusions
- Optional tracking mode

### 📊 **Comprehensive Reporting**
- Parking results sorted by duration
- Illegal parking violation history
- Summary statistics (total, average, longest parking times)
- Real-time console output

## Installation

### Prerequisites
- Python 3.8+
- CUDA-compatible GPU (optional, for acceleration)

### Install Dependencies
```bash
pip install -r requirements.txt
```

### Required Packages
```
ultralytics
opencv-python
torch
torchvision
numpy
tqdm
```

## Configuration

All important parameters are configurable via global variables at the top of `car_detector.py`:

```python
#Configuration
ALLOWED_PARKING_TIME = 40  # seconds - time limit before illegal parking
POSITION_THRESHOLD = 300   # pixels - max movement to consider stationary
PARKING_THRESHOLD = 30     # frames to wait before counting as parked
CONFIDENCE_THRESHOLD = 0.45 # minimum confidence for car detection
IOU_THRESHOLD = 0.45       # IoU threshold for NMS
DEFAULT_MODEL = "yolov9c.pt" # default YOLO model file
```

## Usage

### Basic Usage
```bash
# Process video with parking timer
python car_detector.py --input videos/carpark1.mp4 --show-timer

# Process image
python car_detector.py --input test_image.jpg --show-timer

# Custom output path
python car_detector.py --input video.mp4 --output output/result.mp4 --show-timer
```

### Command Line Options
```bash
python car_detector.py [OPTIONS]

Options:
  --model MODEL           YOLO model file (default: yolov9c.pt)
  --input INPUT           Input image or video (required)
  --output OUTPUT         Output file path (optional)
  --conf CONF             Confidence threshold (default: 0.45)
  --iou IOU               IoU threshold (default: 0.45)
  --no-track              Disable tracking mode
  --parking-threshold N   Frames to wait before counting as parked (default: 30)
  --show-timer            Display parking time on video
```

### Examples

#### Process Video with Timer
```bash
python car_detector.py --input videos/carpark2.mp4 --output ./output/videos/2.mp4 --model yolov9c.pt --show-timer
```

#### Detection Only (No Tracking)
```bash
python car_detector.py --input video.mp4 --no-track --show-timer
```

#### Custom Thresholds
```bash
python car_detector.py --input video.mp4 --conf 0.6 --iou 0.5 --parking-threshold 45 --show-timer
```

## Visual Indicators

### Bounding Box Colors
- **🟢 Green**: Moving cars
- **🟡 Yellow**: Legal parking (under time limit)
- **🔴 Red**: Illegal parking (exceeded time limit)

### Label Format
```
Car #123: 0.95 | Parked: 2m 30s
Car #456: 0.87 | ILLEGAL: 5m 15s
Car #789: 0.92 | Moving
```

## Output

### Video Output
- Processed videos saved to `output/videos/` directory
- Timestamped filenames for easy organization
- Maintains original video quality and frame rate

### Console Output
```
============================================================
PARKING RESULTS (Sorted by Parking Time)
============================================================
Car ID   Status     Parking Time     Start Time           
------------------------------------------------------------
#123     ILLEGAL    5m 30s          14:25:30            
#456     Parked     3m 15s          14:27:45            
#789     Parked     1m 45s          14:29:20            
------------------------------------------------------------
Total parked cars: 3
Total parking time: 10m 30s
Average parking time: 3m 30s
Longest parking time: 5m 30s
Illegal parking violations: 1
============================================================
```

### Historical Violations
```
============================================================
HISTORICAL ILLEGAL PARKING VIOLATIONS
============================================================
Car ID   Duration     Start Time        Violation Time    
------------------------------------------------------------
#123     5m 30s       14:25:30          14:30:00          
------------------------------------------------------------
Total violations: 1
============================================================
```

## How It Works

### 1. **Detection Phase**
- YOLO model detects cars in each frame
- Filters for car class (class 2 in COCO dataset)
- Applies confidence and IoU thresholds

### 2. **Tracking Phase**
- Assigns unique track IDs to detected cars
- Maintains car identity across frames
- Handles occlusions and reappearances

### 3. **Stationary Detection**
- Monitors car position changes
- Considers car "stationary" if movement < POSITION_THRESHOLD
- Requires PARKING_THRESHOLD frames of stability

### 4. **Parking Timer**
- Starts counting when car becomes stationary
- Resets timer when car moves significantly
- Tracks individual parking duration per car

### 5. **Illegal Parking Detection**
- Monitors parking duration against ALLOWED_PARKING_TIME
- Marks cars as "illegal" when limit exceeded
- Records violations for reporting

## Configuration Guide

### For Different Scenarios

#### **Strict Parking Detection**
```python
ALLOWED_PARKING_TIME = 30  # 30 seconds limit
POSITION_THRESHOLD = 100   # 100 pixels movement
PARKING_THRESHOLD = 20     # 20 frames to start parking
CONFIDENCE_THRESHOLD = 0.6 # Higher confidence required
```

#### **Lenient Detection**
```python
ALLOWED_PARKING_TIME = 60  # 60 seconds limit
POSITION_THRESHOLD = 500   # 500 pixels movement
PARKING_THRESHOLD = 50     # 50 frames to start parking
CONFIDENCE_THRESHOLD = 0.3 # Lower confidence threshold
```

#### **High Traffic Areas**
```python
ALLOWED_PARKING_TIME = 20  # Short parking limit
POSITION_THRESHOLD = 200   # Moderate movement allowance
PARKING_THRESHOLD = 15     # Quick parking detection
IOU_THRESHOLD = 0.6        # Aggressive duplicate removal
```

## Troubleshooting

### Common Issues

#### **No Cars Detected**
- Lower `CONFIDENCE_THRESHOLD` (try 0.2-0.3)
- Check if model file exists
- Verify input video/image quality

#### **False Parking Detection**
- Increase `POSITION_THRESHOLD` for camera jitter
- Increase `PARKING_THRESHOLD` for more stability
- Check for moving objects in background

#### **Multiple Detections per Car**
- Increase `IOU_THRESHOLD` (try 0.5-0.7)
- Lower `CONFIDENCE_THRESHOLD`
- Check for overlapping cars

#### **Poor Tracking**
- Ensure `--no-track` is not used
- Check video frame rate and quality
- Verify model compatibility

## Performance Tips

### **For Real-time Processing**
- Use smaller YOLO models (yolov8n, yolov9n)
- Lower resolution input videos
- Enable GPU acceleration

### **For High Accuracy**
- Use larger YOLO models (yolov8x, yolov9x)
- Higher resolution inputs
- Increase confidence thresholds

### **For Large Videos**
- Process in chunks
- Use SSD storage for output
- Monitor memory usage

## File Structure

```
cardetection/
├── car_detector.py          # Main detection script
├── requirements.txt         # Python dependencies
├── README.md               # This file
├── videos/                 # Input videos
│   ├── carpark1.mp4
│   ├── carpark2.mp4
│   └── carpark3.mp4
├── output/                 # Output directory
│   ├── images/            # Processed images
│   └── videos/            # Processed videos
└── weight/                # Model weights
    └── yolov9c.pt
```

## License

This project is open source and available under the MIT License.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Support

For issues and questions, please open an issue on the GitHub repository.
