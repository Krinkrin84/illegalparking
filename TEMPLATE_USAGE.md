# Object Detection Template Usage Guide

This template allows you to detect any objects using YOLO models with configurable classes.

## Quick Start

### 1. Download a YOLO Model
First, download an official YOLO model:
```bash
yolo download model=yolov9c.pt
```

### 2. Basic Usage Examples

#### Detect ALL objects in a video:
```bash
python object_detector_template.py --input videos/BusyParkingLotUAVVideo.mp4 --model yolov9c.pt
```

#### Detect only CARS (class 2):
```bash
python object_detector_template.py --input videos/BusyParkingLotUAVVideo.mp4 --model yolov9c.pt --classes "2"
```

#### Detect VEHICLES (cars, motorcycles, buses, trucks):
```bash
python object_detector_template.py --input videos/BusyParkingLotUAVVideo.mp4 --model yolov9c.pt --classes "2,3,5,7"
```

#### Detect PEOPLE and ANIMALS:
```bash
python object_detector_template.py --input videos/BusyParkingLotUAVVideo.mp4 --model yolov9c.pt --classes "0,14,15,16,17,18,19,20,21,22,23"
```

## Common Class IDs

| Class ID | Object | Class ID | Object |
|----------|--------|----------|--------|
| 0 | person | 14 | bird |
| 1 | bicycle | 15 | cat |
| 2 | car | 16 | dog |
| 3 | motorcycle | 17 | horse |
| 4 | airplane | 18 | sheep |
| 5 | bus | 19 | cow |
| 6 | train | 20 | elephant |
| 7 | truck | 21 | bear |
| 8 | boat | 22 | zebra |
| 9 | traffic light | 23 | giraffe |
| 10 | fire hydrant | 24 | backpack |
| 11 | stop sign | 25 | umbrella |
| 12 | parking meter | 26 | handbag |
| 13 | bench | 27 | tie |

## Advanced Usage

### Custom Confidence Threshold
```bash
python object_detector_template.py --input video.mp4 --model yolov9c.pt --classes "2,3" --conf 0.7
```

### Custom IoU Threshold
```bash
python object_detector_template.py --input video.mp4 --model yolov9c.pt --classes "2,3" --iou 0.3
```

### Specify Output File
```bash
python object_detector_template.py --input video.mp4 --model yolov9c.pt --classes "2" --output my_output.mp4
```

## Popular Detection Scenarios

### 1. Vehicle Detection
```bash
python object_detector_template.py --input video.mp4 --model yolov9c.pt --classes "2,3,5,7" --conf 0.6
```
- Class 2: Car
- Class 3: Motorcycle  
- Class 5: Bus
- Class 7: Truck

### 2. People Detection
```bash
python object_detector_template.py --input video.mp4 --model yolov9c.pt --classes "0" --conf 0.5
```
- Class 0: Person

### 3. Animal Detection
```bash
python object_detector_template.py --input video.mp4 --model yolov9c.pt --classes "14,15,16,17,18,19,20,21,22,23" --conf 0.6
```
- Classes 14-23: Various animals

### 4. Traffic Objects
```bash
python object_detector_template.py --input video.mp4 --model yolov9c.pt --classes "2,3,5,7,9,11" --conf 0.5
```
- Classes 2,3,5,7: Vehicles
- Class 9: Traffic light
- Class 11: Stop sign

### 5. Personal Items
```bash
python object_detector_template.py --input video.mp4 --model yolov9c.pt --classes "24,25,26,27,28,67" --conf 0.5
```
- Class 24: Backpack
- Class 25: Umbrella
- Class 26: Handbag
- Class 27: Tie
- Class 28: Suitcase
- Class 67: Cell phone

## Customizing Colors

You can modify the colors for different classes in the `draw_detections` method:

```python
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
```

## Custom Class Names

If you're using a custom model with different classes, modify the `class_names` dictionary in the `__init__` method:

```python
self.class_names = {
    0: 'your_class_1',
    1: 'your_class_2',
    2: 'your_class_3',
    # ... add all your classes
}
```

## Command Line Options

| Option | Description | Example |
|--------|-------------|---------|
| `--model` | Path to YOLO model file | `yolov9c.pt` |
| `--input` | Input image or video file | `video.mp4` |
| `--output` | Output file path (optional) | `output.mp4` |
| `--classes` | Comma-separated class IDs | `"2,3,5,7"` |
| `--conf` | Confidence threshold (0.1-1.0) | `0.6` |
| `--iou` | IoU threshold (0.1-1.0) | `0.45` |

## Tips

1. **Higher confidence** (0.7-0.9) = Fewer but more accurate detections
2. **Lower confidence** (0.3-0.5) = More detections but may include false positives
3. **Multiple classes** = Use comma-separated values: `"2,3,5,7"`
4. **All classes** = Leave `--classes` empty or don't specify it
5. **Custom model** = Make sure your model file is compatible with Ultralytics

## Troubleshooting

- **Model not found**: Download the model first using `yolo download model=yolov9c.pt`
- **No detections**: Try lowering the confidence threshold with `--conf 0.3`
- **Too many detections**: Try raising the confidence threshold with `--conf 0.7`
- **Wrong classes**: Check the class IDs in the table above or use `--classes ""` to detect all classes 