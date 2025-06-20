# Docker Deployment for Car Detection and Tracking

This guide explains how to deploy the car detection and tracking application using Docker containers.

## Prerequisites

### 1. Install Docker
- **Windows/Mac**: Install [Docker Desktop](https://www.docker.com/products/docker-desktop)
- **Linux**: Install [Docker Engine](https://docs.docker.com/engine/install/)

### 2. Install Docker Compose
- Usually included with Docker Desktop
- For Linux: `sudo apt-get install docker-compose`

### 3. NVIDIA Docker (Optional, for GPU support)
- Install [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html)
- Requires NVIDIA GPU with CUDA support

## Quick Start

### 1. Build the Docker Image
```bash
# Linux/Mac
./run_docker.sh build

# Windows
run_docker.bat build

# Or manually
docker-compose build
```

### 2. Run with GPU Support
```bash
# Linux/Mac
./run_docker.sh gpu

# Windows
run_docker.bat gpu

# Or manually
docker-compose up car-detection
```

### 3. Run with CPU Only
```bash
# Linux/Mac
./run_docker.sh cpu

# Windows
run_docker.bat cpu

# Or manually
docker-compose up car-detection-cpu
```

## Usage Examples

### Process a Video File
```bash
# Linux/Mac
./run_docker.sh custom videos/BusyParkingLotUAVVideo.mp4

# Windows
run_docker.bat custom videos\BusyParkingLotUAVVideo.mp4
```

### Process with Custom Model and Output
```bash
# Linux/Mac
./run_docker.sh custom videos/BusyParkingLotUAVVideo.mp4 yolov9c.pt output.mp4

# Windows
run_docker.bat custom videos\BusyParkingLotUAVVideo.mp4 yolov9c.pt output.mp4
```

### Manual Docker Commands
```bash
# Run detection on a video
docker-compose run --rm car-detection python tracking_detector.py \
    --input /app/videos/BusyParkingLotUAVVideo.mp4 \
    --model yolov9c.pt \
    --output /app/output/detected_video.mp4

# Run with custom parameters
docker-compose run --rm car-detection python tracking_detector.py \
    --input /app/videos/BusyParkingLotUAVVideo.mp4 \
    --model yolov9c.pt \
    --conf 0.6 \
    --iou 0.4 \
    --track-thresh 0.5
```

## Directory Structure

```
project/
├── Dockerfile                 # Docker image definition
├── docker-compose.yml         # Docker Compose configuration
├── run_docker.sh             # Linux/Mac runner script
├── run_docker.bat            # Windows runner script
├── requirements.txt          # Python dependencies
├── tracking_detector.py      # Main application
├── videos/                   # Input videos (mounted)
│   └── BusyParkingLotUAVVideo.mp4
├── output/                   # Output files (mounted)
│   ├── images/
│   └── videos/
└── models/                   # Model files (mounted)
    └── yolov9c.pt
```

## Docker Configuration

### Dockerfile Features
- **Base Image**: Python 3.10-slim (better PyTorch CUDA support)
- **System Dependencies**: OpenCV, CUDA libraries
- **Python Dependencies**: PyTorch, Ultralytics, ByteTracker
- **Volume Mounts**: Input videos, output files, model files
- **GPU Support**: CUDA 12.1 compatible

### Docker Compose Services

#### GPU Service (`car-detection`)
```yaml
services:
  car-detection:
    build: .
    volumes:
      - ./videos:/app/videos:ro      # Read-only input
      - ./output:/app/output         # Writable output
      - ./models:/app/models:ro      # Read-only models
    environment:
      - CUDA_VISIBLE_DEVICES=0       # Use first GPU
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia          # NVIDIA GPU support
              count: 1
              capabilities: [gpu]
```

#### CPU Service (`car-detection-cpu`)
```yaml
services:
  car-detection-cpu:
    build: .
    volumes:
      - ./videos:/app/videos:ro
      - ./output:/app/output
      - ./models:/app/models:ro
    environment:
      - CUDA_VISIBLE_DEVICES=""      # Disable GPU
```

## Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `CUDA_VISIBLE_DEVICES` | GPU device to use | `0` (GPU) or `""` (CPU) |
| `PYTHONPATH` | Python path | `/app` |

## Troubleshooting

### Common Issues

#### 1. Docker Build Fails
```bash
# Check Docker installation
docker --version
docker-compose --version

# Clean build cache
docker system prune -a
docker-compose build --no-cache
```

#### 2. GPU Not Available
```bash
# Check NVIDIA Docker
docker run --rm --gpus all nvidia/cuda:11.0-base nvidia-smi

# Install NVIDIA Container Toolkit
# Follow: https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html
```

#### 3. Permission Issues (Linux)
```bash
# Add user to docker group
sudo usermod -aG docker $USER
newgrp docker

# Or run with sudo
sudo docker-compose up
```

#### 4. Memory Issues
```bash
# Increase Docker memory limit in Docker Desktop settings
# Or use CPU-only mode
./run_docker.sh cpu
```

#### 5. Model Download Issues
```bash
# Download model manually
wget https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov9c.pt

# Or use a different model
./run_docker.sh custom videos/video.mp4 yolov8n.pt
```

### Performance Optimization

#### 1. GPU Memory
```bash
# Monitor GPU usage
nvidia-smi

# Limit GPU memory
export CUDA_VISIBLE_DEVICES=0
export CUDA_MEM_FRACTION=0.8
```

#### 2. CPU Performance
```bash
# Use CPU-only mode for small videos
./run_docker.sh cpu

# Adjust detection parameters
--conf 0.7 --iou 0.5
```

#### 3. Storage
```bash
# Use SSD for better I/O performance
# Mount volumes to fast storage
docker-compose run -v /fast/storage:/app/output car-detection
```

## Advanced Usage

### Custom Model Integration
```bash
# Copy your model to models directory
cp your_model.pt models/

# Use custom model
./run_docker.sh custom videos/video.mp4 models/your_model.pt
```

### Batch Processing
```bash
# Process multiple videos
for video in videos/*.mp4; do
    ./run_docker.sh custom "$video"
done
```

### Web Interface (Future)
```bash
# Expose web interface
docker-compose up -d
# Access at http://localhost:8080
```

## Monitoring and Logs

### View Container Logs
```bash
# View logs
docker-compose logs car-detection

# Follow logs in real-time
docker-compose logs -f car-detection

# View specific service logs
docker-compose logs car-detection-cpu
```

### Resource Monitoring
```bash
# Monitor container resources
docker stats

# Monitor GPU usage
nvidia-smi -l 1
```

## Security Considerations

### 1. Volume Mounts
- Input videos are mounted as read-only (`:ro`)
- Output directory is writable
- Model files are read-only

### 2. Network Access
- Container runs in isolated network
- No external network access by default
- Port 8080 exposed for future web interface

### 3. User Permissions
- Container runs as non-root user
- Limited file system access
- No sudo privileges

## Support

For issues and questions:
1. Check the troubleshooting section
2. Review Docker and NVIDIA documentation
3. Check application logs: `docker-compose logs`
4. Verify system requirements

## License

This Docker configuration is provided under the same license as the main project. 