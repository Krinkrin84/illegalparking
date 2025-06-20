#!/bin/bash

# Car Detection Docker Runner Script

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

print_header() {
    echo -e "${BLUE}================================${NC}"
    echo -e "${BLUE}  Car Detection Docker Runner  ${NC}"
    echo -e "${BLUE}================================${NC}"
}

# Function to check if Docker is installed
check_docker() {
    if ! command -v docker &> /dev/null; then
        print_error "Docker is not installed. Please install Docker first."
        exit 1
    fi
    
    if ! command -v docker-compose &> /dev/null; then
        print_error "Docker Compose is not installed. Please install Docker Compose first."
        exit 1
    fi
    
    print_status "Docker and Docker Compose are installed."
}

# Function to check if NVIDIA Docker is available
check_nvidia_docker() {
    if command -v nvidia-docker &> /dev/null || docker info 2>/dev/null | grep -q nvidia; then
        print_status "NVIDIA Docker support detected."
        return 0
    else
        print_warning "NVIDIA Docker not detected. Will use CPU-only mode."
        return 1
    fi
}

# Function to build the Docker image
build_image() {
    print_status "Building Docker image..."
    docker-compose build
    print_status "Docker image built successfully."
}

# Function to run with GPU support
run_gpu() {
    print_status "Running with GPU support..."
    docker-compose up car-detection
}

# Function to run with CPU only
run_cpu() {
    print_status "Running with CPU only..."
    docker-compose up car-detection-cpu
}

# Function to run custom command
run_custom() {
    local input_file=$1
    local model_file=$2
    local output_file=$3
    
    if [ -z "$input_file" ]; then
        print_error "Input file is required."
        echo "Usage: $0 custom <input_file> [model_file] [output_file]"
        exit 1
    fi
    
    if [ ! -f "$input_file" ]; then
        print_error "Input file '$input_file' does not exist."
        exit 1
    fi
    
    # Set default model if not provided
    if [ -z "$model_file" ]; then
        model_file="yolov9c.pt"
    fi
    
    print_status "Running custom command..."
    print_status "Input: $input_file"
    print_status "Model: $model_file"
    if [ ! -z "$output_file" ]; then
        print_status "Output: $output_file"
    fi
    
    # Build command
    cmd="python tracking_detector.py --input /app/videos/$(basename $input_file) --model $model_file"
    if [ ! -z "$output_file" ]; then
        cmd="$cmd --output /app/output/$(basename $output_file)"
    fi
    
    # Run with appropriate service
    if check_nvidia_docker; then
        docker-compose run --rm car-detection $cmd
    else
        docker-compose run --rm car-detection-cpu $cmd
    fi
}

# Function to show help
show_help() {
    echo "Usage: $0 [COMMAND] [OPTIONS]"
    echo ""
    echo "Commands:"
    echo "  build     Build the Docker image"
    echo "  gpu       Run with GPU support"
    echo "  cpu       Run with CPU only"
    echo "  custom    Run custom detection command"
    echo "  help      Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0 build"
    echo "  $0 gpu"
    echo "  $0 cpu"
    echo "  $0 custom videos/BusyParkingLotUAVVideo.mp4"
    echo "  $0 custom videos/BusyParkingLotUAVVideo.mp4 yolov9c.pt output.mp4"
    echo ""
    echo "Notes:"
    echo "  - Input files should be in the 'videos' directory"
    echo "  - Output files will be saved in the 'output' directory"
    echo "  - Model files should be in the current directory"
}

# Main script
main() {
    print_header
    
    # Check Docker installation
    check_docker
    
    # Parse command
    case "${1:-help}" in
        "build")
            build_image
            ;;
        "gpu")
            if check_nvidia_docker; then
                build_image
                run_gpu
            else
                print_error "NVIDIA Docker not available. Use 'cpu' mode instead."
                exit 1
            fi
            ;;
        "cpu")
            build_image
            run_cpu
            ;;
        "custom")
            run_custom "$2" "$3" "$4"
            ;;
        "help"|*)
            show_help
            ;;
    esac
}

# Run main function with all arguments
main "$@" 