# Use Python 3.10 for better PyTorch CUDA support
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    wget \
    curl \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    libgcc-s1 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Install ByteTracker
RUN pip install --no-cache-dir byte-track

# Install PyTorch with CUDA support (if available)
RUN pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cu121

# Copy application code
COPY . .

# Create output directories
RUN mkdir -p output/images output/videos

# Set environment variables
ENV PYTHONPATH=/app
ENV CUDA_VISIBLE_DEVICES=0

# Expose port (if needed for web interface)
EXPOSE 8080

# Default command
CMD ["python", "tracking_detector.py", "--help"] 