#!/usr/bin/env python3
"""
Docker Test Script
Test if the Docker environment is properly configured
"""

import os
import sys
import subprocess

def test_docker_installation():
    """Test if Docker is installed and running"""
    print("=== Testing Docker Installation ===")
    
    try:
        # Test Docker version
        result = subprocess.run(['docker', '--version'], capture_output=True, text=True)
        if result.returncode == 0:
            print(f"✓ Docker installed: {result.stdout.strip()}")
        else:
            print("✗ Docker not installed or not accessible")
            return False
    except FileNotFoundError:
        print("✗ Docker not found in PATH")
        return False
    
    try:
        # Test Docker Compose
        result = subprocess.run(['docker-compose', '--version'], capture_output=True, text=True)
        if result.returncode == 0:
            print(f"✓ Docker Compose installed: {result.stdout.strip()}")
        else:
            print("✗ Docker Compose not installed")
            return False
    except FileNotFoundError:
        print("✗ Docker Compose not found in PATH")
        return False
    
    try:
        # Test if Docker daemon is running
        result = subprocess.run(['docker', 'info'], capture_output=True, text=True)
        if result.returncode == 0:
            print("✓ Docker daemon is running")
        else:
            print("✗ Docker daemon is not running")
            return False
    except Exception as e:
        print(f"✗ Error testing Docker daemon: {e}")
        return False
    
    return True

def test_nvidia_docker():
    """Test if NVIDIA Docker is available"""
    print("\n=== Testing NVIDIA Docker ===")
    
    try:
        # Test NVIDIA Docker
        result = subprocess.run(['docker', 'run', '--rm', '--gpus', 'all', 'nvidia/cuda:11.0-base', 'nvidia-smi'], 
                              capture_output=True, text=True, timeout=30)
        if result.returncode == 0:
            print("✓ NVIDIA Docker is available")
            print("GPU Information:")
            print(result.stdout)
            return True
        else:
            print("✗ NVIDIA Docker not available")
            print("This is normal if you don't have NVIDIA GPU or NVIDIA Container Toolkit")
            return False
    except subprocess.TimeoutExpired:
        print("✗ NVIDIA Docker test timed out")
        return False
    except Exception as e:
        print(f"✗ Error testing NVIDIA Docker: {e}")
        return False

def test_dockerfile():
    """Test if Dockerfile exists and is valid"""
    print("\n=== Testing Dockerfile ===")
    
    if not os.path.exists('Dockerfile'):
        print("✗ Dockerfile not found")
        return False
    
    print("✓ Dockerfile exists")
    
    # Check if docker-compose.yml exists
    if not os.path.exists('docker-compose.yml'):
        print("✗ docker-compose.yml not found")
        return False
    
    print("✓ docker-compose.yml exists")
    
    # Check if requirements.txt exists
    if not os.path.exists('requirements.txt'):
        print("✗ requirements.txt not found")
        return False
    
    print("✓ requirements.txt exists")
    
    return True

def test_project_structure():
    """Test if project structure is correct"""
    print("\n=== Testing Project Structure ===")
    
    required_files = [
        'tracking_detector.py',
        'car_detector.py',
        'object_detector_template.py'
    ]
    
    required_dirs = [
        'videos',
        'output'
    ]
    
    for file in required_files:
        if os.path.exists(file):
            print(f"✓ {file} exists")
        else:
            print(f"⚠ {file} not found")
    
    for dir in required_dirs:
        if os.path.exists(dir):
            print(f"✓ {dir}/ directory exists")
        else:
            print(f"⚠ {dir}/ directory not found")
            # Create directory
            try:
                os.makedirs(dir, exist_ok=True)
                print(f"  Created {dir}/ directory")
            except Exception as e:
                print(f"  Failed to create {dir}/ directory: {e}")
    
    return True

def main():
    """Main test function"""
    print("Docker Environment Test")
    print("=" * 50)
    
    all_tests_passed = True
    
    # Test Docker installation
    if not test_docker_installation():
        all_tests_passed = False
    
    # Test NVIDIA Docker
    nvidia_available = test_nvidia_docker()
    
    # Test Dockerfile
    if not test_dockerfile():
        all_tests_passed = False
    
    # Test project structure
    if not test_project_structure():
        all_tests_passed = False
    
    # Summary
    print("\n" + "=" * 50)
    print("SUMMARY")
    print("=" * 50)
    
    if all_tests_passed:
        print("✓ All basic tests passed!")
        print("\nYou can now build and run the Docker container:")
        print("\nFor Windows:")
        print("  run_docker.bat build")
        print("  run_docker.bat gpu    # If NVIDIA Docker available")
        print("  run_docker.bat cpu    # CPU only")
        print("\nFor Linux/Mac:")
        print("  ./run_docker.sh build")
        print("  ./run_docker.sh gpu   # If NVIDIA Docker available")
        print("  ./run_docker.sh cpu   # CPU only")
        
        if nvidia_available:
            print("\n✓ NVIDIA Docker is available - GPU acceleration will work!")
        else:
            print("\n⚠ NVIDIA Docker not available - will use CPU mode")
            
    else:
        print("✗ Some tests failed. Please check the issues above.")
        print("\nTo fix Docker issues:")
        print("1. Install Docker Desktop (Windows/Mac) or Docker Engine (Linux)")
        print("2. Start Docker daemon")
        print("3. For GPU support, install NVIDIA Container Toolkit")
    
    return all_tests_passed

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 