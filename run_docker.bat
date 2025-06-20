@echo off
setlocal enabledelayedexpansion

REM Car Detection Docker Runner Script for Windows

echo ================================
echo   Car Detection Docker Runner  
echo ================================

REM Check if Docker is installed
docker --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Docker is not installed. Please install Docker Desktop first.
    exit /b 1
)

docker-compose --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Docker Compose is not installed. Please install Docker Compose first.
    exit /b 1
)

echo [INFO] Docker and Docker Compose are installed.

REM Check command line arguments
if "%1"=="" goto :show_help
if "%1"=="help" goto :show_help
if "%1"=="build" goto :build_image
if "%1"=="gpu" goto :run_gpu
if "%1"=="cpu" goto :run_cpu
if "%1"=="custom" goto :run_custom
goto :show_help

:build_image
echo [INFO] Building Docker image...
docker-compose build
if errorlevel 1 (
    echo [ERROR] Failed to build Docker image.
    exit /b 1
)
echo [INFO] Docker image built successfully.
goto :end

:run_gpu
echo [INFO] Running with GPU support...
call :build_image
docker-compose up car-detection
goto :end

:run_cpu
echo [INFO] Running with CPU only...
call :build_image
docker-compose up car-detection-cpu
goto :end

:run_custom
if "%2"=="" (
    echo [ERROR] Input file is required.
    echo Usage: %0 custom ^<input_file^> [model_file] [output_file]
    exit /b 1
)

if not exist "%2" (
    echo [ERROR] Input file '%2' does not exist.
    exit /b 1
)

if "%3"=="" (
    set "model_file=yolov9c.pt"
) else (
    set "model_file=%3"
)

echo [INFO] Running custom command...
echo [INFO] Input: %2
echo [INFO] Model: !model_file!
if not "%4"=="" echo [INFO] Output: %4

REM Build command
set "cmd=python tracking_detector.py --input /app/videos/%~nx2 --model !model_file!"
if not "%4"=="" set "cmd=!cmd! --output /app/output/%~nx4"

REM Run with GPU support if available
docker info 2>nul | findstr /i nvidia >nul
if errorlevel 1 (
    echo [WARNING] NVIDIA Docker not detected. Using CPU-only mode.
    docker-compose run --rm car-detection-cpu !cmd!
) else (
    echo [INFO] NVIDIA Docker support detected. Using GPU mode.
    docker-compose run --rm car-detection !cmd!
)
goto :end

:show_help
echo Usage: %0 [COMMAND] [OPTIONS]
echo.
echo Commands:
echo   build     Build the Docker image
echo   gpu       Run with GPU support
echo   cpu       Run with CPU only
echo   custom    Run custom detection command
echo   help      Show this help message
echo.
echo Examples:
echo   %0 build
echo   %0 gpu
echo   %0 cpu
echo   %0 custom videos\BusyParkingLotUAVVideo.mp4
echo   %0 custom videos\BusyParkingLotUAVVideo.mp4 yolov9c.pt output.mp4
echo.
echo Notes:
echo   - Input files should be in the 'videos' directory
echo   - Output files will be saved in the 'output' directory
echo   - Model files should be in the current directory
goto :end

:end
endlocal 