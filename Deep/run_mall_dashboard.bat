@echo off
echo ====================================
echo Mall Analytics Real-time Dashboard
echo ====================================
echo.

REM Check if Python is available
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python is not installed or not in PATH
    echo Please install Python 3.8+ and ensure it's in your system PATH
    pause
    exit /b 1
)

REM Check CUDA availability
echo Checking GPU/CUDA availability...
python -c "import torch; print('GPU Available:', torch.cuda.is_available()); print('GPU Name:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'No GPU')" 2>nul
if errorlevel 1 (
    echo WARNING: Could not check CUDA availability
    echo Please ensure PyTorch with CUDA is installed
)

echo.
echo Starting Mall Analytics Dashboard...
echo Controls:
echo   'q' - Quit
echo   'r' - Reset analytics
echo   's' - Save current report
echo.

REM Run the dashboard with default settings
python mall_analytics_dashboard.py --video "Small video Mall shopping mall video shopping mall CCTV camera video no copyright full HD 4K video.mp4"

echo.
echo Dashboard processing completed.
pause
