@echo off
echo Starting Mall CCTV Analytics System...
echo.

REM Check if Python is available
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo Error: Python is not installed or not in PATH
    pause
    exit /b 1
)

REM Check if required packages are installed
echo Checking dependencies...
python -c "import torch, cv2, numpy, matplotlib, seaborn" >nul 2>&1
if %errorlevel% neq 0 (
    echo Installing required packages...
    pip install -r requirements.txt
)

REM Check GPU availability
echo Checking GPU availability...
python -c "import torch; print(f'CUDA Available: {torch.cuda.is_available()}'); print(f'GPU Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"CPU Only\"}')"
echo.

REM Run the analytics system
echo Starting analytics...
python cctv.py

echo.
echo Analytics completed! Check the output files:
echo - mall_crowd_heatmap.png (Heatmap visualization)
echo - mall_analytics_report.json (Detailed report)
echo - output_analytics.mp4 (Annotated video)
echo.
pause
