@echo off
echo Starting Real-time Traffic Heatmap System...
echo.

cd /d "c:\Users\deepv\OneDrive\Desktop\Hive Dynamics\Deep"

echo Activating Python environment...
call "..\\.venv\\Scripts\\activate.bat"

echo.
echo Checking GPU/CUDA availability...
echo.
python gpu_test.py

if %ERRORLEVEL% NEQ 0 (
    echo.
    echo ❌ GPU test failed! Real-time heatmap requires CUDA.
    echo 📝 Please install CUDA support first.
    echo.
    echo Press any key to exit...
    pause
    exit /b 1
)

echo.
echo ✅ GPU test passed! Starting real-time heatmap analysis...
echo.
echo 🎮 Controls during playback:
echo   • Press 'q' to quit
echo   • Press 'r' to reset heatmap
echo.

python realtime_heatmap.py --grid-size 25 --decay 0.98

echo.
echo Analysis complete! Press any key to exit...
pause
