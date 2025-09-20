@echo off
echo Installing PyTorch with CUDA 12.x support for your system...
echo.
echo Your system specs:
echo - NVIDIA Driver: 566.07
echo - CUDA Version: 12.7
echo - Required: PyTorch with CUDA 12.x
echo.

echo Uninstalling existing PyTorch versions...
pip uninstall torch torchvision torchaudio -y

echo.
echo Installing PyTorch with CUDA 12.x support...
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

echo.
echo Testing CUDA installation...
python -c "import torch; print('CUDA Available:', torch.cuda.is_available()); print('CUDA Version:', torch.version.cuda); print('GPU Name:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A')"

echo.
echo Running comprehensive GPU test...
python gpu_test.py

echo.
echo Installation complete! You can now run the real-time heatmap.
echo Press any key to exit...
pause
