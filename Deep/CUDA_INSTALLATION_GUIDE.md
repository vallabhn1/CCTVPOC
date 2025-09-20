# CUDA Installation Guide for Real-time Heatmap

## ✅ Your System Status
- **GPU Driver**: 566.07 ✅ (Excellent!)
- **CUDA Version**: 12.7 ✅ (Latest!)
- **Required**: PyTorch with CUDA 12.x support

## 🚨 Important: GPU Required

The real-time heatmap system **requires** NVIDIA GPU with CUDA support. It will **not** run on CPU-only systems.

## 🎯 Quick Setup for Your System

Since you already have NVIDIA Driver 566.07 and CUDA 12.7, you just need to install the correct PyTorch version:

### Install PyTorch with CUDA 12.x Support
```cmd
# Uninstall CPU-only PyTorch first
pip uninstall torch torchvision torchaudio

# Install CUDA 12.x-enabled PyTorch (for your CUDA 12.7)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

### Verify Installation
```cmd
# Navigate to the Deep folder
cd "c:\Users\deepv\OneDrive\Desktop\Hive Dynamics\Deep"

# Run the GPU test
python gpu_test.py
```

You should see:
```
✅ GPU Available: [Your GPU Name]
🔥 CUDA Version: 12.1+ (compatible with your 12.7)
💾 GPU Memory: [X.X] GB
🚀 GPU Driver Version: 566.07 (Compatible with CUDA 12.7)
```

## 🚀 Running the Real-time Heatmap

Once CUDA is installed and verified:

```cmd
# Interactive mode (side-by-side display)
python realtime_heatmap.py

# Save output video
python realtime_heatmap.py --output my_heatmap.mp4

# Custom settings
python realtime_heatmap.py --grid-size 20 --decay 0.95 --conf-thres 0.3
```

## 🔧 Troubleshooting

### Problem: "CUDA not available"
**Solution:**
1. Check `nvidia-smi` works
2. Reinstall PyTorch with CUDA: `pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118`
3. Restart Python/terminal

### Problem: "RuntimeError: CUDA out of memory"
**Solutions:**
1. Close other applications using GPU
2. Use larger grid size: `--grid-size 40`
3. Lower video resolution
4. Restart the script

### Problem: "Model failed to load on GPU"
**Solutions:**
1. Check GPU memory: `nvidia-smi`
2. Restart the system
3. Update GPU drivers

### Problem: Slow performance
**Solutions:**
1. Use optimal grid size: `--grid-size 25-30`
2. Check GPU utilization: `nvidia-smi`
3. Ensure no CPU fallback in logs

## 📊 Performance Tips

### Optimal Settings for Different GPUs:

**High-end GPU (RTX 3080/4080+, 8GB+ VRAM):**
```cmd
python realtime_heatmap.py --grid-size 20 --decay 0.98 --conf-thres 0.2
```

**Mid-range GPU (GTX 1660/RTX 3060, 4-6GB VRAM):**
```cmd
python realtime_heatmap.py --grid-size 30 --decay 0.98 --conf-thres 0.25
```

**Entry-level GPU (GTX 1050/1650, 2-4GB VRAM):**
```cmd
python realtime_heatmap.py --grid-size 40 --decay 0.99 --conf-thres 0.3
```

## 🎮 Interactive Controls

During playback:
- **'q'**: Quit the application
- **'r'**: Reset heatmap (clear accumulated heat)

## 📁 Output Files

- **Side-by-side video**: Original + heatmap when using `--output`
- **Live display**: Real-time window when not using `--no-display`

## 💡 Alternative Solutions

If you cannot install CUDA:
1. Use the original `cctv.py` script (has CPU fallback)
2. Consider cloud GPU services (Google Colab, AWS)
3. Use a different machine with NVIDIA GPU

---

**Remember:** The real-time heatmap specifically requires GPU for optimal performance. The side-by-side real-time visualization is computationally intensive and designed for GPU acceleration.
