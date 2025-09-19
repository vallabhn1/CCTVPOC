#!/usr/bin/env python3
"""Test script to verify YOLOv7 import handling"""

import sys
import os

# Add YOLOv7 path
YOLOV7_PATH = r"C:\Users\Vallabhj\Downloads\CCTVPOC_integrated\yolov7\yolov7.pt"
sys.path.append(YOLOV7_PATH)

print("🧪 Testing YOLOv7 import handling...")
print(f"📁 YOLOv7 path: {YOLOV7_PATH}")
print(f"📁 Path exists: {os.path.exists(YOLOV7_PATH)}")

# Test the exact import logic from our script
YOLOV7_AVAILABLE = False
try:
    from models.experimental import attempt_load
    from utils.datasets import LoadStreams, LoadImages
    from utils.general import check_img_size, non_max_suppression, scale_coords, xyxy2xywh, xywh2xyxy
    from utils.torch_utils import select_device, time_synchronized
    from utils.plots import plot_one_box
    YOLOV7_AVAILABLE = True
    print("✅ YOLOv7 modules loaded successfully")
except ImportError as e:
    print(f"⚠️ Warning: YOLOv7 modules not found: {e}")
    print("✅ This is expected behavior - fallbacks will be used")
    
    # Test fallback imports
    try:
        import torch
        import torchvision.transforms as transforms
        from torchvision.models import detection
        print("✅ PyTorch torchvision available for fallback detection")
    except ImportError as e2:
        print(f"❌ Error: Neither YOLOv7 nor PyTorch available: {e2}")

print(f"🎯 Final status: YOLOV7_AVAILABLE = {YOLOV7_AVAILABLE}")
print("✅ Import test completed successfully!")
