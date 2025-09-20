#!/usr/bin/env python3
"""
Quick test script for Mall Analytics Dashboard
"""

import os
import sys

# Add the current directory to Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

def test_imports():
    """Test all required imports"""
    print("🔍 Testing imports...")
    
    try:
        import torch
        print(f"✅ PyTorch {torch.__version__}")
        print(f"✅ CUDA Available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"✅ GPU: {torch.cuda.get_device_name(0)}")
    except ImportError as e:
        print(f"❌ PyTorch import failed: {e}")
        return False
    
    try:
        import cv2
        print(f"✅ OpenCV {cv2.__version__}")
    except ImportError as e:
        print(f"❌ OpenCV import failed: {e}")
        return False
    
    try:
        import numpy as np
        print(f"✅ NumPy {np.__version__}")
    except ImportError as e:
        print(f"❌ NumPy import failed: {e}")
        return False
    
    try:
        import matplotlib
        print(f"✅ Matplotlib {matplotlib.__version__}")
    except ImportError as e:
        print(f"❌ Matplotlib import failed: {e}")
        return False
    
    return True

def test_dashboard_class():
    """Test dashboard class initialization"""
    print("\n🔍 Testing dashboard class...")
    
    try:
        from mall_analytics_dashboard import MallAnalyticsDashboard, PersonData, TrafficPeak
        print("✅ Dashboard classes imported successfully")
        
        # Test video path
        video_path = os.path.join(current_dir, "C:\Users\Vallabhj\Downloads\CCTVPOC_integrated\Deep\Small video Mall shopping mall video shopping mall CCTV camera video no copyright full HD 4K video.mp4")
        if os.path.exists(video_path):
            print(f"✅ Video file found: {os.path.basename(video_path)}")
        else:
            print(f"⚠️ Video file not found: {video_path}")
            return False
        
        return True
        
    except Exception as e:
        print(f"❌ Dashboard class test failed: {e}")
        return False

def main():
    """Main test function"""
    print("🧪 Mall Analytics Dashboard - Component Test")
    print("=" * 50)
    
    # Test imports
    if not test_imports():
        print("\n❌ Import tests failed!")
        return False
    
    # Test dashboard class
    if not test_dashboard_class():
        print("\n❌ Dashboard class tests failed!")
        return False
    
    print("\n✅ All tests passed!")
    print("\n🚀 Dashboard is ready to run!")
    print("\nTo start the dashboard:")
    print("   python mall_analytics_dashboard.py")
    print("or")
    print("   run_mall_dashboard.bat")
    
    return True

if __name__ == "__main__":
    main()
