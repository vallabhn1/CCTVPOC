#!/usr/bin/env python3
"""
Installation verification script for Mall CCTV Analytics System
Run this script after installation to verify all components are working correctly.
"""

import sys
import importlib.util

def check_python_version():
    """Check if Python version is compatible"""
    print("🐍 Checking Python version...")
    if sys.version_info < (3, 8):
        print("❌ Python 3.8+ required. Current version:", sys.version)
        return False
    print(f"✅ Python {sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}")
    return True

def check_package(package_name, display_name=None):
    """Check if a package is installed"""
    display_name = display_name or package_name
    try:
        spec = importlib.util.find_spec(package_name)
        if spec is not None:
            print(f"✅ {display_name}")
            return True
        else:
            print(f"❌ {display_name} - Not installed")
            return False
    except Exception as e:
        print(f"❌ {display_name} - Error: {e}")
        return False

def check_gpu_support():
    """Check GPU/CUDA support"""
    print("\n🚀 Checking GPU support...")
    try:
        import torch
        if torch.cuda.is_available():
            print(f"✅ CUDA available - {torch.cuda.get_device_name(0)}")
            print(f"   GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
            return True
        else:
            print("⚠️  CUDA not available - Will use CPU (slower performance)")
            return False
    except ImportError:
        print("❌ PyTorch not installed")
        return False

def check_models():
    """Check if core modules can be imported"""
    print("\n📦 Checking core modules...")
    
    modules_to_check = [
        ('mall_analytics', 'Mall Analytics Engine'),
        ('models', 'Data Models'),
        ('cv2', 'OpenCV'),
        ('torch', 'PyTorch'),
        ('numpy', 'NumPy'),
        ('ultralytics', 'YOLOv8 (Ultralytics)')
    ]
    
    all_good = True
    for module, display in modules_to_check:
        if not check_package(module, display):
            all_good = False
    
    return all_good

def test_analytics_initialization():
    """Test if MallAnalytics can be initialized"""
    print("\n🧠 Testing analytics initialization...")
    try:
        from mall_analytics import MallAnalytics
        analytics = MallAnalytics()
        print("✅ MallAnalytics initialized successfully")
        print(f"   Device: {analytics.device}")
        print(f"   Gender classifier: {'✅ Loaded' if analytics.gender_classifier is not None else '❌ Failed'}")
        return True
    except Exception as e:
        print(f"❌ Failed to initialize MallAnalytics: {e}")
        return False

def main():
    """Main verification function"""
    print("="*60)
    print("   MALL CCTV ANALYTICS SYSTEM - INSTALLATION CHECK")
    print("="*60)
    
    checks = [
        check_python_version(),
        check_models(),
        check_gpu_support(),
        test_analytics_initialization()
    ]
    
    print("\n" + "="*60)
    if all(checks):
        print("🎉 ALL CHECKS PASSED! System is ready for use.")
        print("\nNext steps:")
        print("1. Run: python mall_surveillance.py")
        print("2. Models will download automatically on first run")
        print("3. Press ESC to exit, SPACE to pause/resume")
    else:
        print("⚠️  Some checks failed. Please install missing packages:")
        print("   pip install -r requirements.txt")
    print("="*60)

if __name__ == "__main__":
    main()
