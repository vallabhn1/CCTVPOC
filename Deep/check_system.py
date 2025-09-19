"""
System Requirements Checker for Mall CCTV Analytics
This script verifies that all dependencies and requirements are met.
"""

import sys
import os
import subprocess
from pathlib import Path

def check_python_version():
    """Check if Python version is compatible"""
    version = sys.version_info
    print(f"Python Version: {version.major}.{version.minor}.{version.micro}")
    
    if version.major == 3 and version.minor >= 8:
        print("✅ Python version is compatible")
        return True
    else:
        print("❌ Python 3.8 or higher is required")
        return False

def check_gpu_availability():
    """Check if GPU is available for processing"""
    try:
        import torch
        cuda_available = torch.cuda.is_available()
        print(f"CUDA Available: {cuda_available}")
        
        if cuda_available:
            gpu_count = torch.cuda.device_count()
            print(f"GPU Count: {gpu_count}")
            for i in range(gpu_count):
                gpu_name = torch.cuda.get_device_name(i)
                print(f"GPU {i}: {gpu_name}")
            print("✅ GPU acceleration is available")
        else:
            print("⚠️ GPU not available - will use CPU (slower processing)")
        
        return cuda_available
    except ImportError:
        print("❌ PyTorch not installed")
        return False

def check_required_packages():
    """Check if all required packages are installed"""
    required_packages = [
        'torch', 'torchvision', 'cv2', 'numpy', 'matplotlib', 
        'seaborn', 'pandas', 'PIL', 'scipy', 'tqdm', 'yaml', 'requests'
    ]
    
    missing_packages = []
    installed_packages = []
    
    for package in required_packages:
        try:
            if package == 'cv2':
                import cv2
                installed_packages.append(f"opencv-python ({cv2.__version__})")
            elif package == 'PIL':
                import PIL
                installed_packages.append(f"Pillow ({PIL.__version__})")
            elif package == 'yaml':
                import yaml
                installed_packages.append(f"PyYAML")
            else:
                module = __import__(package)
                version = getattr(module, '__version__', 'unknown')
                installed_packages.append(f"{package} ({version})")
        except ImportError:
            missing_packages.append(package)
    
    print("\nInstalled Packages:")
    for pkg in installed_packages:
        print(f"✅ {pkg}")
    
    if missing_packages:
        print("\nMissing Packages:")
        for pkg in missing_packages:
            print(f"❌ {pkg}")
        return False
    else:
        print("✅ All required packages are installed")
        return True

def check_yolov7_setup():
    """Check if YOLOv7 is properly set up"""
    yolov7_path = r"C:\Users\Vallabhj\Downloads\CCTVPOC_integrated\yolov7\yolov7.pt"
    
    print(f"\nYOLOv7 Path: {yolov7_path}")
    
    if not os.path.exists(yolov7_path):
        print("❌ YOLOv7 directory not found")
        print("Please clone YOLOv7 repository to the specified path")
        return False
    
    # Check for essential YOLOv7 files
    essential_files = [
        'models/experimental.py',
        'utils/datasets.py',
        'utils/general.py',
        'utils/torch_utils.py',
        'utils/plots.py'
    ]
    
    missing_files = []
    for file in essential_files:
        file_path = os.path.join(yolov7_path, file)
        if os.path.exists(file_path):
            print(f"✅ {file}")
        else:
            print(f"❌ {file}")
            missing_files.append(file)
    
    # Check for weights file
    weights_file = os.path.join(yolov7_path, 'yolov7.pt')
    if os.path.exists(weights_file):
        print(f"✅ yolov7.pt weights file found")
    else:
        print(f"⚠️ yolov7.pt weights file not found")
        print("You may need to download it from: https://github.com/WongKinYiu/yolov7/releases/download/v0.1/yolov7.pt")
    
    if missing_files:
        print("❌ YOLOv7 setup incomplete")
        return False
    else:
        print("✅ YOLOv7 setup appears complete")
        return True

def check_video_file():
    """Check if the video file exists"""
    video_path = r"C:\Users\Vallabhj\Downloads\CCTVPOC_integrated\Deep\Small video Mall shopping mall video shopping mall CCTV camera video no copyright full HD 4K video.mp4"
    
    print(f"\nVideo File: {video_path}")
    
    if os.path.exists(video_path):
        # Get video file size
        size_mb = os.path.getsize(video_path) / (1024 * 1024)
        print(f"✅ Video file found ({size_mb:.1f} MB)")
        
        # Try to get video properties
        try:
            import cv2
            cap = cv2.VideoCapture(video_path)
            if cap.isOpened():
                width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                fps = cap.get(cv2.CAP_PROP_FPS)
                frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                duration = frame_count / fps if fps > 0 else 0
                
                print(f"   Resolution: {width}x{height}")
                print(f"   FPS: {fps:.1f}")
                print(f"   Duration: {duration:.1f} seconds")
                print(f"   Total Frames: {frame_count}")
                
                cap.release()
                return True
            else:
                print("❌ Cannot open video file")
                return False
        except Exception as e:
            print(f"⚠️ Error reading video properties: {e}")
            return True  # File exists but properties couldn't be read
    else:
        print("❌ Video file not found")
        print("Please ensure the video file is in the correct location")
        return False

def check_disk_space():
    """Check available disk space"""
    try:
        import shutil
        free_space = shutil.disk_usage('.').free / (1024**3)  # Convert to GB
        print(f"\nAvailable Disk Space: {free_space:.1f} GB")
        
        if free_space > 5:  # At least 5GB recommended
            print("✅ Sufficient disk space available")
            return True
        else:
            print("⚠️ Low disk space - ensure at least 5GB free for processing")
            return False
    except Exception as e:
        print(f"⚠️ Could not check disk space: {e}")
        return True

def install_missing_packages():
    """Install missing packages automatically"""
    print("\nAttempting to install missing packages...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("✅ Packages installed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install packages: {e}")
        return False

def main():
    """Main function to run all checks"""
    print("🔍 Mall CCTV Analytics - System Requirements Check")
    print("=" * 60)
    
    checks_passed = 0
    total_checks = 6
    
    # Run all checks
    if check_python_version():
        checks_passed += 1
    
    if check_required_packages():
        checks_passed += 1
    else:
        print("\nWould you like to install missing packages automatically? (y/n): ", end="")
        response = input().lower().strip()
        if response in ['y', 'yes']:
            if install_missing_packages():
                checks_passed += 1
    
    if check_gpu_availability():
        checks_passed += 1
    
    if check_yolov7_setup():
        checks_passed += 1
    
    if check_video_file():
        checks_passed += 1
    
    if check_disk_space():
        checks_passed += 1
    
    # Summary
    print("\n" + "=" * 60)
    print(f"System Check Summary: {checks_passed}/{total_checks} checks passed")
    
    if checks_passed == total_checks:
        print("🎉 System is ready for mall analytics!")
        print("You can now run: python cctv.py")
    elif checks_passed >= total_checks - 1:
        print("⚠️ System is mostly ready with minor issues")
        print("Analytics should work but may have reduced functionality")
    else:
        print("❌ System has significant issues that need to be resolved")
        print("Please fix the issues above before running analytics")
    
    print("\nNext steps:")
    print("1. Fix any issues shown above")
    print("2. Run: python cctv.py")
    print("3. Check output files: heatmap.png, analytics_report.json")

if __name__ == "__main__":
    main()
