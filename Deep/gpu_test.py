#!/usr/bin/env python3
"""
GPU/CUDA Test Script
Tests GPU availability and performance for the real-time heatmap system
"""

import torch
import time
import sys

def test_gpu_availability():
    """Test basic GPU availability"""
    print("🔍 GPU/CUDA Availability Test")
    print("="*40)
    
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        print(f"CUDA version: {torch.version.cuda}")
        print(f"GPU count: {torch.cuda.device_count()}")
        
        for i in range(torch.cuda.device_count()):
            print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
            props = torch.cuda.get_device_properties(i)
            print(f"  Memory: {props.total_memory / 1024**3:.1f} GB")
            print(f"  Compute capability: {props.major}.{props.minor}")
        
        return True
    else:
        print("❌ CUDA is not available!")
        print("\n📋 To install CUDA support:")
        print("1. Install NVIDIA GPU drivers")
        print("2. Install CUDA toolkit from NVIDIA")
        print("3. Install PyTorch with CUDA:")
        print("   pip uninstall torch torchvision")
        print("   pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118")
        print("4. Restart your system")
        print("\n💡 Check GPU status with: nvidia-smi")
        return False

def test_gpu_performance():
    """Test GPU performance"""
    if not torch.cuda.is_available():
        print("❌ Cannot test GPU performance - CUDA not available")
        return
    
    print("\n🚀 GPU Performance Test")
    print("="*40)
    
    device = torch.device('cuda')
    
    # Test tensor operations
    print("Testing tensor operations...")
    size = 1000
    
    # CPU test
    start_time = time.time()
    a_cpu = torch.randn(size, size)
    b_cpu = torch.randn(size, size)
    c_cpu = torch.matmul(a_cpu, b_cpu)
    cpu_time = time.time() - start_time
    
    # GPU test
    start_time = time.time()
    a_gpu = torch.randn(size, size, device=device)
    b_gpu = torch.randn(size, size, device=device)
    torch.cuda.synchronize()  # Wait for GPU
    
    gpu_start = time.time()
    c_gpu = torch.matmul(a_gpu, b_gpu)
    torch.cuda.synchronize()  # Wait for GPU
    gpu_time = time.time() - gpu_start
    
    print(f"CPU time: {cpu_time:.4f} seconds")
    print(f"GPU time: {gpu_time:.4f} seconds")
    print(f"Speedup: {cpu_time / gpu_time:.2f}x")
    
    # Memory test
    print(f"\nGPU Memory Usage:")
    print(f"Allocated: {torch.cuda.memory_allocated(0) / 1024**2:.1f} MB")
    print(f"Reserved: {torch.cuda.memory_reserved(0) / 1024**2:.1f} MB")

def test_model_loading():
    """Test model loading on GPU"""
    if not torch.cuda.is_available():
        print("❌ Cannot test model loading - CUDA not available")
        return
    
    print("\n🤖 Model Loading Test")
    print("="*40)
    
    try:
        import torchvision
        device = torch.device('cuda')
        
        print("Loading Faster R-CNN model...")
        start_time = time.time()
        model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
        model.to(device)
        model.eval()
        load_time = time.time() - start_time
        
        print(f"✅ Model loaded successfully in {load_time:.2f} seconds")
        print(f"Model device: {next(model.parameters()).device}")
        
        # Test inference
        print("Testing inference...")
        dummy_input = torch.randn(1, 3, 640, 640, device=device)
        
        start_time = time.time()
        with torch.no_grad():
            output = model(dummy_input)
        torch.cuda.synchronize()
        inference_time = time.time() - start_time
        
        print(f"✅ Inference completed in {inference_time:.4f} seconds")
        print(f"Output device: {output[0]['boxes'].device}")
        
    except Exception as e:
        print(f"❌ Model loading failed: {e}")

def main():
    """Main test function"""
    print("🧪 Real-time Heatmap GPU Test Suite")
    print("="*50)
    
    # Test 1: GPU Availability
    gpu_available = test_gpu_availability()
    
    if gpu_available:
        # Test 2: GPU Performance
        test_gpu_performance()
        
        # Test 3: Model Loading
        test_model_loading()
        
        print("\n🎉 All GPU tests completed!")
        print("✅ Your system is ready for GPU-accelerated real-time heatmap processing")
    else:
        print("\n❌ GPU not available - real-time heatmap requires CUDA")
        print("📝 Please follow the installation instructions above")
        sys.exit(1)

if __name__ == "__main__":
    main()
