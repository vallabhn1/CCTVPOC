#!/usr/bin/env python3
"""
Real-time Traffic Heatmap Viewer        # Device setup - GPU ONLY, no CPU fallback
        if not torch.cuda.is_available():
            raise RuntimeError("❌ CUDA/GPU is required but not available! Please ensure:\n"
                             "   • NVIDIA GPU with CUDA support is installed\n"
                             "   • CUDA drivers are properly installed (You have CUDA 12.7)\n"
                             "   • PyTorch with CUDA 12.x: pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121\n"
                             "   • GPU is not being used by other processes")
        
        self.device = torch.device('cuda')
        
        # Clear GPU cache to ensure maximum available memory
        torch.cuda.empty_cache()
        
        print(f"🔧 Using device: {self.device}")
        print(f"🚀 GPU: {torch.cuda.get_device_name(0)}")
        print(f"💾 GPU Memory Total: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
        print(f"💾 GPU Memory Available: {(torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_allocated(0)) / 1024**3:.1f} GB")
        print(f"🔥 CUDA Compute Capability: {torch.cuda.get_device_properties(0).major}.{torch.cuda.get_device_properties(0).minor}")inal video side by side with live updating heatmap of person movement
"""

import cv2
import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from collections import defaultdict, deque
import time
import argparse
import os
from typing import List, Tuple, Dict
import threading
import queue

# Try to import YOLOv7 dependencies
try:
    import sys
    YOLOV7_PATH = r'c:\Users\deepv\OneDrive\Desktop\yolov7'
    if os.path.exists(YOLOV7_PATH):
        sys.path.append(YOLOV7_PATH)
        from models.experimental import attempt_load
        from utils.general import non_max_suppression, scale_coords
        from utils.torch_utils import select_device
        YOLOV7_AVAILABLE = True
        print("✅ YOLOv7 modules imported successfully")
    else:
        YOLOV7_AVAILABLE = False
        print("⚠️ YOLOv7 path not found, using fallback detection")
except ImportError as e:
    YOLOV7_AVAILABLE = False
    print(f"⚠️ YOLOv7 import failed: {e}")
    print("🔄 Using fallback detection method")

class RealTimeHeatmap:
    def __init__(self, video_path: str, weights_path: str = None, 
                 conf_threshold: float = 0.25, iou_threshold: float = 0.45,
                 heatmap_decay: float = 0.98, grid_size: int = 20):
        """
        Initialize real-time heatmap system
        
        Args:
            video_path: Path to input video
            weights_path: Path to YOLOv7 weights (optional)
            conf_threshold: Detection confidence threshold
            iou_threshold: IoU threshold for NMS
            heatmap_decay: Decay factor for heatmap (0.98 = 2% decay per frame)
            grid_size: Size of heatmap grid cells in pixels
        """
        self.video_path = video_path
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.heatmap_decay = heatmap_decay
        self.grid_size = grid_size
        
        # Device setup - GPU ONLY, no CPU fallback
        if not torch.cuda.is_available():
            raise RuntimeError("❌ CUDA/GPU is required but not available! Please ensure:\n"
                             "   • NVIDIA GPU with CUDA support is installed\n"
                             "   • CUDA drivers are properly installed\n"
                             "   • PyTorch with CUDA: pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118\n"
                             "   • GPU is not being used by other processes")
        
        self.device = torch.device('cuda')
        print(f"🔧 Using device: {self.device}")
        print(f"🚀 GPU: {torch.cuda.get_device_name(0)}")
        print(f"� GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
        
        # Initialize detection model
        self._init_detection_model(weights_path)
        
        # Heatmap data
        self.heatmap_data = None
        self.heatmap_history = deque(maxlen=30)  # Keep last 30 frames for smoothing
        
        # Video properties (will be set when video is loaded)
        self.frame_width = 0
        self.frame_height = 0
        self.fps = 30
        
        # Color maps for heatmap
        self.colormap = cm.get_cmap('hot')
        
        print("🎯 Real-time heatmap system initialized")

    def _init_detection_model(self, weights_path: str = None):
        """Initialize detection model (YOLOv7 or fallback)"""
        self.yolov7_available = YOLOV7_AVAILABLE
        
        if self.yolov7_available and weights_path and os.path.exists(weights_path):
            try:
                # Load YOLOv7 model
                self.model = attempt_load(weights_path, map_location=self.device)
                self.model.eval()
                self.img_size = 640
                self.names = self.model.module.names if hasattr(self.model, 'module') else self.model.names
                print(f"✅ YOLOv7 model loaded from {weights_path}")
            except Exception as e:
                print(f"❌ Error loading YOLOv7: {e}")
                self._init_fallback_model()
        else:
            self._init_fallback_model()

    def _init_fallback_model(self):
        """Initialize fallback detection model"""
        try:
            # Use Faster R-CNN from torchvision - GPU ONLY
            print("🔧 Loading Faster R-CNN model on GPU...")
            
            # Clear GPU cache before loading model
            torch.cuda.empty_cache()
            
            self.model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
            self.model.to(self.device)
            self.model.eval()
            
            # Force GPU mode and optimize for CUDA 12.7
            if next(self.model.parameters()).device.type != 'cuda':
                raise RuntimeError("❌ Model failed to load on GPU")
            
            # Enable mixed precision for CUDA 12.7 optimization
            if hasattr(torch.cuda, 'is_bf16_supported') and torch.cuda.is_bf16_supported():
                print("✅ BF16 mixed precision supported - optimizing for CUDA 12.7")
            
            self.img_size = 640
            self.names = {0: 'person'}  # COCO person class
            self.yolov7_available = False
            
            print("✅ Fallback model (Faster R-CNN) initialized successfully on GPU")
            print(f"🎯 Model device: {next(self.model.parameters()).device}")
            print(f"🔥 GPU Memory after model load: {torch.cuda.memory_allocated(0) / 1024**3:.2f} GB")
            
        except Exception as e:
            print(f"❌ Error initializing fallback model on GPU: {e}")
            raise RuntimeError(f"Failed to initialize detection model on GPU: {e}")

    def detect_people(self, frame: np.ndarray) -> List[Tuple[int, int, int, int, float]]:
        """
        Detect people in frame - GPU ONLY
        
        Returns:
            List of detections: [(x1, y1, x2, y2, confidence), ...]
        """
        if self.model is None:
            raise RuntimeError("❌ No detection model available on GPU")
        
        # Verify model is on GPU
        if next(self.model.parameters()).device.type != 'cuda':
            raise RuntimeError("❌ Detection model not on GPU")
        
        if self.yolov7_available:
            return self._detect_yolov7(frame)
        else:
            return self._detect_fallback(frame)

    def _detect_yolov7(self, frame: np.ndarray) -> List[Tuple[int, int, int, int, float]]:
        """YOLOv7 detection - GPU ONLY"""
        try:
            # Prepare image
            img = cv2.resize(frame, (self.img_size, self.img_size))
            img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB
            img = np.ascontiguousarray(img)
            img = torch.from_numpy(img).to(self.device)
            img = img.float() / 255.0
            if img.ndimension() == 3:
                img = img.unsqueeze(0)
            
            # Verify tensor is on GPU
            if img.device.type != 'cuda':
                raise RuntimeError("❌ YOLOv7 input tensor not on GPU")
            
            # Inference on GPU
            with torch.no_grad():
                pred = self.model(img)[0]
            
            # Verify predictions are on GPU
            if pred.device.type != 'cuda':
                raise RuntimeError("❌ YOLOv7 predictions not on GPU")
            
            # Apply NMS
            pred = non_max_suppression(pred, self.conf_threshold, self.iou_threshold, classes=[0], agnostic=False)
            
            detections = []
            if pred[0] is not None:
                # Scale coordinates back to original frame size
                pred[0][:, :4] = scale_coords(img.shape[2:], pred[0][:, :4], frame.shape).round()
                
                for det in pred[0]:
                    x1, y1, x2, y2, conf, cls = det.cpu().numpy()
                    if conf > self.conf_threshold:
                        detections.append((int(x1), int(y1), int(x2), int(y2), float(conf)))
            
            return detections
        except Exception as e:
            print(f"❌ YOLOv7 GPU detection error: {e}")
            raise RuntimeError(f"YOLOv7 GPU detection failed: {e}")

    def _detect_fallback(self, frame: np.ndarray) -> List[Tuple[int, int, int, int, float]]:
        """Fallback detection using Faster R-CNN - GPU ONLY optimized for CUDA 12.7"""
        try:
            # Prepare image
            transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.ToTensor(),
            ])
            
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img_tensor = transform(frame_rgb).unsqueeze(0).to(self.device, non_blocking=True)
            
            # Verify tensor is on GPU
            if img_tensor.device.type != 'cuda':
                raise RuntimeError("❌ Input tensor not on GPU")
            
            # Inference on GPU with optimizations for CUDA 12.7
            with torch.no_grad():
                # Use autocast for mixed precision if available
                if hasattr(torch.cuda.amp, 'autocast'):
                    with torch.cuda.amp.autocast():
                        predictions = self.model(img_tensor)
                else:
                    predictions = self.model(img_tensor)
            
            # Verify predictions are on GPU
            if predictions[0]['boxes'].device.type != 'cuda':
                print("⚠️ Warning: Predictions not on GPU, moving...")
                predictions[0]['boxes'] = predictions[0]['boxes'].cuda()
                predictions[0]['scores'] = predictions[0]['scores'].cuda()
                predictions[0]['labels'] = predictions[0]['labels'].cuda()
            
            detections = []
            if len(predictions) > 0 and len(predictions[0]['boxes']) > 0:
                boxes = predictions[0]['boxes'].cpu().numpy()
                scores = predictions[0]['scores'].cpu().numpy()
                labels = predictions[0]['labels'].cpu().numpy()
                
                # Filter for person class (label 1 in COCO) and confidence
                person_mask = (labels == 1) & (scores > self.conf_threshold)
                
                for box, score in zip(boxes[person_mask], scores[person_mask]):
                    x1, y1, x2, y2 = box.astype(int)
                    detections.append((x1, y1, x2, y2, float(score)))
            
            return detections
        except Exception as e:
            print(f"❌ GPU detection error: {e}")
            raise RuntimeError(f"GPU detection failed: {e}")

    def update_heatmap(self, detections: List[Tuple[int, int, int, int, float]]):
        """Update heatmap based on detections"""
        if self.heatmap_data is None:
            # Initialize heatmap grid
            grid_h = (self.frame_height + self.grid_size - 1) // self.grid_size
            grid_w = (self.frame_width + self.grid_size - 1) // self.grid_size
            self.heatmap_data = np.zeros((grid_h, grid_w), dtype=np.float32)
        
        # Apply decay to existing heatmap
        self.heatmap_data *= self.heatmap_decay
        
        # Add new detections
        for x1, y1, x2, y2, conf in detections:
            # Calculate center point
            center_x = (x1 + x2) // 2
            center_y = (y1 + y2) // 2
            
            # Convert to grid coordinates
            grid_x = min(center_x // self.grid_size, self.heatmap_data.shape[1] - 1)
            grid_y = min(center_y // self.grid_size, self.heatmap_data.shape[0] - 1)
            
            # Add detection with confidence weighting
            self.heatmap_data[grid_y, grid_x] += conf
            
            # Also add to neighboring cells for smoother visualization
            for dy in [-1, 0, 1]:
                for dx in [-1, 0, 1]:
                    ny, nx = grid_y + dy, grid_x + dx
                    if 0 <= ny < self.heatmap_data.shape[0] and 0 <= nx < self.heatmap_data.shape[1]:
                        if dy != 0 or dx != 0:  # Don't double-add to center
                            self.heatmap_data[ny, nx] += conf * 0.3

    def generate_heatmap_overlay(self) -> np.ndarray:
        """Generate heatmap overlay image"""
        if self.heatmap_data is None:
            return np.zeros((self.frame_height, self.frame_width, 3), dtype=np.uint8)
        
        # Normalize heatmap data
        max_val = np.max(self.heatmap_data)
        if max_val > 0:
            normalized_heatmap = self.heatmap_data / max_val
        else:
            normalized_heatmap = self.heatmap_data
        
        # Apply colormap
        colored_heatmap = self.colormap(normalized_heatmap)
        colored_heatmap = (colored_heatmap[:, :, :3] * 255).astype(np.uint8)
        
        # Resize to match frame dimensions
        heatmap_resized = cv2.resize(colored_heatmap, (self.frame_width, self.frame_height))
        
        return heatmap_resized

    def draw_detections(self, frame: np.ndarray, detections: List[Tuple[int, int, int, int, float]]) -> np.ndarray:
        """Draw detection boxes on frame"""
        frame_copy = frame.copy()
        
        for x1, y1, x2, y2, conf in detections:
            # Draw bounding box
            cv2.rectangle(frame_copy, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Draw confidence
            label = f'Person: {conf:.2f}'
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
            cv2.rectangle(frame_copy, (x1, y1 - label_size[1] - 10), 
                         (x1 + label_size[0], y1), (0, 255, 0), -1)
            cv2.putText(frame_copy, label, (x1, y1 - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
        
        return frame_copy

    def process_video(self, display: bool = True, save_output: str = None):
        """Process video with real-time heatmap display"""
        if not os.path.exists(self.video_path):
            print(f"❌ Video file not found: {self.video_path}")
            return
        
        cap = cv2.VideoCapture(self.video_path)
        
        # Get video properties
        self.frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        print(f"🎥 Video: {self.video_path}")
        print(f"📏 Resolution: {self.frame_width}x{self.frame_height}")
        print(f"⏱️ FPS: {self.fps}")
        print(f"📊 Total frames: {total_frames}")
        
        # Setup video writer if saving output
        out = None
        if save_output:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            # Double width for side-by-side display
            out = cv2.VideoWriter(save_output, fourcc, self.fps, 
                                (self.frame_width * 2, self.frame_height))
        
        frame_count = 0
        start_time = time.time()
        
        print("🚀 Starting real-time processing...")
        print("Press 'q' to quit, 'r' to reset heatmap")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                print("📹 End of video reached")
                break
            
            # Detect people
            detections = self.detect_people(frame)
            
            # Update heatmap
            self.update_heatmap(detections)
            
            # Draw detections on original frame
            frame_with_detections = self.draw_detections(frame, detections)
            
            # Generate heatmap overlay
            heatmap_overlay = self.generate_heatmap_overlay()
            
            # Add titles and info
            cv2.putText(frame_with_detections, "Original Video + Detections", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(frame_with_detections, f"Detections: {len(detections)}", 
                       (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            cv2.putText(frame_with_detections, f"Frame: {frame_count}/{total_frames}", 
                       (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            
            cv2.putText(heatmap_overlay, "Real-time Traffic Heatmap", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(heatmap_overlay, f"Heat Max: {np.max(self.heatmap_data):.2f}", 
                       (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.putText(heatmap_overlay, "Hot: White, Cold: Black", 
                       (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 2)
            
            # Combine side by side
            combined_frame = np.hstack([frame_with_detections, heatmap_overlay])
            
            if display:
                cv2.imshow('Real-time Traffic Analysis', combined_frame)
                
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('r'):
                    # Reset heatmap
                    self.heatmap_data = None
                    print("🔄 Heatmap reset")
            
            if out:
                out.write(combined_frame)
            
            frame_count += 1
            
            # Print progress every 100 frames
            if frame_count % 100 == 0:
                elapsed = time.time() - start_time
                fps_processing = frame_count / elapsed
                progress = (frame_count / total_frames) * 100
                
                # GPU memory monitoring
                gpu_memory_used = torch.cuda.memory_allocated(0) / 1024**3
                gpu_memory_cached = torch.cuda.memory_reserved(0) / 1024**3
                
                print(f"Progress: {progress:.1f}% - Processing FPS: {fps_processing:.1f}")
                print(f"GPU Memory: {gpu_memory_used:.1f}GB used, {gpu_memory_cached:.1f}GB cached")
        
        # Cleanup
        cap.release()
        if out:
            out.release()
        cv2.destroyAllWindows()
        
        elapsed_total = time.time() - start_time
        print(f"\n✅ Processing completed!")
        print(f"📊 Total frames processed: {frame_count}")
        print(f"⏱️ Total time: {elapsed_total:.2f} seconds")
        print(f"🏃 Average processing FPS: {frame_count / elapsed_total:.2f}")
        
        if save_output:
            print(f"💾 Output saved to: {save_output}")

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='Real-time Traffic Heatmap Viewer')
    parser.add_argument('--video', type=str, 
                       default=r'c:\Users\deepv\OneDrive\Desktop\Hive Dynamics\Deep\Small video Mall shopping mall video shopping mall CCTV camera video no copyright full HD 4K video.mp4',
                       help='Path to input video file')
    parser.add_argument('--weights', type=str, default=None,
                       help='Path to YOLOv7 weights file')
    parser.add_argument('--output', type=str, default=None,
                       help='Path to save output video (optional)')
    parser.add_argument('--conf-thres', type=float, default=0.25,
                       help='Detection confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45,
                       help='IoU threshold for NMS')
    parser.add_argument('--grid-size', type=int, default=30,
                       help='Heatmap grid size in pixels')
    parser.add_argument('--decay', type=float, default=0.99,
                       help='Heatmap decay factor (0.95-0.999)')
    parser.add_argument('--no-display', action='store_true',
                       help='Don\'t display video during processing')
    
    args = parser.parse_args()
    
    print("🌡️ Real-time Traffic Heatmap System")
    print("="*50)
    
    # Check GPU availability - MANDATORY
    if not torch.cuda.is_available():
        print("❌ CUDA/GPU is NOT available!")
        print("🚫 This application requires GPU/CUDA to run.")
        print("📋 To fix this issue:")
        print("   1. Install NVIDIA GPU drivers (You have driver 566.07 - Good!)")
        print("   2. Install CUDA toolkit 12.7")
        print("   3. Install PyTorch with CUDA 12.x:")
        print("      pip uninstall torch torchvision torchaudio")
        print("      pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121")
        print("   4. Restart your system")
        print("\n💡 To check GPU status, run: nvidia-smi")
        return
    
    print(f"✅ GPU Available: {torch.cuda.get_device_name(0)}")
    print(f"🔥 CUDA Version: {torch.version.cuda}")
    print(f"💾 GPU Memory Total: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    print(f"🚀 GPU Driver Version: 566.07 (Compatible with CUDA 12.7)")
    print(f"⚡ GPU Compute Capability: {torch.cuda.get_device_properties(0).major}.{torch.cuda.get_device_properties(0).minor}")
    
    # Clear GPU cache for optimal performance
    torch.cuda.empty_cache()
    print(f"🧹 GPU cache cleared - Ready for processing")
    
    # Initialize heatmap system
    heatmap_system = RealTimeHeatmap(
        video_path=args.video,
        weights_path=args.weights,
        conf_threshold=args.conf_thres,
        iou_threshold=args.iou_thres,
        heatmap_decay=args.decay,
        grid_size=args.grid_size
    )
    
    # Process video
    heatmap_system.process_video(
        display=not args.no_display,
        save_output=args.output
    )
    
    print("\n🎉 Real-time heatmap analysis complete!")

if __name__ == "__main__":
    main()
