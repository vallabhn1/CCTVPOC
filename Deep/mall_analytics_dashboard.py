#!/usr/bin/env python3
"""
Mall Analytics Real-time Dashboard
Comprehensive CCTV analytics with real-time dashboard displaying:
1. Total unique visitors
2. Real-time heatmap
3. Dwell time of each unique person (ID)
4. Peak traffic timestamps and areas
5. Live statistics and visualization

GPU-Only Processing for CUDA 12.7 compatibility
"""

import cv2
import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import seaborn as sns
from collections import defaultdict, deque, Counter
import time
import argparse
import os
import json
from datetime import datetime, timedelta
from typing import List, Tuple, Dict, Set
import threading
import queue
from dataclasses import dataclass
import pandas as pd

# Try to import YOLOv7 dependencies
try:
    import sys
    YOLOV7_PATH = r"C:\Users\Vallabhj\Downloads\CCTVPOC_integrated\yolov7.pt"
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

@dataclass
class PersonData:
    """Data class for tracking individual person statistics"""
    track_id: int
    first_seen: float
    last_seen: float
    total_frames: int
    areas_visited: Set[str]
    area_dwell_times: Dict[str, float]
    current_area: str
    last_position: Tuple[int, int]
    bbox_history: List[Tuple[int, int, int, int]]
    is_active: bool = True
    
    @property
    def total_dwell_time(self) -> float:
        """Calculate total time person spent in the mall"""
        return self.last_seen - self.first_seen
    
    @property
    def average_dwell_per_area(self) -> float:
        """Calculate average dwell time per area"""
        if len(self.area_dwell_times) == 0:
            return 0.0
        return sum(self.area_dwell_times.values()) / len(self.area_dwell_times)

class TrafficPeak:
    """Class to track peak traffic information"""
    def __init__(self):
        self.timestamp: float = 0
        self.area: str = ""
        self.count: int = 0
        self.frame_number: int = 0

class MallAnalyticsDashboard:
    def __init__(self, video_path: str, weights_path: str = None, 
                 conf_threshold: float = 0.25, iou_threshold: float = 0.45,
                 heatmap_decay: float = 0.98, grid_size: int = 30):
        """
        Initialize Mall Analytics Dashboard
        
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
        print(f"🔥 CUDA Compute Capability: {torch.cuda.get_device_properties(0).major}.{torch.cuda.get_device_properties(0).minor}")
        
        # Initialize detection model
        self._init_detection_model(weights_path)
        
        # Video properties (will be set when video is loaded)
        self.frame_width = 0
        self.frame_height = 0
        self.fps = 30
        self.total_frames = 0
        
        # Mall areas definition (customizable based on mall layout)
        self.mall_areas = {
            'entrance': {'x': 0, 'y': 0, 'w': 200, 'h': 480, 'color': (0, 255, 0)},
            'clothing_store': {'x': 200, 'y': 0, 'w': 200, 'h': 240, 'color': (255, 0, 0)},
            'electronics': {'x': 400, 'y': 0, 'w': 200, 'h': 240, 'color': (0, 0, 255)},
            'food_court': {'x': 200, 'y': 240, 'w': 400, 'h': 240, 'color': (255, 255, 0)},
            'exit': {'x': 600, 'y': 0, 'w': 200, 'h': 480, 'color': (255, 0, 255)},
            'central_corridor': {'x': 200, 'y': 120, 'w': 400, 'h': 240, 'color': (0, 255, 255)}
        }
        
        # Tracking data
        self.people_data: Dict[int, PersonData] = {}
        self.next_track_id = 0
        self.frame_count = 0
        self.start_time = time.time()
        
        # Heatmap data
        self.heatmap_data = None
        self.area_traffic_counts = defaultdict(lambda: defaultdict(int))  # area -> frame_count -> count
        
        # Peak traffic tracking
        self.current_peak = TrafficPeak()
        self.all_time_peak = TrafficPeak()
        self.hourly_peaks = []  # List of peaks for each time period
        
        # Real-time statistics
        self.total_unique_visitors = 0
        self.current_active_visitors = 0
        self.area_popularity = defaultdict(int)
        
        # Color maps for visualization
        self.colormap = cm.get_cmap('hot')
        self.track_colors = {}  # Store consistent colors for each track ID
        
        print("🎯 Mall Analytics Dashboard initialized successfully")

    def _init_detection_model(self, weights_path: str = None):
        """Initialize detection model (YOLOv7 or fallback) - GPU ONLY"""
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
        """Initialize fallback detection model - GPU ONLY"""
        try:
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
        """Detect people in frame - GPU ONLY"""
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
        """Fallback detection using Faster R-CNN - GPU ONLY"""
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

    def get_area_from_position(self, x: float, y: float) -> str:
        """Determine which mall area a position belongs to"""
        for area_name, area_coords in self.mall_areas.items():
            if (area_coords['x'] <= x <= area_coords['x'] + area_coords['w'] and
                area_coords['y'] <= y <= area_coords['y'] + area_coords['h']):
                return area_name
        return 'unknown'

    def simple_tracker(self, detections: List[Tuple[int, int, int, int, float]], 
                      frame_time: float) -> List[Tuple[int, int, int, int, float, int]]:
        """Simple tracking algorithm with ID assignment"""
        if not detections:
            # Mark all current people as inactive if no detections
            for person in self.people_data.values():
                if person.is_active:
                    person.is_active = False
                    person.last_seen = frame_time
            return []
        
        tracked_detections = []
        used_track_ids = set()
        
        for x1, y1, x2, y2, conf in detections:
            center_x = (x1 + x2) / 2
            center_y = (y1 + y2) / 2
            current_area = self.get_area_from_position(center_x, center_y)
            
            # Find best matching existing person
            best_match_id = None
            best_distance = float('inf')
            max_distance_threshold = 100  # pixels
            
            for track_id, person in self.people_data.items():
                if not person.is_active or track_id in used_track_ids:
                    continue
                
                # Calculate distance from last known position
                last_x, last_y = person.last_position
                distance = np.sqrt((center_x - last_x)**2 + (center_y - last_y)**2)
                
                if distance < best_distance and distance < max_distance_threshold:
                    best_distance = distance
                    best_match_id = track_id
            
            if best_match_id is not None:
                # Update existing person
                person = self.people_data[best_match_id]
                person.last_seen = frame_time
                person.last_position = (center_x, center_y)
                person.bbox_history.append((x1, y1, x2, y2))
                person.total_frames += 1
                
                # Update area information
                if current_area != 'unknown':
                    person.areas_visited.add(current_area)
                    if person.current_area != current_area:
                        # Person moved to new area
                        if person.current_area and person.current_area != 'unknown':
                            # Calculate time spent in previous area
                            time_in_area = frame_time - person.last_seen + (1.0 / self.fps)
                            person.area_dwell_times[person.current_area] += time_in_area
                        person.current_area = current_area
                
                used_track_ids.add(best_match_id)
                tracked_detections.append((x1, y1, x2, y2, conf, best_match_id))
                
            else:
                # Create new person
                new_id = self.next_track_id
                self.next_track_id += 1
                
                new_person = PersonData(
                    track_id=new_id,
                    first_seen=frame_time,
                    last_seen=frame_time,
                    total_frames=1,
                    areas_visited=set([current_area]) if current_area != 'unknown' else set(),
                    area_dwell_times=defaultdict(float),
                    current_area=current_area,
                    last_position=(center_x, center_y),
                    bbox_history=[(x1, y1, x2, y2)],
                    is_active=True
                )
                
                self.people_data[new_id] = new_person
                self.total_unique_visitors += 1
                
                # Assign consistent color for this track ID
                self.track_colors[new_id] = (
                    int(255 * ((new_id * 67) % 256) / 256),
                    int(255 * ((new_id * 113) % 256) / 256),
                    int(255 * ((new_id * 197) % 256) / 256)
                )
                
                tracked_detections.append((x1, y1, x2, y2, conf, new_id))
        
        # Update current active visitors count
        self.current_active_visitors = len([p for p in self.people_data.values() if p.is_active])
        
        return tracked_detections

    def update_heatmap(self, tracked_detections: List[Tuple[int, int, int, int, float, int]]):
        """Update heatmap based on tracked detections"""
        if self.heatmap_data is None:
            # Initialize heatmap grid
            grid_h = (self.frame_height + self.grid_size - 1) // self.grid_size
            grid_w = (self.frame_width + self.grid_size - 1) // self.grid_size
            self.heatmap_data = np.zeros((grid_h, grid_w), dtype=np.float32)
        
        # Apply decay to existing heatmap
        self.heatmap_data *= self.heatmap_decay
        
        # Add new detections
        for x1, y1, x2, y2, conf, track_id in tracked_detections:
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

    def update_traffic_peaks(self, tracked_detections: List[Tuple[int, int, int, int, float, int]], 
                           frame_time: float):
        """Update peak traffic information"""
        # Count people in each area
        area_counts = defaultdict(int)
        
        for x1, y1, x2, y2, conf, track_id in tracked_detections:
            center_x = (x1 + x2) / 2
            center_y = (y1 + y2) / 2
            area = self.get_area_from_position(center_x, center_y)
            if area != 'unknown':
                area_counts[area] += 1
        
        # Update area traffic counts for this frame
        for area, count in area_counts.items():
            self.area_traffic_counts[area][self.frame_count] = count
            self.area_popularity[area] += count
        
        # Find current peak area
        if area_counts:
            peak_area = max(area_counts.items(), key=lambda x: x[1])
            peak_area_name, peak_count = peak_area
            
            # Update current peak
            self.current_peak.timestamp = frame_time
            self.current_peak.area = peak_area_name
            self.current_peak.count = peak_count
            self.current_peak.frame_number = self.frame_count
            
            # Update all-time peak
            if peak_count > self.all_time_peak.count:
                self.all_time_peak.timestamp = frame_time
                self.all_time_peak.area = peak_area_name
                self.all_time_peak.count = peak_count
                self.all_time_peak.frame_number = self.frame_count

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

    def create_dashboard_frame(self, original_frame: np.ndarray, 
                              tracked_detections: List[Tuple[int, int, int, int, float, int]]) -> np.ndarray:
        """Create comprehensive dashboard display"""
        # Create dashboard layout
        dashboard_width = self.frame_width * 2
        dashboard_height = self.frame_height + 400  # Extra space for statistics
        dashboard = np.zeros((dashboard_height, dashboard_width, 3), dtype=np.uint8)
        
        # 1. Original video with detections (top-left)
        frame_with_detections = self.draw_detections_and_areas(original_frame, tracked_detections)
        dashboard[0:self.frame_height, 0:self.frame_width] = frame_with_detections
        
        # 2. Heatmap (top-right)
        heatmap_overlay = self.generate_heatmap_overlay()
        dashboard[0:self.frame_height, self.frame_width:dashboard_width] = heatmap_overlay
        
        # 3. Statistics panel (bottom)
        stats_panel = self.create_statistics_panel()
        dashboard[self.frame_height:dashboard_height, 0:dashboard_width] = stats_panel
        
        return dashboard

    def draw_detections_and_areas(self, frame: np.ndarray, 
                                 tracked_detections: List[Tuple[int, int, int, int, float, int]]) -> np.ndarray:
        """Draw detections, areas, and labels on frame"""
        frame_copy = frame.copy()
        
        # Draw mall areas
        for area_name, coords in self.mall_areas.items():
            cv2.rectangle(frame_copy, 
                         (coords['x'], coords['y']), 
                         (coords['x'] + coords['w'], coords['y'] + coords['h']),
                         coords['color'], 2)
            cv2.putText(frame_copy, area_name.replace('_', ' ').title(), 
                       (coords['x'] + 5, coords['y'] + 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, coords['color'], 2)
        
        # Draw tracked people
        for x1, y1, x2, y2, conf, track_id in tracked_detections:
            color = self.track_colors.get(track_id, (0, 255, 0))
            
            # Draw bounding box
            cv2.rectangle(frame_copy, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
            
            # Get person data for labels
            person = self.people_data.get(track_id)
            if person:
                current_time = time.time()
                elapsed_time = current_time - self.start_time
                dwell_time = person.total_dwell_time
                
                # Draw labels
                label1 = f'ID:{track_id} T:{dwell_time:.1f}s'
                label2 = f'Area:{person.current_area}'
                
                # Label background
                label_size = cv2.getTextSize(label1, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)[0]
                cv2.rectangle(frame_copy, (int(x1), int(y1) - 35), 
                             (int(x1) + max(label_size[0], 120), int(y1)), color, -1)
                
                # Labels
                cv2.putText(frame_copy, label1, (int(x1) + 2, int(y1) - 20), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)
                cv2.putText(frame_copy, label2, (int(x1) + 2, int(y1) - 5), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)
        
        # Add title and current stats
        cv2.putText(frame_copy, "Live CCTV Analytics + Detection", 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(frame_copy, f"Active: {self.current_active_visitors} | Total: {self.total_unique_visitors}", 
                   (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.putText(frame_copy, f"Frame: {self.frame_count}/{self.total_frames}", 
                   (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        return frame_copy

    def create_statistics_panel(self) -> np.ndarray:
        """Create detailed statistics panel"""
        panel_height = 400
        panel_width = self.frame_width * 2
        panel = np.zeros((panel_height, panel_width, 3), dtype=np.uint8)
        panel.fill(30)  # Dark gray background
        
        # Calculate current statistics
        current_time = time.time()
        elapsed_time = current_time - self.start_time
        
        # Title
        cv2.putText(panel, "MALL ANALYTICS DASHBOARD - REAL-TIME STATISTICS", 
                   (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        # Column 1: Visitor Statistics
        y_offset = 70
        cv2.putText(panel, "VISITOR STATISTICS:", (20, y_offset), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        
        y_offset += 30
        cv2.putText(panel, f"Total Unique Visitors: {self.total_unique_visitors}", 
                   (30, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        y_offset += 25
        cv2.putText(panel, f"Currently Active: {self.current_active_visitors}", 
                   (30, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        y_offset += 25
        avg_session_time = 0
        if len(self.people_data) > 0:
            total_time = sum([p.total_dwell_time for p in self.people_data.values()])
            avg_session_time = total_time / len(self.people_data)
        cv2.putText(panel, f"Avg Session Time: {avg_session_time:.1f}s", 
                   (30, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Column 2: Individual Dwell Times
        col2_x = 350
        y_offset = 70
        cv2.putText(panel, "INDIVIDUAL DWELL TIMES:", (col2_x, y_offset), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        
        y_offset += 30
        active_people = [(id, p) for id, p in self.people_data.items() if p.is_active]
        for i, (track_id, person) in enumerate(active_people[:8]):  # Show max 8 people
            dwell_time = person.total_dwell_time
            areas_count = len(person.areas_visited)
            cv2.putText(panel, f"ID {track_id}: {dwell_time:.1f}s ({areas_count} areas)", 
                       (col2_x + 10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
            y_offset += 22
        
        # Column 3: Peak Traffic Information
        col3_x = 700
        y_offset = 70
        cv2.putText(panel, "PEAK TRAFFIC ANALYSIS:", (col3_x, y_offset), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        
        y_offset += 30
        cv2.putText(panel, f"Current Peak Area: {self.current_peak.area}", 
                   (col3_x + 10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        y_offset += 25
        cv2.putText(panel, f"Current Peak Count: {self.current_peak.count}", 
                   (col3_x + 10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        y_offset += 25
        peak_time_str = datetime.fromtimestamp(self.start_time + self.all_time_peak.timestamp).strftime("%H:%M:%S")
        cv2.putText(panel, f"All-time Peak: {self.all_time_peak.count} at {peak_time_str}", 
                   (col3_x + 10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        y_offset += 25
        cv2.putText(panel, f"Peak Area: {self.all_time_peak.area}", 
                   (col3_x + 10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Column 4: Area Popularity
        col4_x = 1050
        y_offset = 70
        cv2.putText(panel, "AREA POPULARITY:", (col4_x, y_offset), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        
        y_offset += 30
        if self.area_popularity:
            sorted_areas = sorted(self.area_popularity.items(), key=lambda x: x[1], reverse=True)
            for area, count in sorted_areas[:6]:  # Show top 6 areas
                cv2.putText(panel, f"{area.replace('_', ' ').title()}: {count}", 
                           (col4_x + 10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
                y_offset += 22
        
        # Bottom row: System information
        bottom_y = panel_height - 50
        cv2.putText(panel, f"Processing FPS: {(self.frame_count / elapsed_time):.1f}", 
                   (20, bottom_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        
        gpu_memory = torch.cuda.memory_allocated(0) / 1024**3
        cv2.putText(panel, f"GPU Memory: {gpu_memory:.1f}GB", 
                   (250, bottom_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        
        cv2.putText(panel, f"Video Time: {elapsed_time:.1f}s", 
                   (450, bottom_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        
        return panel

    def save_analytics_report(self, filename: str = None):
        """Save comprehensive analytics report"""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"mall_analytics_report_{timestamp}.json"
        
        # Prepare detailed report
        report = {
            'analysis_timestamp': datetime.now().isoformat(),
            'video_file': self.video_path,
            'processing_info': {
                'total_frames_processed': self.frame_count,
                'total_processing_time': time.time() - self.start_time,
                'average_fps': self.frame_count / (time.time() - self.start_time),
                'gpu_used': True,
                'device': str(self.device)
            },
            'visitor_statistics': {
                'total_unique_visitors': self.total_unique_visitors,
                'current_active_visitors': self.current_active_visitors,
                'average_session_time': sum([p.total_dwell_time for p in self.people_data.values()]) / len(self.people_data) if self.people_data else 0
            },
            'peak_traffic_analysis': {
                'all_time_peak': {
                    'timestamp': self.all_time_peak.timestamp,
                    'area': self.all_time_peak.area,
                    'count': self.all_time_peak.count,
                    'frame_number': self.all_time_peak.frame_number
                },
                'current_peak': {
                    'timestamp': self.current_peak.timestamp,
                    'area': self.current_peak.area,
                    'count': self.current_peak.count,
                    'frame_number': self.current_peak.frame_number
                }
            },
            'area_popularity': dict(self.area_popularity),
            'individual_dwell_times': {
                str(track_id): {
                    'total_dwell_time': person.total_dwell_time,
                    'areas_visited': list(person.areas_visited),
                    'area_dwell_times': dict(person.area_dwell_times),
                    'total_frames': person.total_frames,
                    'is_active': person.is_active
                } for track_id, person in self.people_data.items()
            }
        }
        
        with open(filename, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"📄 Comprehensive analytics report saved to: {filename}")
        return filename

    def process_video(self, display: bool = True, save_output: str = None):
        """Process video with real-time dashboard display"""
        if not os.path.exists(self.video_path):
            print(f"❌ Video file not found: {self.video_path}")
            return
        
        cap = cv2.VideoCapture(self.video_path)
        
        # Get video properties
        self.frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.fps = cap.get(cv2.CAP_PROP_FPS)
        self.total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        print(f"🎥 Video: {self.video_path}")
        print(f"📏 Resolution: {self.frame_width}x{self.frame_height}")
        print(f"⏱️ FPS: {self.fps}")
        print(f"📊 Total frames: {self.total_frames}")
        
        # Setup video writer if saving output
        out = None
        if save_output:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            # Dashboard dimensions
            dashboard_width = self.frame_width * 2
            dashboard_height = self.frame_height + 400
            out = cv2.VideoWriter(save_output, fourcc, self.fps, 
                                (dashboard_width, dashboard_height))
        
        self.frame_count = 0
        self.start_time = time.time()
        
        print("🚀 Starting real-time dashboard processing...")
        print("Press 'q' to quit, 'r' to reset analytics, 's' to save report")
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    print("📹 End of video reached")
                    break
                
                frame_time = self.frame_count / self.fps
                
                # Detect people
                detections = self.detect_people(frame)
                
                # Track people and assign IDs
                tracked_detections = self.simple_tracker(detections, frame_time)
                
                # Update analytics
                self.update_heatmap(tracked_detections)
                self.update_traffic_peaks(tracked_detections, frame_time)
                
                # Create dashboard
                dashboard_frame = self.create_dashboard_frame(frame, tracked_detections)
                
                if display:
                    # Display dashboard
                    display_frame = cv2.resize(dashboard_frame, (1400, 900))  # Resize for display
                    cv2.imshow('Mall Analytics Dashboard', display_frame)
                    
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('q'):
                        break
                    elif key == ord('r'):
                        # Reset analytics
                        self.people_data.clear()
                        self.total_unique_visitors = 0
                        self.current_active_visitors = 0
                        self.area_popularity.clear()
                        self.heatmap_data = None
                        self.next_track_id = 0
                        print("🔄 Analytics reset")
                    elif key == ord('s'):
                        # Save current report
                        self.save_analytics_report()
                
                if out:
                    out.write(dashboard_frame)
                
                self.frame_count += 1
                
                # Print progress every 100 frames
                if self.frame_count % 100 == 0:
                    elapsed = time.time() - self.start_time
                    fps_processing = self.frame_count / elapsed
                    progress = (self.frame_count / self.total_frames) * 100
                    
                    # GPU memory monitoring
                    gpu_memory_used = torch.cuda.memory_allocated(0) / 1024**3
                    
                    print(f"Progress: {progress:.1f}% - Processing FPS: {fps_processing:.1f} - GPU: {gpu_memory_used:.1f}GB")
        
        except KeyboardInterrupt:
            print("\n⏹️ Processing interrupted by user")
        
        finally:
            # Cleanup
            cap.release()
            if out:
                out.release()
            cv2.destroyAllWindows()
            
            elapsed_total = time.time() - self.start_time
            print(f"\n✅ Processing completed!")
            print(f"📊 Total frames processed: {self.frame_count}")
            print(f"⏱️ Total time: {elapsed_total:.2f} seconds")
            print(f"🏃 Average processing FPS: {self.frame_count / elapsed_total:.2f}")
            
            if save_output:
                print(f"💾 Dashboard video saved to: {save_output}")
            
            # Save final analytics report
            report_file = self.save_analytics_report()
            print(f"📈 Final analytics report saved to: {report_file}")

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='Mall Analytics Real-time Dashboard')
    parser.add_argument('--video', type=str, 
                       default=r"C:\Users\Vallabhj\Downloads\CCTVPOC_integrated\Deep\Small video Mall shopping mall video shopping mall CCTV camera video no copyright full HD 4K video.mp4",
                       help='Path to input video file')
    parser.add_argument('--weights', type=str, default=None,
                       help='Path to YOLOv7 weights file')
    parser.add_argument('--output', type=str, default=None,
                       help='Path to save dashboard output video (optional)')
    parser.add_argument('--conf-thres', type=float, default=0.25,
                       help='Detection confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45,
                       help='IoU threshold for NMS')
    parser.add_argument('--grid-size', type=int, default=30,
                       help='Heatmap grid size in pixels')
    parser.add_argument('--decay', type=float, default=0.99,
                       help='Heatmap decay factor (0.95-0.999)')
    parser.add_argument('--no-display', action='store_true',
                       help='Don\'t display dashboard during processing')
    
    args = parser.parse_args()
    
    print("🏬 Mall Analytics Real-time Dashboard")
    print("="*60)
    
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
    
    # Initialize dashboard system
    dashboard = MallAnalyticsDashboard(
        video_path=args.video,
        weights_path=args.weights,
        conf_threshold=args.conf_thres,
        iou_threshold=args.iou_thres,
        heatmap_decay=args.decay,
        grid_size=args.grid_size
    )
    
    # Process video
    dashboard.process_video(
        display=not args.no_display,
        save_output=args.output
    )
    
    print("\n🎉 Mall Analytics Dashboard processing complete!")
    print("\n📈 Dashboard Features Demonstrated:")
    print("   ✅ Total unique visitors tracking")
    print("   ✅ Real-time heatmap visualization")
    print("   ✅ Individual person dwell time tracking")
    print("   ✅ Peak traffic timestamp and area analysis")
    print("   ✅ Live statistics and area popularity")
    print("   ✅ GPU-accelerated processing")

if __name__ == "__main__":
    main()
