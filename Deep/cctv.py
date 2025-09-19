import sys
import os
import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict, Counter
from datetime import datetime
import pandas as pd
import json
from pathlib import Path
import argparse
from typing import Dict, List, Tuple, Set
import time
import importlib

# --- YOLOv7 Safe Globals Patch ---
import torch.serialization

try:
    common = importlib.import_module("models.common")
    yolo = importlib.import_module("models.yolo")

    safe_layers = []
    for name in [
        "Conv", "Bottleneck", "SPP", "DWConv", "Focus",
        "BottleneckCSP", "C3", "C3TR", "C3SPP", "C3Ghost",
        "GhostConv", "GhostBottleneck"
    ]:
        if hasattr(common, name):
            safe_layers.append(getattr(common, name))

    if hasattr(yolo, "Model"):
        safe_layers.append(yolo.Model)

    if safe_layers:
        torch.serialization.add_safe_globals(safe_layers)
        print(f"✅ Registered YOLOv7 safe globals: {[cls.__name__ for cls in safe_layers]}")
    else:
        print("⚠️ Warning: No YOLOv7 safe globals found to register.")

except Exception as e:
    print("⚠️ Warning: YOLOv7 safe globals patch failed:", e)


# Add YOLOv7 path to system path (folder path, NOT a .pt file)
YOLOV7_PATH = r"C:\Users\Vallabhj\Downloads\CCTVPOC_integrated\yolov7"
if YOLOV7_PATH not in sys.path:
    sys.path.append(YOLOV7_PATH)

# Initialize YOLOv7 availability flag
YOLOV7_AVAILABLE = False

# Essential imports that we know work
import torchvision
import torchvision.transforms as transforms
from torchvision.models import detection

# Try to import YOLOv7 modules - if this fails, we'll use fallbacks
try:
    import importlib.util

    # Paths expected inside the YOLOv7 repo
    yolov7_models_path = os.path.join(YOLOV7_PATH, "models", "experimental.py")
    yolov7_utils_general_path = os.path.join(YOLOV7_PATH, "utils", "general.py")

    if os.path.exists(yolov7_models_path) and os.path.exists(yolov7_utils_general_path):
        # Dynamic import of YOLOv7 experimental (contains attempt_load)
        spec = importlib.util.spec_from_file_location("experimental", yolov7_models_path)
        experimental = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(experimental)

        # Use attempt_load from YOLOv7, but we'll patch it below for PyTorch 2.6+
        attempt_load = experimental.attempt_load

        # Patch attempt_load to use weights_only=False on torch.load (PyTorch 2.6+)
        def patched_attempt_load(weights, map_location=None, inplace=True, fuse=True):
            """
            Load YOLOv7 checkpoint forcing torch.load(..., weights_only=False)
            to allow custom classes to be unpickled when the safe globals are registered.
            """
            ckpt = torch.load(weights, map_location=map_location, weights_only=False)
            # YOLOv7 checkpoint commonly stores the model under 'model' key
            if isinstance(ckpt, dict) and 'model' in ckpt:
                model = ckpt['model'].float()
            else:
                # Fallback: ckpt may be the model itself
                model = ckpt.float() if hasattr(ckpt, 'float') else ckpt
            model.eval()
            return model

        # Override
        attempt_load = patched_attempt_load
        print("✅ YOLOv7 attempt_load patched to use weights_only=False")

        # Import general utils
        spec = importlib.util.spec_from_file_location("general", yolov7_utils_general_path)
        general = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(general)
        check_img_size = general.check_img_size
        non_max_suppression = general.non_max_suppression
        scale_coords = general.scale_coords
        # optional conversions
        if hasattr(general, 'xyxy2xywh'):
            xyxy2xywh = general.xyxy2xywh
        if hasattr(general, 'xywh2xyxy'):
            xywh2xyxy = general.xywh2xyxy

        # Try importing torch_utils (select_device, time_synchronized) and plots if available
        try:
            yolov7_torch_utils_path = os.path.join(YOLOV7_PATH, "utils", "torch_utils.py")
            if os.path.exists(yolov7_torch_utils_path):
                spec = importlib.util.spec_from_file_location("torch_utils", yolov7_torch_utils_path)
                torch_utils = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(torch_utils)
                select_device = torch_utils.select_device
                time_synchronized = torch_utils.time_synchronized
            else:
                raise FileNotFoundError("torch_utils.py not found")
        except Exception:
            def select_device(device=''):
                if torch.cuda.is_available() and device != 'cpu':
                    try:
                        print(f"🚀 Using GPU: {torch.cuda.get_device_name(0)}")
                    except Exception:
                        print("🚀 Using GPU")
                    return torch.device('cuda')
                else:
                    print("🔄 Using CPU (slower processing)")
                    return torch.device('cpu')

            def time_synchronized():
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                return time.time()

        try:
            yolov7_plots_path = os.path.join(YOLOV7_PATH, "utils", "plots.py")
            if os.path.exists(yolov7_plots_path):
                spec = importlib.util.spec_from_file_location("plots", yolov7_plots_path)
                plots = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(plots)
                plot_one_box = plots.plot_one_box
            else:
                raise FileNotFoundError("plots.py not found")
        except Exception:
            def plot_one_box(x, img, color=None, label=None, line_thickness=3):
                import random
                color = color or [random.randint(0, 255) for _ in range(3)]
                c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))
                cv2.rectangle(img, c1, c2, color, thickness=line_thickness, lineType=cv2.LINE_AA)
                if label:
                    tf = max(line_thickness - 1, 1)
                    t_size = cv2.getTextSize(label, 0, fontScale=line_thickness / 3, thickness=tf)[0]
                    c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
                    cv2.rectangle(img, c1, c2, color, -1, cv2.LINE_AA)
                    cv2.putText(img, label, (c1[0], c1[1] - 2), 0, line_thickness / 3, [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)

        YOLOV7_AVAILABLE = True
        print("✅ YOLOv7 modules loaded successfully using dynamic imports")
    else:
        raise ImportError("YOLOv7 module files not found at expected locations")

except Exception as e:
    print(f"⚠️ Warning: Could not load YOLOv7 modules: {e}")
    print("🔄 Using fallback implementations...")

    # Fallback implementations for missing YOLOv7 utilities

    def attempt_load(weights_path, map_location=None):
        """Fallback model loader that returns a torchvision Faster R-CNN"""
        try:
            model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
            if map_location:
                model = model.to(map_location)
            return model
        except Exception as e:
            print(f"Error loading fallback model: {e}")
            return None

    def select_device(device=''):
        """Device selection - GPU preferred, CPU fallback"""
        try:
            if torch.cuda.is_available() and device != 'cpu':
                try:
                    print(f"🚀 Using GPU: {torch.cuda.get_device_name(0)}")
                except Exception:
                    print("🚀 Using GPU")
                return torch.device('cuda')
            else:
                print("🔄 Using CPU (slower processing)")
                return torch.device('cpu')
        except Exception as e:
            print(f"⚠️ Device selection error: {e}")
            return torch.device('cpu')

    def check_img_size(img_size, s=32):
        """Fallback image size checker"""
        return img_size if img_size % s == 0 else ((img_size // s) + 1) * s

    def non_max_suppression(predictions, conf_thres=0.25, iou_thres=0.45, classes=None, agnostic=False):
        """Fallback NMS implementation using torchvision.ops.nms"""
        try:
            import torchvision
            if not isinstance(predictions, list):
                predictions = [predictions]

            output = []
            for pred in predictions:
                if pred is None or len(pred) == 0:
                    output.append(None)
                    continue

                conf_mask = pred[:, 4] > conf_thres
                pred = pred[conf_mask]

                if len(pred) == 0:
                    output.append(None)
                    continue

                boxes = pred[:, :4]
                scores = pred[:, 4]
                keep = torchvision.ops.nms(boxes, scores, iou_thres)
                output.append(pred[keep])

            return output
        except Exception as e:
            print(f"NMS fallback error: {e}")
            return [None] * len(predictions)

    def scale_coords(img1_shape, coords, img0_shape, ratio_pad=None):
        """Fallback coordinate scaling"""
        try:
            gain = min(img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1])
            coords[:, [0, 2]] /= gain  # x scaling
            coords[:, [1, 3]] /= gain  # y scaling
            return coords
        except Exception:
            return coords

    def xyxy2xywh(x):
        try:
            y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
            y[:, 0] = (x[:, 0] + x[:, 2]) / 2
            y[:, 1] = (x[:, 1] + x[:, 3]) / 2
            y[:, 2] = x[:, 2] - x[:, 0]
            y[:, 3] = x[:, 3] - x[:, 1]
            return y
        except Exception:
            return x

    def xywh2xyxy(x):
        try:
            y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
            y[:, 0] = x[:, 0] - x[:, 2] / 2
            y[:, 1] = x[:, 1] - x[:, 3] / 2
            y[:, 2] = x[:, 0] + x[:, 2] / 2
            y[:, 3] = x[:, 1] + x[:, 3] / 2
            return y
        except Exception:
            return x

    def plot_one_box(x, img, color=None, label=None, line_thickness=3):
        try:
            import random
            color = color or [random.randint(0, 255) for _ in range(3)]
            c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))
            cv2.rectangle(img, c1, c2, color, thickness=line_thickness, lineType=cv2.LINE_AA)
            if label:
                tf = max(line_thickness - 1, 1)
                t_size = cv2.getTextSize(label, 0, fontScale=line_thickness / 3, thickness=tf)[0]
                c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
                cv2.rectangle(img, c1, c2, color, -1, cv2.LINE_AA)
                cv2.putText(img, label, (c1[0], c1[1] - 2), 0, line_thickness / 3, [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)
        except Exception as e:
            print(f"Plot box fallback error: {e}")

    def time_synchronized():
        try:
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            return time.time()
        except Exception:
            return time.time()

    # Dummy dataset loader classes
    class LoadStreams:
        def __init__(self, *args, **kwargs):
            pass

    class LoadImages:
        def __init__(self, *args, **kwargs):
            pass

    print("✅ Using PyTorch torchvision for detection with fallback utilities")


class Person:
    """Class to represent a person being tracked"""
    def __init__(self, track_id: int, bbox: List[float], timestamp: float):
        self.track_id = track_id
        self.bboxes = [bbox]
        self.timestamps = [timestamp]
        self.shopping_areas_visited = set()
        self.total_time_in_mall = 0.0
        self.is_shopping = False
        self.last_area = None
        self.area_dwell_times = defaultdict(float)

    def update_position(self, bbox: List[float], timestamp: float, current_area: str = None):
        self.bboxes.append(bbox)
        self.timestamps.append(timestamp)

        if current_area:
            if self.last_area != current_area:
                self.shopping_areas_visited.add(current_area)
                self.last_area = current_area

            if len(self.timestamps) > 1:
                time_diff = timestamp - self.timestamps[-2]
                self.area_dwell_times[current_area] += time_diff

    def calculate_total_time(self):
        if len(self.timestamps) > 1:
            self.total_time_in_mall = self.timestamps[-1] - self.timestamps[0]
        return self.total_time_in_mall

    def determine_shopping_behavior(self, min_areas_for_shopping: int = 2, min_dwell_time: float = 30.0):
        total_dwell = sum(self.area_dwell_times.values())
        areas_with_significant_time = sum(1 for t in self.area_dwell_times.values() if t > min_dwell_time)
        self.is_shopping = (len(self.shopping_areas_visited) >= min_areas_for_shopping and
                            areas_with_significant_time >= 1 and total_dwell > 60.0)
        return self.is_shopping


class MallAnalytics:
    """Main class for mall CCTV analytics"""

    def __init__(self, yolov7_weights: str = None, video_path: str = None,
                 conf_threshold: float = 0.25, iou_threshold: float = 0.45):

        print("🔍 Checking GPU availability...")
        # use select_device if available (from YOLOv7 utils) otherwise fallback
        try:
            self.device = select_device('')
        except Exception:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.video_path = video_path
        self.yolov7_available = YOLOV7_AVAILABLE
        self.using_fallback = False

        # Load model with fallback
        if YOLOV7_AVAILABLE:
            # Load YOLOv7 model
            weights_path = yolov7_weights or os.path.join(YOLOV7_PATH, 'yolov7.pt')
            if not os.path.exists(weights_path):
                print(f"⚠️ YOLOv7 weights not found at {weights_path}")
                print("🔄 Using fallback detection model instead...")
                # Try downloading weights if download function exists
                try:
                    self._download_yolov7_weights(weights_path)
                except Exception:
                    self._init_fallback_model()
            else:
                try:
                    # Properly load model using patched attempt_load
                    self.model = attempt_load(weights_path, map_location=self.device)
                    # ensure model exists
                    if self.model is None:
                        raise RuntimeError("attempt_load returned None")
                    self.model.eval()

                    # Get model info safely
                    try:
                        self.img_size = check_img_size(640, s=self.model.stride.max())
                    except Exception:
                        # fallback if stride not present
                        self.img_size = 640
                    self.names = self.model.module.names if hasattr(self.model, 'module') else getattr(self.model, 'names', ['person'])
                    print("✅ YOLOv7 model loaded successfully")
                except Exception as e:
                    print(f"⚠️ Error loading YOLOv7 model: {e}")
                    print("🔄 Using fallback detection model instead...")
                    self._init_fallback_model()
                    self.using_fallback = True
        else:
            # Fallback to a simpler detection method
            print("🔄 Using fallback detection method...")
            self._init_fallback_model()
            self.using_fallback = True

        # Tracking variables
        self.people = {}
        self.next_id = 0
        self.frame_count = 0
        self.fps = 30

        # Mall areas definition
        self.mall_areas = {
            'entrance': {'x': 0, 'y': 0, 'w': 200, 'h': 480},
            'clothing_store': {'x': 200, 'y': 0, 'w': 200, 'h': 240},
            'electronics': {'x': 400, 'y': 0, 'w': 200, 'h': 240},
            'food_court': {'x': 200, 'y': 240, 'w': 400, 'h': 240},
            'exit': {'x': 600, 'y': 0, 'w': 200, 'h': 480},
            'central_corridor': {'x': 200, 'y': 120, 'w': 400, 'h': 240}
        }

        # Analytics data
        self.area_visit_counts = defaultdict(int)
        self.area_heatmap = defaultdict(int)
        self.hourly_visitors = defaultdict(int)

        # Results storage
        self.analytics_results = {
            'unique_visitors': 0,
            'shoppers': 0,
            'non_shoppers': 0,
            'area_popularity': {},
            'ignored_areas': [],
            'peak_hours': {},
            'avg_dwell_time_per_area': {}
        }

    def get_area_from_position(self, x: float, y: float) -> str:
        for area_name, area_coords in self.mall_areas.items():
            if (area_coords['x'] <= x <= area_coords['x'] + area_coords['w'] and
                    area_coords['y'] <= y <= area_coords['y'] + area_coords['h']):
                return area_name
        return 'unknown'

    def simple_tracker(self, detections: List, frame_time: float) -> List:
        if not detections:
            return []

        tracked_detections = []
        used_ids = set()

        for detection in detections:
            x1, y1, x2, y2, conf, cls = detection
            if int(cls) != 0:  # Only track persons (class 0 in COCO)
                continue

            center_x = (x1 + x2) / 2
            center_y = (y1 + y2) / 2
            bbox = [x1, y1, x2, y2]

            # Find best matching existing person
            best_match_id = None
            best_distance = float('inf')

            for person_id, person in self.people.items():
                if person_id in used_ids:
                    continue

                if person.bboxes:
                    last_bbox = person.bboxes[-1]
                    last_center_x = (last_bbox[0] + last_bbox[2]) / 2
                    last_center_y = (last_bbox[1] + last_bbox[3]) / 2

                    distance = np.sqrt((center_x - last_center_x) ** 2 + (center_y - last_center_y) ** 2)

                    if distance < best_distance and distance < 100:
                        best_distance = distance
                        best_match_id = person_id

            if best_match_id is not None:
                current_area = self.get_area_from_position(center_x, center_y)
                self.people[best_match_id].update_position(bbox, frame_time, current_area)
                tracked_detections.append((best_match_id, bbox, current_area))
                used_ids.add(best_match_id)
            else:
                current_area = self.get_area_from_position(center_x, center_y)
                new_person = Person(self.next_id, bbox, frame_time)
                new_person.update_position(bbox, frame_time, current_area)
                self.people[self.next_id] = new_person
                tracked_detections.append((self.next_id, bbox, current_area))
                self.next_id += 1

        return tracked_detections

    def detect_people(self, frame: np.ndarray):
        if self.yolov7_available and hasattr(self, 'model') and self.model is not None:
            if hasattr(self.model, 'module') or 'yolo' in str(type(self.model)).lower():
                return self._detect_yolov7(frame)
            else:
                return self._detect_fallback(frame)
        else:
            return self._detect_fallback(frame)

    def _detect_yolov7(self, frame: np.ndarray):
        # Prepare image
        img = cv2.resize(frame, (self.img_size, self.img_size))
        img = img[:, :, ::-1].transpose(2, 0, 1)
        img = np.ascontiguousarray(img)
        img = torch.from_numpy(img).to(self.device)
        img = img.float() / 255.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Inference
        with torch.no_grad():
            # Some YOLOv7 model wrappers accept `img` and return [pred]
            pred = self.model(img, augment=False)[0]

        # Apply NMS
        pred = non_max_suppression(pred, self.conf_threshold, self.iou_threshold, classes=None, agnostic=False)

        detections = []
        if pred and pred[0] is not None:
            pred[0][:, :4] = scale_coords(img.shape[2:], pred[0][:, :4], frame.shape).round()
            for *xyxy, conf, cls in pred[0]:
                detections.append([float(xyxy[0]), float(xyxy[1]), float(xyxy[2]), float(xyxy[3]), float(conf), float(cls)])

        return detections

    def _detect_fallback(self, frame: np.ndarray):
        if self.model is None:
            h, w = frame.shape[:2]
            dummy_detections = [
                [w * 0.2, h * 0.3, w * 0.3, h * 0.7, 0.8, 0],
                [w * 0.6, h * 0.2, w * 0.7, h * 0.6, 0.7, 0],
            ]
            return dummy_detections

        try:
            transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize((self.img_size, self.img_size)),
                transforms.ToTensor()
            ])
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img_tensor = transform(frame_rgb).unsqueeze(0).to(self.device)

            with torch.no_grad():
                predictions = self.model(img_tensor)

            detections = []
            if len(predictions) > 0 and len(predictions[0].get('boxes', [])) > 0:
                boxes = predictions[0]['boxes'].cpu().numpy()
                scores = predictions[0]['scores'].cpu().numpy()
                labels = predictions[0]['labels'].cpu().numpy()

                h, w = frame.shape[:2]
                scale_x = w / self.img_size
                scale_y = h / self.img_size

                for box, score, label in zip(boxes, scores, labels):
                    if label == 1 and score > self.conf_threshold:
                        x1, y1, x2, y2 = box
                        x1 *= scale_x
                        y1 *= scale_y
                        x2 *= scale_x
                        y2 *= scale_y
                        detections.append([x1, y1, x2, y2, score, 0])

            return detections

        except Exception as e:
            print(f"⚠️ Fallback detection error: {e}")
            return []

    def update_heatmap(self, tracked_detections: List):
        for track_id, bbox, area in tracked_detections:
            x1, y1, x2, y2 = bbox
            center_x = int((x1 + x2) / 2)
            center_y = int((y1 + y2) / 2)
            heatmap_key = f"{center_x // 20}_{center_y // 20}"
            self.area_heatmap[heatmap_key] += 1
            if area != 'unknown':
                self.area_visit_counts[area] += 1

    def draw_visualizations(self, frame: np.ndarray, tracked_detections: List) -> np.ndarray:
        for area_name, coords in self.mall_areas.items():
            cv2.rectangle(frame,
                          (coords['x'], coords['y']),
                          (coords['x'] + coords['w'], coords['y'] + coords['h']),
                          (100, 100, 100), 2)
            cv2.putText(frame, area_name,
                        (coords['x'] + 5, coords['y'] + 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        for track_id, bbox, area in tracked_detections:
            x1, y1, x2, y2 = bbox
            color = (
                int(255 * ((track_id * 67) % 256) / 256),
                int(255 * ((track_id * 113) % 256) / 256),
                int(255 * ((track_id * 197) % 256) / 256)
            )
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
            cv2.putText(frame, f'ID:{track_id} Area:{area}',
                        (int(x1), int(y1) - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        return frame

    def generate_heatmap(self, save_path: str = "heatmap.png"):
        if not self.area_heatmap:
            print("No heatmap data available")
            return

        max_x = max(int(key.split('_')[0]) for key in self.area_heatmap.keys())
        max_y = max(int(key.split('_')[1]) for key in self.area_heatmap.keys())

        heatmap_grid = np.zeros((max_y + 1, max_x + 1))

        for key, count in self.area_heatmap.items():
            x, y = map(int, key.split('_'))
            heatmap_grid[y, x] = count

        plt.figure(figsize=(12, 8))
        sns.heatmap(heatmap_grid, annot=False, cmap='hot', cbar=True)
        plt.title('Mall Crowd Heatmap - Areas with Maximum Crowd')
        plt.xlabel('X Position (Grid Units)')
        plt.ylabel('Y Position (Grid Units)')
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()

        print(f"Heatmap saved to: {save_path}")

    def analyze_shopping_behavior(self):
        shoppers = 0
        non_shoppers = 0

        for person in self.people.values():
            person.calculate_total_time()
            is_shopping = person.determine_shopping_behavior()
            if is_shopping:
                shoppers += 1
            else:
                non_shoppers += 1

        self.analytics_results['unique_visitors'] = len(self.people)
        self.analytics_results['shoppers'] = shoppers
        self.analytics_results['non_shoppers'] = non_shoppers

    def analyze_area_popularity(self):
        total_visits = sum(self.area_visit_counts.values())
        if total_visits == 0:
            return

        area_percentages = {}
        for area, count in self.area_visit_counts.items():
            percentage = (count / total_visits) * 100
            area_percentages[area] = {'visits': count, 'percentage': percentage}

        sorted_areas = sorted(area_percentages.items(), key=lambda x: x[1]['visits'], reverse=True)
        self.analytics_results['area_popularity'] = dict(sorted_areas)

        ignored_threshold = 0.05 * total_visits
        ignored_areas = [area for area, data in area_percentages.items() if data['visits'] < ignored_threshold]
        self.analytics_results['ignored_areas'] = ignored_areas

        avg_dwell_times = {}
        for area in self.mall_areas.keys():
            total_dwell = 0
            visitor_count = 0
            for person in self.people.values():
                if area in person.area_dwell_times:
                    total_dwell += person.area_dwell_times[area]
                    visitor_count += 1
            avg_dwell_times[area] = (total_dwell / visitor_count) if visitor_count > 0 else 0

        self.analytics_results['avg_dwell_time_per_area'] = avg_dwell_times

    def print_analytics_summary(self):
        print("\n" + "=" * 60)
        print("           MALL CCTV ANALYTICS SUMMARY")
        print("=" * 60)

        print(f"\n📊 VISITOR STATISTICS:")
        print(f"   • Unique Visitors: {self.analytics_results['unique_visitors']}")
        print(f"   • Shoppers: {self.analytics_results['shoppers']}")
        print(f"   • Non-Shoppers (Empty Handed): {self.analytics_results['non_shoppers']}")

        if self.analytics_results['unique_visitors'] > 0:
            shopping_rate = (self.analytics_results['shoppers'] / self.analytics_results['unique_visitors']) * 100
            print(f"   • Shopping Rate: {shopping_rate:.1f}%")

        print(f"\n🏬 AREA POPULARITY ANALYSIS:")
        if self.analytics_results['area_popularity']:
            for area, data in self.analytics_results['area_popularity'].items():
                print(f"   • {area.replace('_', ' ').title()}: {data['visits']} visits ({data['percentage']:.1f}%)")

        print(f"\n⏱️ AVERAGE DWELL TIME PER AREA:")
        for area, avg_time in self.analytics_results['avg_dwell_time_per_area'].items():
            print(f"   • {area.replace('_', ' ').title()}: {avg_time:.1f} seconds")

        print(f"\n🚫 IGNORED AREAS (Low Traffic):")
        if self.analytics_results['ignored_areas']:
            for area in self.analytics_results['ignored_areas']:
                print(f"   • {area.replace('_', ' ').title()}")
        else:
            print("   • None - All areas received attention")

        print("\n" + "=" * 60)

    def save_analytics_report(self, filename: str = "mall_analytics_report.json"):
        self.analytics_results['analysis_timestamp'] = datetime.now().isoformat()
        self.analytics_results['video_file'] = self.video_path

        person_details = []
        for person_id, person in self.people.items():
            person_data = {
                'id': person_id,
                'total_time_in_mall': person.total_time_in_mall,
                'areas_visited': list(person.shopping_areas_visited),
                'is_shopping': person.is_shopping,
                'area_dwell_times': dict(person.area_dwell_times)
            }
            person_details.append(person_data)

        self.analytics_results['detailed_person_data'] = person_details

        with open(filename, 'w') as f:
            json.dump(self.analytics_results, f, indent=2)

        print(f"📄 Detailed analytics report saved to: {filename}")

    def process_video(self, output_video: str = None, display_video: bool = True):
        if not self.video_path or not os.path.exists(self.video_path):
            print(f"Video file not found: {self.video_path}")
            return

        cap = cv2.VideoCapture(self.video_path)
        self.fps = cap.get(cv2.CAP_PROP_FPS) or 30

        if output_video:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_video, fourcc, self.fps,
                                  (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                                   int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))))

        print(f"🎥 Processing video: {self.video_path}")
        print(f"🔧 Using device: {self.device}")
        print(f"⚙️ FPS: {self.fps}")

        frame_count = 0
        start_time = time.time()

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_time = frame_count / self.fps

            detections = self.detect_people(frame)
            tracked_detections = self.simple_tracker(detections, frame_time)
            self.update_heatmap(tracked_detections)
            frame = self.draw_visualizations(frame, tracked_detections)

            cv2.putText(frame, f"Unique Visitors: {len(self.people)}",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(frame, f"Frame: {frame_count}",
                        (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(frame, f"Active Tracks: {len(tracked_detections)}",
                        (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            if output_video:
                out.write(frame)

            if display_video:
                cv2.imshow('Mall CCTV Analytics', frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

            frame_count += 1
            if frame_count % 100 == 0:
                elapsed = time.time() - start_time
                fps_processing = frame_count / elapsed if elapsed > 0 else 0
                print(f"Processed {frame_count} frames (Processing FPS: {fps_processing:.1f})")

        cap.release()
        if output_video:
            out.release()
        cv2.destroyAllWindows()

        print(f"✅ Video processing completed. Total frames: {frame_count}")

        self.analyze_shopping_behavior()
        self.analyze_area_popularity()
        self.print_analytics_summary()
        self.generate_heatmap("mall_crowd_heatmap.png")
        self.save_analytics_report()

    def _init_fallback_model(self):
        try:
            self.model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
            self.model.to(self.device)
            self.model.eval()
            self.img_size = 640
            # Minimal COCO person label; enlarge if needed
            self.names = ['person']
            print("✅ Fallback model (Faster R-CNN) initialized successfully")
        except Exception as e:
            print(f"❌ Error initializing fallback model: {e}")
            self.model = None
            self.img_size = 640
            self.names = ['person']
            print("⚠️ Using dummy detection for testing purposes")

    def _download_yolov7_weights(self, weights_path):
        try:
            import urllib.request
            url = "https://github.com/WongKinYiu/yolov7/releases/download/v0.1/yolov7.pt"
            print(f"Downloading YOLOv7 weights from {url}...")
            urllib.request.urlretrieve(url, weights_path)
            print("✅ YOLOv7 weights downloaded successfully")
        except Exception as e:
            print(f"❌ Error downloading weights: {e}")
            raise e


def main():
    parser = argparse.ArgumentParser(description='Mall CCTV Analytics using YOLOv7')
    parser.add_argument('--video', type=str, default=r"C:\Users\Vallabhj\Downloads\CCTVPOC_integrated\Deep\Small video Mall shopping mall video shopping mall CCTV camera video no copyright full HD 4K video.mp4",
                        help='Path to input video file')
    parser.add_argument('--weights', type=str, default=None,
                        help='Path to YOLOv7 weights file')
    parser.add_argument('--output', type=str, default='output_analytics.mp4',
                        help='Path to output video file')
    parser.add_argument('--conf-thres', type=float, default=0.25,
                        help='Confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45,
                        help='IoU threshold for NMS')
    parser.add_argument('--no-display', action='store_true',
                        help='Don\'t display video during processing')

    args = parser.parse_args()

    print("🚀 Starting Mall CCTV Analytics System")
    print("🔍 Checking system requirements...")

    if not torch.cuda.is_available():
        print("⚠️ WARNING: CUDA/GPU is not available!")
        print("🔄 Falling back to CPU processing (slower but functional)")
        print("📊 Continuing with CPU processing...")
    else:
        try:
            print(f"✅ GPU Available: {torch.cuda.get_device_name(0)}")
        except Exception:
            print("✅ GPU Available")
        print(f"🔥 CUDA Version: {torch.version.cuda}")

    print(f"📹 Video: {args.video}")
    print(f"🎯 YOLOv7 Path: {YOLOV7_PATH}")

    analytics = MallAnalytics(
        yolov7_weights=args.weights,
        video_path=args.video,
        conf_threshold=args.conf_thres,
        iou_threshold=args.iou_thres
    )

    analytics.process_video(
        output_video=args.output,
        display_video=not args.no_display
    )

    print("🎉 Analysis complete! Check the generated files:")
    print("   • mall_crowd_heatmap.png - Heatmap visualization")
    print("   • mall_analytics_report.json - Detailed analytics report")
    print(f"   • {args.output} - Annotated video output")


if __name__ == "__main__":
    main()
