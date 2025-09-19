import cv2
import torch
import os
import sys
import numpy as np
import json
import time
from pathlib import Path

# -------------------------------------------------
# 🔧 Fix: Make local yolov7 folder importable
# -------------------------------------------------
YOLOV7_PATH = os.path.join(os.path.dirname(__file__), "..", "yolov7")
if YOLOV7_PATH not in sys.path:
    sys.path.append(YOLOV7_PATH)

try:
    from models.experimental import attempt_load
    from utils.general import non_max_suppression, scale_coords
    from utils.torch_utils import select_device
    from utils.datasets import letterbox   # ✅ preprocessing

    # Patch attempt_load to always use weights_only=False
    _orig_attempt_load = attempt_load

    def safe_attempt_load(weights, map_location=None, inplace=True, fuse=True):
        ckpt = torch.load(weights, map_location=map_location, weights_only=False)
        if isinstance(ckpt, dict) and "model" in ckpt:
            model = ckpt["model"].float()
        else:
            model = ckpt.float() if hasattr(ckpt, "float") else ckpt
        model.eval()
        return model

    attempt_load = safe_attempt_load
    YOLOV7_AVAILABLE = True
    print("✅ YOLOv7 modules imported successfully (Himanshu)")
except ImportError as e:
    print("⚠️ YOLOv7 not available in Himanshu module, fallback will be used:", e)
    attempt_load = None
    YOLOV7_AVAILABLE = False


class MallAnalytics:
    def __init__(self, weights="yolov7.pt"):
        print("🔍 Checking GPU availability...")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.weights = weights
        self.model = None

        if YOLOV7_AVAILABLE and attempt_load is not None:
            try:
                self.model = attempt_load(self.weights, map_location=self.device)
                self.model.to(self.device).eval()
                print("✅ YOLOv7 model loaded successfully in Himanshu")
            except Exception as e:
                print("⚠️ YOLOv7 failed in Himanshu, using fallback:", e)
                self.model = None
        else:
            print("⚠️ YOLOv7 not available, using fallback in Himanshu")
            self.model = None

    def process_video(self, video_path, output_path="result_himanshu.mp4", report_path="result_himanshu_report.json"):
        print(f"🎥 [Himanshu] Processing video: {video_path}")

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"❌ Could not open video file: {video_path}")
            return

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        # 🔢 Stats
        frame_count = 0
        total_detections = 0
        class_counts = {}
        start_time = time.time()

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame_count += 1

            if self.model is not None:
                # ✅ Preprocess frame
                img = letterbox(frame, new_shape=640)[0]
                img = img[:, :, ::-1].transpose(2, 0, 1)
                img = np.ascontiguousarray(img)

                img = torch.from_numpy(img).to(self.device)
                img = img.float() / 255.0
                img = img.unsqueeze(0)

                # Run detection
                with torch.no_grad():
                    pred = self.model(img)[0]
                    detections = non_max_suppression(pred, conf_thres=0.25, iou_thres=0.45)

                for det in detections:
                    if det is not None and len(det):
                        det[:, :4] = scale_coords(img.shape[2:], det[:, :4], frame.shape).round()
                        for *xyxy, conf, cls in reversed(det):
                            cls_id = int(cls.item())
                            total_detections += 1
                            class_counts[cls_id] = class_counts.get(cls_id, 0) + 1
                            cv2.rectangle(frame, (int(xyxy[0]), int(xyxy[1])),
                                          (int(xyxy[2]), int(xyxy[3])), (0, 255, 0), 2)

            out.write(frame)

        cap.release()
        out.release()
        elapsed = time.time() - start_time

        # ✅ Save report
        report = {
            "status": "success",
            "module": "Himanshu",
            "video": video_path,
            "output_video": output_path,
            "frames_processed": frame_count,
            "total_detections": total_detections,
            "detections_per_class": class_counts,
            "fps": round(frame_count / elapsed, 2)
        }

        with open(report_path, "w") as f:
            json.dump(report, f, indent=4)

        print(f"✅ Himanshu analytics finished. Processed {frame_count} frames. "
              f"Detections: {total_detections}. Output: {output_path}, Report: {report_path}")
