import torch
import os
import json

from Himanshu.mall_analytics import MallAnalytics as HimanshuAnalytics
from Deep.cctv import MallAnalytics as DeepAnalytics


def ensure_weights(weights):
    return weights if weights else "yolov7.pt"


# ✅ COCO class labels for YOLOv7
COCO_CLASSES = [
    "person", "bicycle", "car", "motorbike", "aeroplane", "bus",
    "train", "truck", "boat", "traffic light", "fire hydrant", "stop sign",
    "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep",
    "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
    "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard",
    "sports ball", "kite", "baseball bat", "baseball glove", "skateboard",
    "surfboard", "tennis racket", "bottle", "wine glass", "cup", "fork",
    "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
    "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair",
    "sofa", "pottedplant", "bed", "diningtable", "toilet", "tvmonitor",
    "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave",
    "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase",
    "scissors", "teddy bear", "hair drier", "toothbrush"
]


class UnifiedMallAnalytics:
    def __init__(self, video=None, weights="yolov7.pt", output=None):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.weights = ensure_weights(weights)
        self.video = video
        self.output = output

        # ---------------- Himanshu ----------------
        try:
            self.him = HimanshuAnalytics(weights=self.weights)
        except Exception as e:
            print(f"⚠️ Could not import Himanshu analytics: {e}")
            self.him = None

        # ---------------- Deep ----------------
        try:
            if "video_path" in DeepAnalytics.__init__.__code__.co_varnames:
                self.deep = DeepAnalytics(video_path=video, yolov7_weights=self.weights)
            else:
                self.deep = DeepAnalytics(yolov7_weights=self.weights)
        except Exception as e:
            print(f"⚠️ Could not import Deep analytics: {e}")
            self.deep = None

    def run(self, video, output, conf_thres, iou_thres):
        print("🚀 Starting UnifiedMallAnalytics")

        if output:
            base, ext = os.path.splitext(output)
            him_output = f"{base}_himanshu{ext}"
            him_report = f"{base}_himanshu_report.json"
            deep_output = f"{base}_deep{ext}"
            # Deep already generates mall_analytics_report.json
        else:
            him_output, deep_output, him_report = None, None, None

        # ---------------- Himanshu ----------------
        if self.him:
            print("▶️ Running Himanshu analytics...")
            try:
                self.him.process_video(video, him_output, report_path=him_report)

                # Post-process JSON: map class IDs to names
                if os.path.exists(him_report):
                    with open(him_report, "r") as f:
                        data = json.load(f)

                    if "detections_per_class" in data:
                        mapped = {}
                        for cls_id, count in data["detections_per_class"].items():
                            cls_name = COCO_CLASSES[int(cls_id)] if int(cls_id) < len(COCO_CLASSES) else f"class_{cls_id}"
                            mapped[cls_name] = count
                        data["detections_per_class"] = mapped

                        with open(him_report, "w") as f:
                            json.dump(data, f, indent=4)

                print(f"✅ Himanshu analytics finished. Output: {him_output}, Report: {him_report}")
            except Exception as e:
                print(f"❌ Himanshu analytics failed: {e}")
        else:
            print("ℹ️ Himanshu analytics not available, skipping")

        # ---------------- Deep ----------------
        if self.deep:
            print("▶️ Running Deep analytics...")
            try:
                # Deep already generates mall_analytics_report.json + heatmap
                self.deep.process_video(output_video=deep_output, display_video=False)
                print(f"✅ Deep analytics finished. Output: {deep_output}, Report: mall_analytics_report.json")
            except Exception as e:
                print(f"❌ Deep analytics failed: {e}")
        else:
            print("ℹ️ Deep analytics not available, skipping")

        print("✅ Unified analysis finished")
