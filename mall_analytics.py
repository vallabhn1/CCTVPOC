import torch
import os
import json

from Himanshu.mall_analytics import MallAnalytics as HimanshuAnalytics
from Deep.cctv import MallAnalytics as DeepAnalytics


def ensure_weights(weights):
    """Return YOLOv7 weights path or default."""
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
    """
    Run both Himanshu and Deep analytics on a video,
    produce annotated videos, JSON reports, and a combined JSON.
    """

    def __init__(self, video=None, weights="yolo\yolov7.pt", output=None):
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

    def run(self, video, output, conf_thres=0.25, iou_thres=0.45):
        print("🚀 Starting UnifiedMallAnalytics")

        if output:
            base, ext = os.path.splitext(output)
            him_output = f"{base}_himanshu_result{ext}"
            him_report = f"{base}_himanshu_report.json"
            deep_output = f"{base}_deep_result{ext}"
            deep_report = f"{base}_deep_report.json"
            combined_report = f"{base}_combined_report.json"
        else:
            him_output, deep_output, him_report, deep_report, combined_report = None, None, None, None, None

        # ---------------- Himanshu ----------------
        him_data = {}
        if self.him:
            print("▶️ Running Himanshu analytics...")
            try:
                self.him.process_video(video, him_output, report_path=him_report)

                if os.path.exists(him_report):
                    with open(him_report, "r") as f:
                        him_data = json.load(f)

                    # Map class IDs to labels if needed
                    if "detections_per_class" in him_data:
                        mapped = {}
                        for cls_id, count in him_data["detections_per_class"].items():
                            cls_name = COCO_CLASSES[int(cls_id)] if int(cls_id) < len(COCO_CLASSES) else f"class_{cls_id}"
                            mapped[cls_name] = count
                        him_data["detections_per_class"] = mapped

                        with open(him_report, "w") as f:
                            json.dump(him_data, f, indent=4)

                print(f"✅ Himanshu analytics finished. Output: {him_output}, Report: {him_report}")
            except Exception as e:
                print(f"❌ Himanshu analytics failed: {e}")
        else:
            print("ℹ️ Himanshu analytics not available, skipping")

        # ---------------- Deep ----------------
        deep_data = {}
        if self.deep:
            print("▶️ Running Deep analytics...")
            try:
                self.deep.process_video(output_video=deep_output, display_video=False)

                # Deep writes its own JSON
                if os.path.exists("mall_analytics_report.json"):
                    with open("mall_analytics_report.json", "r") as f:
                        deep_data = json.load(f)
                    with open(deep_report, "w") as f:
                        json.dump(deep_data, f, indent=4)

                print(f"✅ Deep analytics finished. Output: {deep_output}, Report: {deep_report}")
            except Exception as e:
                print(f"❌ Deep analytics failed: {e}")
        else:
            print("ℹ️ Deep analytics not available, skipping")

        # ---------------- Combine Reports ----------------
        if combined_report:
            combined = {
                "video": video,
                "himanshu": him_data if him_data else None,
                "deep": deep_data if deep_data else None
            }
            with open(combined_report, "w") as f:
                json.dump(combined, f, indent=4)
            print(f"📊 Combined report saved to {combined_report}")

        print("✅ Unified analysis finished")
