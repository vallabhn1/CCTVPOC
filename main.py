import argparse
import os
import glob
from mall_analytics import UnifiedMallAnalytics


def run(video, weights, output, conf_thres, iou_thres, no_display):
    core = UnifiedMallAnalytics(video=video, weights=weights, output=output)
    core.run(video, output, conf_thres, iou_thres)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--video", nargs="+", type=str, required=True,
        help="path(s) to input video(s) or folder(s)"
    )
    parser.add_argument("--weights", type=str, default="yolov7.pt", help="YOLOv7 weights path")
    parser.add_argument("--output", type=str, default=None, help="output base name (optional)")
    parser.add_argument("--conf-thres", type=float, default=0.25, help="confidence threshold")
    parser.add_argument("--iou-thres", type=float, default=0.45, help="IoU threshold")
    parser.add_argument("--no-display", action="store_true", help="disable display")

    args = parser.parse_args()

    # Collect all video files
    video_files = []
    for v in args.video:
        if os.path.isdir(v):
            video_files.extend(glob.glob(os.path.join(v, "*.mp4")))
        elif os.path.isfile(v):
            video_files.append(v)
        else:
            print(f"⚠️ Skipping invalid path: {v}")

    if not video_files:
        print("❌ No valid video files found.")
        exit(1)

    # Process each video separately
    for vid in video_files:
        base_name = os.path.splitext(os.path.basename(vid))[0]

        # If user gave --output, respect it, else build per-video output name
        if args.output:
            output_name = args.output
        else:
            output_name = f"{base_name}_result.mp4"

        print(f"\n🎥 Processing video: {vid}")
        run(vid, args.weights, output_name, args.conf_thres, args.iou_thres, args.no_display)
