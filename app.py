import streamlit as st
import os
import shutil
import subprocess
import sys
import glob
from mall_analytics import UnifiedMallAnalytics

st.set_page_config(page_title="Mall CCTV Analytics", layout="wide")

# Persistent state
if "uploaded_files" not in st.session_state:
    st.session_state.uploaded_files = []
if "selected_video" not in st.session_state:
    st.session_state.selected_video = None
if "results_zip" not in st.session_state:
    st.session_state.results_zip = None

# ------------------------------
# PAGE 1: Video Selection
# ------------------------------
if st.session_state.selected_video is None:
    st.title("🏬 Mall CCTV Analytics - Select a Video")

    uploaded_files = st.file_uploader(
        "📤 Upload one or more videos",
        type=["mp4", "avi", "mov"],
        accept_multiple_files=True
    )

    # Save uploads persistently
    if uploaded_files:
        os.makedirs("uploads", exist_ok=True)
        st.session_state.uploaded_files = uploaded_files

    if st.session_state.uploaded_files:
        cols = st.columns(len(st.session_state.uploaded_files))  # horizontal layout

        for i, uploaded_file in enumerate(st.session_state.uploaded_files):
            video_path = os.path.join("uploads", uploaded_file.name)
            if not os.path.exists(video_path):  # Save only once
                with open(video_path, "wb") as f:
                    f.write(uploaded_file.read())

            with cols[i]:
                st.video(video_path, format="video/mp4", start_time=0)
                st.caption(uploaded_file.name)

                if st.button(f"▶️ Select {uploaded_file.name}", key=f"select_{i}"):
                    st.session_state.selected_video = video_path
                    st.rerun()

# ------------------------------
# PAGE 2: Analysis
# ------------------------------
else:
    video_path = st.session_state.selected_video
    base_name, _ = os.path.splitext(os.path.basename(video_path))
    results_dir = "results"
    os.makedirs(results_dir, exist_ok=True)

    st.title(f"📹 Selected Video: {os.path.basename(video_path)}")
    st.video(video_path)

    if st.button("🚀 Run Analytics + Realtime Dashboard"):
        st.write("⏳ Cleaning old dashboard reports/heatmaps...")

        # --- Clean old mall_analytics_dashboard outputs ---
        for f in os.listdir("."):
            if f.startswith("mall_analytics_report") and f.endswith(".json"):
                os.remove(f)
            if f.startswith("mall_crowd_heatmap") or f.startswith("heatmap"):
                if f.endswith(".png") or f.endswith(".jpg"):
                    os.remove(f)

        st.write("🧹 Old reports and heatmaps removed.")
        st.write("🚀 Running analytics and launching realtime dashboard...")

        deep_output = os.path.join(results_dir, f"{base_name}_deep_result.mp4")

        try:
            # --- Launch realtime dashboard in separate window ---
            subprocess.Popen([
                sys.executable, "Deep/mall_analytics_dashboard.py",
                "--video", video_path
            ])

            # --- Run Unified Analytics (saves himanshu + deep + combined) ---
            core = UnifiedMallAnalytics(video=video_path, weights="yolov7.pt", output=deep_output)
            core.run(video_path, deep_output, conf_thres=0.25, iou_thres=0.45)

            # --- Delete intermediate Himanshu/Deep JSONs (keep only combined) ---
            him_report = os.path.join(results_dir, f"{base_name}_himanshu_report.json")
            deep_report = os.path.join(results_dir, f"{base_name}_deep_report.json")
            for f in [him_report, deep_report]:
                if os.path.exists(f):
                    os.remove(f)

            st.success("✅ Analytics completed! Now click below to download results.")

        except Exception as e:
            st.error(f"❌ Error running analytics: {e}")

    # Download button creates ZIP only when clicked
    if st.button("⬇️ Download Analytics Results (ZIP)"):
        zip_dir = os.path.join(results_dir, f"{base_name}_outputs")
        os.makedirs(zip_dir, exist_ok=True)

        # ✅ Copy annotated videos (himanshu + deep) using glob
        for pattern in [
            f"{base_name}*himanshu_result.mp4",
            f"{base_name}*deep_result.mp4"
        ]:
            for fpath in glob.glob(os.path.join(results_dir, pattern)):
                shutil.copy(fpath, zip_dir)

        # ✅ Copy combined report
        combined_report = os.path.join(results_dir, f"{base_name}_combined_report.json")
        if os.path.exists(combined_report):
            shutil.copy(combined_report, zip_dir)

        # ✅ Copy latest dashboard JSON
        json_files = glob.glob("mall_analytics_report*.json")
        dashboard_files = []
        if json_files:
            latest_json = max(json_files, key=os.path.getmtime)
            shutil.copy(latest_json, os.path.join(zip_dir, os.path.basename(latest_json)))
            dashboard_files.append(latest_json)

        # ✅ Copy all latest heatmaps
        heatmap_files = []
        for f in os.listdir("."):
            if f.startswith("mall_crowd_heatmap") or f.startswith("heatmap"):
                if f.endswith(".png") or f.endswith(".jpg"):
                    shutil.copy(f, os.path.join(zip_dir, f))
                    heatmap_files.append(f)

        # ✅ Create zip archive
        zip_path = shutil.make_archive(zip_dir, "zip", zip_dir)

        # ✅ Clean up dashboard files from disk
        for f in dashboard_files + heatmap_files:
            if os.path.exists(f):
                os.remove(f)

        with open(zip_path, "rb") as f:
            zip_bytes = f.read()
        st.download_button(
            label="📥 Save Analytics ZIP",
            data=zip_bytes,
            file_name=f"{base_name}_results.zip",
            mime="application/zip"
        )

    if st.button("🔙 Back to Selection"):
        st.session_state.selected_video = None
        st.session_state.results_zip = None
        st.rerun()
