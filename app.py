import streamlit as st
import os
import shutil
from mall_analytics import UnifiedMallAnalytics

st.title("🏬 Mall CCTV Analytics (Batch Mode with Run Button)")

# Upload multiple videos
uploaded_files = st.file_uploader(
    "📤 Upload one or more videos",
    type=["mp4", "avi", "mov"],
    accept_multiple_files=True
)

if uploaded_files:
    os.makedirs("uploads", exist_ok=True)
    os.makedirs("results", exist_ok=True)

    # Preview uploaded videos
    for uploaded_file in uploaded_files:
        st.subheader(f"🎥 {uploaded_file.name}")
        video_path = os.path.join("uploads", uploaded_file.name)
        with open(video_path, "wb") as f:
            f.write(uploaded_file.read())
        st.video(video_path)

    # Button to run analytics on all videos at once
    if st.button("▶️ Run Analytics on All Videos"):
        st.write(f"⏳ Processing {len(uploaded_files)} video(s)...")

        for uploaded_file in uploaded_files:
            base_name, _ = os.path.splitext(uploaded_file.name)
            video_path = os.path.join("uploads", uploaded_file.name)
            output_video = os.path.join("results", f"{base_name}_result.mp4")

            try:
                # Run analytics
                core = UnifiedMallAnalytics(video=video_path, weights="yolov7.pt", output=output_video)
                core.run(video_path, output_video, conf_thres=0.25, iou_thres=0.45)

                st.success(f"✅ Finished {uploaded_file.name}")

                # Show processed video
                if os.path.exists(output_video):
                    st.video(output_video)

                # Reports
                him_report = os.path.join("results", f"{base_name}_himanshu_report.json")
                deep_report = os.path.join("results", f"{base_name}_deep_report.json")

                # Create zip archive for all outputs
                zip_dir = os.path.join("results", f"{base_name}_outputs")
                os.makedirs(zip_dir, exist_ok=True)

                # Copy files into zip folder
                if os.path.exists(output_video):
                    shutil.copy(output_video, zip_dir)
                if os.path.exists(him_report):
                    shutil.copy(him_report, zip_dir)
                if os.path.exists(deep_report):
                    shutil.copy(deep_report, zip_dir)

                zip_path = shutil.make_archive(zip_dir, "zip", zip_dir)

                # Show download button
                with open(zip_path, "rb") as f:
                    st.download_button(
                        label="⬇️ Download All Results (ZIP)",
                        data=f,
                        file_name=f"{base_name}_results.zip",
                        mime="application/zip"
                    )

            except Exception as e:
                st.error(f"❌ Error processing {uploaded_file.name}: {e}")
