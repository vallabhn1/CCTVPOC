import streamlit as st
import os
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
                core = UnifiedMallAnalytics(video=video_path, weights="yolov7.pt", output=output_video)
                core.run(video_path, output_video, conf_thres=0.25, iou_thres=0.45)

                st.success(f"✅ Finished {uploaded_file.name}")

                # Show processed video
                if os.path.exists(output_video):
                    st.video(output_video)

                # Reports
                him_report = os.path.join("results", f"{base_name}_himanshu_report.json")
                deep_report = os.path.join("results", f"{base_name}_deep_report.json")

                if os.path.exists(him_report):
                    st.write("📄 Himanshu Report:")
                    st.json(open(him_report).read())

                if os.path.exists(deep_report):
                    st.write("📄 Deep Report:")
                    st.json(open(deep_report).read())

            except Exception as e:
                st.error(f"❌ Error processing {uploaded_file.name}: {e}")
