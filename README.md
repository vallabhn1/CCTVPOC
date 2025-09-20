# 🏬 CCTVPOC Integrated

This repository integrates **Deep’s CCTV analytics** and **Himanshu’s Mall analytics** into a single unified pipeline.  
It uses **YOLOv7** as the primary object detection model with a **TorchVision Faster R-CNN fallback** in case YOLOv7 fails.  

---

## 📦 Features
- Unified **`UnifiedMallAnalytics`** class combining both analytics stacks.  
- **YOLOv7**-based detection (auto-downloads `yolov7.pt` if missing).  
- **TorchVision Faster R-CNN fallback** when YOLOv7 is unavailable.  
- Preserves original **Deep/** and **Himanshu/** code.  
- Generates annotated **result video** + analytics summary in terminal or via web UI.  

---

## ⚙️ Installation

### 1. Clone / Download
```bash
git clone https://github.com/yourusername/CCTVPOC_integrated.git
cd CCTVPOC_integrated
```

Or download & unzip directly.

### 2. Create Virtual Environment (recommended)
```bash
python -m venv venv
source venv/bin/activate   # Linux / macOS
venv\Scripts\activate      # Windows
```

### 3. Install Requirements
```bash
pip install -r requirements.txt
```

> ⚠️ Make sure you have **PyTorch** installed with CUDA if you want GPU acceleration.  
Install from [PyTorch official site](https://pytorch.org/get-started/locally/) if needed.

---

## ▶️ Running the Application

You can run the project in **two modes**:

### 1. Command Line Interface (CLI)
Use `main.py` to process videos directly:
```bash
python main.py --video path/to/input.mp4 --output results/ --conf_thres 0.25 --iou_thres 0.45
```

**Arguments:**
- `--video` : Path(s) to video file(s) or folder(s). *(Required)*  
- `--weights` : Path to YOLOv7 weights (default: `yolov7.pt`).  
- `--output` : Folder to save results (default: `results/`).  
- `--conf_thres` : Confidence threshold (default: `0.25`).  
- `--iou_thres` : IOU threshold for NMS (default: `0.45`).  
- `--no_display` : Suppress video display window.  

Example:
```bash
python main.py --video samples/mall.mp4 --output results/
```

---

### 2. Streamlit Web App
Use the interactive UI:
```bash
streamlit run app.py
```

Then open the browser link (usually `http://localhost:8501`).  

- Upload one or more videos.  
- Results + analytics are generated automatically into the `results/` folder.  

---

## 🧰 Project Structure
```
CCTVPOC_integrated/
│── app.py              # Streamlit web UI
│── main.py             # CLI runner
│── mall_analytics.py   # UnifiedMallAnalytics class
│── Deep/               # Deep’s original code
│── Himanshu/           # Himanshu’s original code
│── requirements.txt    # Dependencies
│── yolov7.pt           # YOLOv7 weights (downloaded automatically if missing)
│── README.md           # Documentation
```

---

## 🔑 Notes
- YOLOv7 weights (`yolov7.pt`) must be present in the root folder. If not, the script auto-downloads them.  
- For GPU acceleration, install the **correct PyTorch + CUDA version**.  
- Results are saved in `results/` with annotated video + logs.  
