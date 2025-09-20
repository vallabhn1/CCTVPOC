# Mall Analytics Real-time Dashboard

A comprehensive CCTV analytics system that provides real-time mall visitor analysis with GPU acceleration. The dashboard combines advanced computer vision, person tracking, and statistical analysis to deliver actionable insights about mall traffic patterns.

## 🌟 Features

### Core Analytics
- **Total Unique Visitors**: Real-time count of unique individuals entering the mall
- **Live Heatmap**: Dynamic visualization of crowd density and movement patterns
- **Individual Dwell Time**: Track how long each person (by ID) spends in the mall
- **Peak Traffic Analysis**: Identify timestamps and areas with highest traffic
- **Area Popularity**: Real-time statistics on which areas attract the most visitors

### Real-time Dashboard Components
1. **Live Video Feed** (Top-Left): Original video with person detection and tracking
2. **Traffic Heatmap** (Top-Right): Real-time heat visualization of crowd movement
3. **Statistics Panel** (Bottom): Comprehensive analytics including:
   - Visitor statistics (total, active, session times)
   - Individual dwell times for each tracked person
   - Peak traffic information with timestamps
   - Area popularity rankings
   - System performance metrics

## 🚀 System Requirements

### Hardware
- **NVIDIA GPU** with CUDA support (Required)
- **CUDA 12.7** compatible GPU drivers
- **8GB+ RAM** recommended
- **Modern CPU** (Intel i5/AMD Ryzen 5 or better)

### Software
- **Windows 10/11** with PowerShell
- **Python 3.8+**
- **NVIDIA GPU Driver 566.07** or compatible
- **CUDA Toolkit 12.7**

### Python Dependencies
```bash
torch>=2.3.1+cu121
torchvision>=0.18.1+cu121
torchaudio>=2.3.1+cu121
opencv-python>=4.8.0
numpy>=1.24.0
matplotlib>=3.7.0
seaborn>=0.12.0
pandas>=2.0.0
```

## 📦 Installation

### 1. Install CUDA PyTorch (REQUIRED)
```powershell
# Uninstall any existing PyTorch (CPU version)
pip uninstall torch torchvision torchaudio -y

# Install CUDA 12.1 compatible PyTorch
pip install torch==2.3.1+cu121 torchvision==0.18.1+cu121 torchaudio==2.3.1+cu121 -f https://download.pytorch.org/whl/torch_stable.html
```

### 2. Install Additional Dependencies
```powershell
pip install opencv-python numpy matplotlib seaborn pandas
```

### 3. Verify GPU Setup
```powershell
python -c "import torch; print('CUDA Available:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None')"
```

## 🎯 Quick Start

### Method 1: Using Batch File (Recommended)
1. Double-click `run_mall_dashboard.bat`
2. The dashboard will start automatically with optimal settings

### Method 2: Command Line
```powershell
# Basic usage
python mall_analytics_dashboard.py

# With custom video file
python mall_analytics_dashboard.py --video "path/to/your/video.mp4"

# Save dashboard output
python mall_analytics_dashboard.py --output "dashboard_output.mp4"

# Custom settings
python mall_analytics_dashboard.py --conf-thres 0.3 --grid-size 25 --decay 0.98
```

## 🎮 Controls

During dashboard operation:
- **'q'** - Quit the application
- **'r'** - Reset all analytics data (start fresh)
- **'s'** - Save current analytics report to JSON file

## 📊 Dashboard Layout

```
┌─────────────────────────┬─────────────────────────┐
│                         │                         │
│    Live Video Feed      │    Traffic Heatmap      │
│   + Person Detection    │   + Crowd Density       │
│   + Area Boundaries     │   + Movement Patterns   │
│                         │                         │
├─────────────────────────┴─────────────────────────┤
│                Statistics Panel                    │
│ ┌─────────────┬─────────────┬─────────────┬─────────┐ │
│ │  Visitor    │ Individual  │Peak Traffic │  Area   │ │
│ │ Statistics  │Dwell Times  │ Analysis    │ Popular │ │
│ │             │             │             │         │ │
│ └─────────────┴─────────────┴─────────────┴─────────┘ │
│                System Performance                     │
└───────────────────────────────────────────────────────┘
```

## 📈 Output Files

### 1. Real-time Dashboard Video (Optional)
- **Filename**: `dashboard_output.mp4` (if --output specified)
- **Content**: Full dashboard with all panels and statistics
- **Resolution**: 2x video width + 400px height for stats

### 2. Analytics Report (Auto-generated)
- **Filename**: `mall_analytics_report_YYYYMMDD_HHMMSS.json`
- **Content**: Comprehensive analytics data including:
  ```json
  {
    "analysis_timestamp": "2025-01-24T...",
    "visitor_statistics": {
      "total_unique_visitors": 15,
      "current_active_visitors": 3,
      "average_session_time": 45.2
    },
    "peak_traffic_analysis": {
      "all_time_peak": {
        "timestamp": 1234567890.123,
        "area": "food_court",
        "count": 8,
        "frame_number": 1250
      }
    },
    "individual_dwell_times": {
      "0": {
        "total_dwell_time": 67.5,
        "areas_visited": ["entrance", "clothing_store", "food_court"],
        "area_dwell_times": {
          "entrance": 5.2,
          "clothing_store": 25.1,
          "food_court": 37.2
        }
      }
    }
  }
  ```

## 🏗️ Mall Area Configuration

The system includes predefined mall areas that can be customized in the code:

```python
self.mall_areas = {
    'entrance': {'x': 0, 'y': 0, 'w': 200, 'h': 480, 'color': (0, 255, 0)},
    'clothing_store': {'x': 200, 'y': 0, 'w': 200, 'h': 240, 'color': (255, 0, 0)},
    'electronics': {'x': 400, 'y': 0, 'w': 200, 'h': 240, 'color': (0, 0, 255)},
    'food_court': {'x': 200, 'y': 240, 'w': 400, 'h': 240, 'color': (255, 255, 0)},
    'exit': {'x': 600, 'y': 0, 'w': 200, 'h': 480, 'color': (255, 0, 255)},
    'central_corridor': {'x': 200, 'y': 120, 'w': 400, 'h': 240, 'color': (0, 255, 255)}
}
```

## ⚙️ Command Line Options

| Option | Default | Description |
|--------|---------|-------------|
| `--video` | Auto-detected | Path to input video file |
| `--weights` | None | Path to YOLOv7 weights file (optional) |
| `--output` | None | Path to save dashboard video output |
| `--conf-thres` | 0.25 | Detection confidence threshold (0.1-0.9) |
| `--iou-thres` | 0.45 | IoU threshold for NMS (0.1-0.9) |
| `--grid-size` | 30 | Heatmap grid size in pixels (10-50) |
| `--decay` | 0.99 | Heatmap decay factor (0.95-0.999) |
| `--no-display` | False | Don't show dashboard during processing |

## 🔧 Troubleshooting

### GPU/CUDA Issues
```
❌ CUDA/GPU is NOT available!
```
**Solution**: 
1. Install NVIDIA GPU drivers (566.07+)
2. Install CUDA Toolkit 12.7
3. Install PyTorch with CUDA support:
   ```powershell
   pip install torch==2.3.1+cu121 torchvision==0.18.1+cu121 --index-url https://download.pytorch.org/whl/cu121
   ```

### Memory Issues
```
CUDA out of memory
```
**Solution**:
1. Reduce `--grid-size` to 20 or 15
2. Process smaller resolution videos
3. Close other GPU-intensive applications

### Performance Issues
```
Low processing FPS
```
**Solution**:
1. Increase `--conf-thres` to 0.4 (fewer detections)
2. Use `--no-display` for faster processing
3. Ensure GPU drivers are up to date

### Video File Issues
```
Video file not found
```
**Solution**:
1. Check video file path and permissions
2. Ensure video format is supported (MP4, AVI, MOV)
3. Use absolute file paths

## 🎯 Key Metrics Explained

### 1. Dwell Time
- **Definition**: Total time a person spends in the mall from first detection to last detection
- **Calculation**: `last_seen_timestamp - first_seen_timestamp`
- **Use Case**: Understand visitor engagement and shopping behavior

### 2. Peak Traffic
- **Current Peak**: Highest number of people in any area at the current moment
- **All-time Peak**: Highest number of people recorded in any area during the entire session
- **Includes**: Timestamp, area name, people count, and frame number

### 3. Area Popularity
- **Metric**: Total number of person-visits to each area
- **Ranking**: Areas sorted by total visitor count
- **Insight**: Identifies most and least attractive areas

### 4. Active vs Total Visitors
- **Active**: Currently visible and being tracked
- **Total**: Unique individuals seen throughout the session
- **Retention**: Percentage of visitors currently active

## 🚀 Performance Optimization

### GPU Memory Management
- Automatic GPU cache clearing
- Mixed precision (BF16) when supported
- Non-blocking tensor transfers
- Optimized batch processing

### Processing Speed
- Efficient tracking algorithm
- Optimized heatmap updates
- Minimal memory allocation
- GPU-accelerated inference

### Display Optimization
- Resized display for performance
- Efficient dashboard rendering
- Real-time statistics updates
- Memory-conscious visualization

## 📝 License & Usage

This software is designed for mall analytics and visitor behavior research. Ensure compliance with local privacy laws and regulations when using CCTV footage for analytics.

## 🤝 Support

For technical support or feature requests, refer to the comprehensive error messages and troubleshooting guide above. The system includes detailed logging and GPU diagnostics to help identify and resolve issues quickly.
