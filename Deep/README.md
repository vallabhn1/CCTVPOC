# Mall CCTV Analytics System using YOLOv7

A comprehensive Python application for analyzing mall visitor behavior using CCTV footage and YOLOv7 object detection.

## Features

### 📊 Analytics Capabilities
- **Unique Visitor Counting**: Track and count individual visitors throughout the mall
- **Shopping Behavior Analysis**: Determine who went shopping vs. who left empty-handed
- **Area Popularity Tracking**: Identify which mall areas get the most attention
- **Ignored Area Detection**: Find areas that are being overlooked by visitors
- **Crowd Heatmap Generation**: Visual representation of high-traffic areas
- **Dwell Time Analysis**: Measure how long visitors spend in each area

### 🔧 Technical Features
- **GPU Acceleration**: Optimized to run on GPU for fast processing
- **Real-time Processing**: Process live CCTV feeds or recorded videos
- **Advanced Tracking**: Multi-object tracking with ID persistence
- **Comprehensive Reporting**: Detailed JSON reports and visualizations

## Installation

### 1. Prerequisites
- Python 3.8 or higher
- CUDA-compatible GPU (recommended)
- YOLOv7 repository cloned at: `C:\Users\deepv\OneDrive\Desktop\Final Year Project\yolov7`

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Download YOLOv7 Weights
Download the YOLOv7 weights file and place it in the YOLOv7 directory:
```bash
# Navigate to YOLOv7 directory
cd "C:\Users\deepv\OneDrive\Desktop\Final Year Project\yolov7"

# Download weights (if not already available)
# You can download from: https://github.com/WongKinYiu/yolov7/releases/download/v0.1/yolov7.pt
```

## Usage

### Basic Usage
Run the analytics system with the default video:
```bash
python cctv.py
```

### Advanced Usage
```bash
python cctv.py --video "path/to/your/video.mp4" --weights "path/to/yolov7.pt" --output "analyzed_video.mp4"
```

### Command Line Arguments
- `--video`: Path to input video file (default: included sample video)
- `--weights`: Path to YOLOv7 weights file (default: auto-detect)
- `--output`: Path for output video with annotations (default: 'output_analytics.mp4')
- `--conf-thres`: Confidence threshold for detection (default: 0.25)
- `--iou-thres`: IoU threshold for Non-Maximum Suppression (default: 0.45)
- `--no-display`: Don't show video during processing (for headless operation)

## Configuration

### Mall Area Configuration
Edit `config.json` to customize mall areas for your specific layout:

```json
{
  "mall_areas": {
    "entrance": {"x": 0, "y": 0, "w": 200, "h": 480},
    "clothing_store": {"x": 200, "y": 0, "w": 200, "h": 240},
    "electronics": {"x": 400, "y": 0, "w": 200, "h": 240}
  }
}
```

### Analytics Parameters
Adjust detection and tracking parameters in `config.json`:

```json
{
  "analytics_parameters": {
    "min_areas_for_shopping": 2,
    "min_dwell_time_seconds": 30.0,
    "confidence_threshold": 0.25
  }
}
```

## Output Files

The system generates several output files:

1. **mall_crowd_heatmap.png**: Visual heatmap showing crowd density
2. **mall_analytics_report.json**: Detailed analytics data in JSON format
3. **output_analytics.mp4**: Annotated video with tracking overlays
4. **Console Output**: Real-time analytics summary

## Analytics Metrics

### Visitor Statistics
- Total unique visitors detected
- Number of visitors who went shopping
- Number of visitors who left empty-handed
- Overall shopping conversion rate

### Area Analysis
- Visit counts per area
- Percentage distribution of foot traffic
- Average dwell time per area
- Identification of ignored/low-traffic areas

### Behavioral Insights
- Shopping patterns and preferences
- Peak activity areas
- Traffic flow analysis
- Time-based visitor trends

## Understanding the Results

### Shopping Behavior Classification
A visitor is classified as "shopping" if they:
- Visit at least 2 different areas (configurable)
- Spend significant time in at least one area (>30 seconds)
- Have a total dwell time of more than 1 minute

### Heatmap Interpretation
- **Red/Hot areas**: High crowd density
- **Blue/Cool areas**: Low foot traffic
- **Grid-based**: Each cell represents a 20x20 pixel area

### Area Popularity
Areas are ranked by:
- Total number of visits
- Percentage of overall foot traffic
- Average dwell time per visitor

## GPU Requirements

For optimal performance, ensure:
- CUDA-compatible GPU with at least 4GB VRAM
- Latest NVIDIA drivers installed
- PyTorch with CUDA support

To check GPU availability:
```python
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"GPU device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None'}")
```

## Troubleshooting

### Common Issues

1. **YOLOv7 Import Error**: Ensure YOLOv7 path is correct and all dependencies are installed
2. **GPU Not Detected**: Install CUDA-compatible PyTorch version
3. **Video File Error**: Check video file path and format compatibility
4. **Low Detection Accuracy**: Adjust confidence threshold or use better quality video

### Performance Optimization

- Use GPU acceleration for faster processing
- Reduce video resolution for real-time processing
- Adjust confidence thresholds based on your video quality
- Process videos in batches for large datasets

## Customization

### Adding New Mall Areas
1. Edit the `mall_areas` dictionary in the code or config file
2. Define rectangular areas with x, y, width, height coordinates
3. Use descriptive names for better analytics reporting

### Modifying Detection Parameters
- Adjust confidence and IoU thresholds for your specific use case
- Modify tracking distance thresholds based on camera placement
- Customize shopping behavior criteria

## Future Enhancements

Potential improvements and features:
- Integration with live CCTV streams
- Advanced behavior analysis (loitering detection, group analysis)
- Database integration for long-term analytics
- Web dashboard for real-time monitoring
- Alert system for unusual patterns

## Support

For issues or questions:
1. Check the troubleshooting section
2. Verify all dependencies are correctly installed
3. Ensure YOLOv7 setup is complete
4. Check video file compatibility

## License

This project is for educational and research purposes. Ensure compliance with local privacy laws when using CCTV footage.
