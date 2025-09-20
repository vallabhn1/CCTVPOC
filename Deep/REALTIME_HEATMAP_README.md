# Real-time Traffic Heatmap System

This script provides a **side-by-side view** of the original CCTV video with a **live updating heatmap** that shows traffic patterns and person movement in real-time.

## Features

🎥 **Side-by-Side Display**: Original video with detection boxes alongside real-time heatmap
🔥 **Live Heatmap**: Dynamic heat visualization that updates with each frame
🎯 **Person Detection**: Uses YOLOv7 (if available) or Faster R-CNN fallback
⚡ **GPU Acceleration**: Automatically uses CUDA if available
🎛️ **Customizable**: Adjustable grid size, decay rate, and thresholds
💾 **Save Output**: Option to save the side-by-side video

## How It Works

1. **Detection**: Detects people in each video frame
2. **Heatmap Update**: Adds detection locations to a grid-based heatmap
3. **Decay**: Previous heat values gradually fade over time
4. **Visualization**: Shows original video + detections on left, heatmap on right
5. **Real-time**: Updates continuously as video plays

## Usage

### Basic Usage (Interactive)
```bash
python realtime_heatmap.py
```

### Save Output Video
```bash
python realtime_heatmap.py --output realtime_heatmap_output.mp4
```

### Custom Settings
```bash
python realtime_heatmap.py --grid-size 20 --decay 0.95 --conf-thres 0.3
```

### Use Custom Video
```bash
python realtime_heatmap.py --video "path/to/your/video.mp4"
```

## Command Line Options

| Option | Description | Default |
|--------|-------------|---------|
| `--video` | Path to input video file | Auto-detects mall video |
| `--weights` | Path to YOLOv7 weights (optional) | None (uses fallback) |
| `--output` | Save side-by-side output video | None |
| `--conf-thres` | Detection confidence threshold | 0.25 |
| `--iou-thres` | IoU threshold for NMS | 0.45 |
| `--grid-size` | Heatmap grid cell size (pixels) | 30 |
| `--decay` | Heat decay factor (0.95-0.999) | 0.99 |
| `--no-display` | Process without showing video | False |

## Interactive Controls

- **'q'**: Quit the application
- **'r'**: Reset heatmap (clear all accumulated heat)

## Heatmap Interpretation

- **White/Yellow**: High traffic areas (hot spots)
- **Red/Orange**: Medium traffic areas
- **Black/Dark**: Low traffic areas (cold spots)
- **Real-time**: Heat accumulates as people move through areas
- **Decay**: Old heat gradually fades, showing recent activity

## Grid Size Effects

- **Smaller grid (10-20px)**: More detailed heatmap, higher resolution
- **Larger grid (30-50px)**: Smoother heatmap, better for general patterns
- **Default (30px)**: Good balance for most videos

## Decay Rate Effects

- **High decay (0.95)**: Fast fade, shows very recent activity
- **Medium decay (0.98)**: Balanced, shows recent trends
- **Low decay (0.999)**: Slow fade, accumulates long-term patterns

## Performance Tips

1. **GPU**: Enable CUDA for better performance
2. **Grid Size**: Larger grids process faster
3. **No Display**: Use `--no-display` for fastest processing
4. **Resolution**: Lower resolution videos process faster

## Output Files

When using `--output`, the system creates:
- Side-by-side video with original + heatmap
- Left side: Original video with detection boxes
- Right side: Real-time heatmap visualization

## Troubleshooting

### Common Issues

1. **Video not found**: Check video path is correct
2. **Slow processing**: Try larger `--grid-size` or enable GPU
3. **No detections**: Lower `--conf-thres` value
4. **Memory issues**: Use `--no-display` mode

### Performance Optimization

```bash
# Fast processing (no display)
python realtime_heatmap.py --no-display --grid-size 40

# High quality (slower)
python realtime_heatmap.py --grid-size 15 --conf-thres 0.2

# Balanced settings
python realtime_heatmap.py --grid-size 25 --decay 0.98
```

## Example Commands

```bash
# Basic real-time viewing
python realtime_heatmap.py

# Save output with custom settings
python realtime_heatmap.py --output my_heatmap.mp4 --grid-size 25

# High sensitivity detection
python realtime_heatmap.py --conf-thres 0.15 --grid-size 20

# Fast processing mode
python realtime_heatmap.py --no-display --output fast_output.mp4
```

## Technical Details

- **Detection Models**: YOLOv7 (preferred) or Faster R-CNN (fallback)
- **Heatmap Algorithm**: Grid-based accumulation with exponential decay
- **Color Mapping**: Matplotlib 'hot' colormap (black → red → yellow → white)
- **Video Processing**: OpenCV for video I/O and display
- **Device Support**: CUDA GPU (preferred) or CPU fallback

## Requirements

See `requirements.txt` for full dependency list. Key packages:
- torch, torchvision (PyTorch)
- opencv-python (OpenCV)
- matplotlib (visualization)
- numpy (numerical operations)

---

**Note**: The heatmap shows accumulated person movement over time. Areas with more people movement appear "hotter" (whiter/yellower) while areas with less movement appear "cooler" (darker/redder).
