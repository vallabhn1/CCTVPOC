import cv2
import torch
import json
import numpy as np
from ultralytics import YOLO
from mall_analytics import MallAnalytics
import time
from pathlib import Path
import torchvision
from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2
from torchvision.transforms import functional as F
import os
import gdown

# Suspicious activity classes from the YOLOv11 model
SUSPICIOUS_CLASSES = {
    0: 'Assault',
    1: 'Fighting', 
    2: 'Gun',
    3: 'Kidnapping',
    4: 'Knife',
    5: 'People',
    6: 'Police',
    7: 'Prisoner',
    8: 'Theft/Robbery',
    9: 'Time Bomb'
}

def download_suspicious_activity_model():
    """Download the suspicious activity detection model from Hugging Face"""
    model_url = "https://huggingface.co/Accurateinfosolution/Suspicious_activity_detection_Yolov11_Custom/resolve/main/Suspicious_Activities_nano.pt"
    model_path = "Suspicious_Activities_nano.pt"
    
    if not os.path.exists(model_path):
        print("📥 Downloading suspicious activity detection model...")
        try:
            # Use gdown or direct download
            import urllib.request
            urllib.request.urlretrieve(model_url, model_path)
            print("✅ Suspicious activity model downloaded successfully")
        except Exception as e:
            print(f"⚠️ Failed to download suspicious activity model: {e}")
            print("Please manually download from: https://huggingface.co/Accurateinfosolution/Suspicious_activity_detection_Yolov11_Custom")
            return None
    
    return model_path

# Initialize models with OpenCV's built-in models
def initialize_detection_models():
    print("Initializing detection models...")
    
    # Use OpenCV's pre-trained face detection models
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    profile_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_profileface.xml')
    body_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_upperbody.xml')
    
    return face_cascade, profile_cascade, body_cascade

def detect_gender(frame, bbox, face_cascade, profile_cascade, body_cascade):
    """Enhanced gender detection using multiple features"""
    x1, y1, x2, y2 = [int(b) for b in bbox[:4]]
    person_roi = frame[y1:y2, x1:x2]
    if person_roi.size == 0:
        return None, 0.0
        
    # Convert to grayscale for detection
    gray_roi = cv2.cvtColor(person_roi, cv2.COLOR_BGR2GRAY)
    
    # Detect faces (both frontal and profile)
    frontal_faces = face_cascade.detectMultiScale(gray_roi, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    profile_faces = profile_cascade.detectMultiScale(gray_roi, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    upper_body = body_cascade.detectMultiScale(gray_roi, scaleFactor=1.1, minNeighbors=5, minSize=(50, 50))
    
    # Calculate body proportions
    height = y2 - y1
    width = x2 - x1
    body_ratio = height / width if width > 0 else 0
    shoulder_width_ratio = width / height if height > 0 else 0
    
    # Initialize feature scores
    male_score = 0.0
    total_features = 0
    
    # Face shape analysis (if face detected)
    if len(frontal_faces) > 0 or len(profile_faces) > 0:
        faces = frontal_faces if len(frontal_faces) > 0 else profile_faces
        fx, fy, fw, fh = faces[0]
        face_ratio = fw / fh
        
        # Face shape features
        if face_ratio < 0.85:  # More rectangular face (typical male characteristic)
            male_score += 1.0
        else:  # More rounded face (typical female characteristic)
            male_score += 0.0
        total_features += 1
    
    # Body proportion analysis
    if len(upper_body) > 0:
        # Shoulder width relative to height (males typically have broader shoulders)
        if shoulder_width_ratio > 0.4:
            male_score += 1.0
        else:
            male_score += 0.0
        total_features += 1
    
    # Overall body ratio (height to width ratio)
    if body_ratio > 2.5:  # Typically more feminine proportion
        male_score += 0.0
    else:  # Typically more masculine proportion
        male_score += 1.0
    total_features += 1
    
    # Calculate final confidence
    if total_features == 0:
        return None, 0.0
        
    confidence = male_score / total_features
    gender = 'Male' if confidence > 0.5 else 'Female'
    gender_confidence = max(confidence, 1 - confidence)  # Convert to 0-1 scale
    
    return gender, gender_confidence

def load_config(config_path: str = "config.json") -> dict:
    """Load configuration from JSON file"""
    with open(config_path, 'r') as f:
        return json.load(f)

def main():
    # Load configuration
    config = load_config()
    
    # Initialize analytics
    analytics = MallAnalytics()
    
    # Initialize YOLOv8 model for general detection with optimized settings
    print("Loading YOLOv8 model...")
    yolo_model = YOLO("yolov8x.pt")
    # Configure YOLO parameters for better performance
    yolo_model.conf = 0.3  # Lower confidence threshold
    yolo_model.iou = 0.45  # Lower IoU threshold for NMS
    yolo_model.max_det = 50  # Limit maximum detections per frame
    yolo_model.classes = [0, 26]  # Only detect persons (0) and handbags (26)
    
    # Initialize suspicious activity detection model
    suspicious_model = None
    suspicious_model_path = download_suspicious_activity_model()
    if suspicious_model_path and os.path.exists(suspicious_model_path):
        print("Loading suspicious activity detection model...")
        try:
            suspicious_model = YOLO(suspicious_model_path)
            suspicious_model.conf = 0.6  # Higher confidence for suspicious activities
            suspicious_model.iou = 0.5
            print("✅ Suspicious activity detection model loaded successfully")
        except Exception as e:
            print(f"⚠️ Failed to load suspicious activity model: {e}")
            suspicious_model = None
    else:
        print("⚠️ Suspicious activity detection disabled - model not available")
    
    # Initialize face and gender detection models
    face_cascade, profile_cascade, body_cascade = initialize_detection_models()
    
    # Gender detection labels
    gender_list = ['Male', 'Female']
    
    # Initialize Faster R-CNN model for precise person detection
    print("Loading Faster R-CNN model...")
    weights = torchvision.models.detection.FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT
    frcnn_model = fasterrcnn_resnet50_fpn_v2(weights=weights)
    frcnn_model.to(analytics.device)
    frcnn_model.eval()
    
    # Open video capture
    video_path = "Small video Mall shopping mall video shopping mall CCTV camera video no copyright full HD 4K video.mp4"
    # video_path = "Gang burgle.mp4"

    if not Path(video_path).exists():
        print(f"Error: Video file not found: {video_path}")
        return
    
    cap = cv2.VideoCapture(video_path)
    analytics.fps = cap.get(cv2.CAP_PROP_FPS)
    
    # Initialize video writer if enabled
    if config['output_settings']['save_video']:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter('output.mp4', fourcc, analytics.fps,
                            (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                             int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))))
    
    print("Starting video processing...")
    frame_count = 0
    start_time = time.time()
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        # Resize frame for faster processing while maintaining aspect ratio
        height, width = frame.shape[:2]
        target_width = 640  # Standard YOLO input size
        scale = target_width / width
        target_height = int(height * scale)
        
        if width > target_width:
            process_frame = cv2.resize(frame, (target_width, target_height))
        else:
            process_frame = frame
        
        # Run YOLOv8 inference with optimized frame
        yolo_results = yolo_model(process_frame)  # Detect persons and bags
        
        # Process YOLOv8 detections
        detections = []
        all_detections = []  # Including bags for unattended object detection
        
        # Convert frame to tensor for Faster R-CNN
        frame_tensor = F.to_tensor(frame).unsqueeze(0).to(analytics.device)
        
        # Run Faster R-CNN inference for precise person detection
        with torch.no_grad():
            frcnn_results = frcnn_model(frame_tensor)[0]
        
        # Process Faster R-CNN person detections (class 1 in COCO)
        frcnn_boxes = frcnn_results['boxes'][frcnn_results['labels'] == 1]
        frcnn_scores = frcnn_results['scores'][frcnn_results['labels'] == 1]
        
        # Add high-confidence Faster R-CNN person detections
        for box, score in zip(frcnn_boxes, frcnn_scores):
            if score > config['model_settings']['person_detection_confidence']:
                x1, y1, x2, y2 = box.cpu().numpy()
                detections.append([x1, y1, x2, y2, score.cpu().numpy(), 0])  # 0 for person
        
        # Process YOLOv8 detections for bags
        for r in yolo_results:
            boxes = r.boxes
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                conf = box.conf[0].cpu().numpy()
                cls = int(box.cls[0].cpu().numpy())
                
                # Scale coordinates back to original frame size if resized
                if width > target_width:
                    x1, x2 = x1/scale, x2/scale
                    y1, y2 = y1/scale, y2/scale
                
                # Only process bag detections from YOLOv8
                if conf > config['model_settings']['person_detection_confidence']:
                    if cls == 26:  # handbag in COCO
                        all_detections.append([x1, y1, x2, y2, conf, 1])  # 1 for bag
        
        # Combine all detections
        all_detections.extend(detections)  # Add Faster R-CNN person detections
        
        # Run suspicious activity detection if model is available
        suspicious_detections = []
        if suspicious_model is not None:
            try:
                suspicious_results = suspicious_model(process_frame)
                for r in suspicious_results:
                    boxes = r.boxes
                    for box in boxes:
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        conf = box.conf[0].cpu().numpy()
                        cls = int(box.cls[0].cpu().numpy())
                        
                        # Scale coordinates back to original frame size if resized
                        if width > target_width:
                            x1, x2 = x1/scale, x2/scale
                            y1, y2 = y1/scale, y2/scale
                        
                        # Only include high-confidence suspicious activity detections
                        if conf > 0.6 and cls in SUSPICIOUS_CLASSES:
                            suspicious_detections.append({
                                'bbox': [x1, y1, x2, y2],
                                'confidence': conf,
                                'class': cls,
                                'label': SUSPICIOUS_CLASSES[cls]
                            })
                            # Log suspicious activity alert
                            print(f"🚨 SUSPICIOUS ACTIVITY DETECTED: {SUSPICIOUS_CLASSES[cls]} (Confidence: {conf:.2f}) at {time.strftime('%H:%M:%S')}")
            except Exception as e:
                print(f"Error in suspicious activity detection: {e}")
        
        # Track people and update analytics with more accurate Faster R-CNN detections
        tracked_detections = analytics.simple_tracker(detections, frame_count / analytics.fps)
        analytics.update_heatmap(tracked_detections)
        
        # Update area information for each tracked person
        for track_id, bbox, area in tracked_detections:
            if track_id in analytics.people:
                person = analytics.people[track_id]
                # Get center point of person
                x_center = (bbox[0] + bbox[2]) / 2
                y_center = (bbox[1] + bbox[3]) / 2
                
                # Determine which area the person is in
                current_area = analytics.get_area_from_position(x_center, y_center)
                person.current_area = current_area
        
        # Detect unattended objects
        all_detections.extend(detections)  # Add person detections
        analytics.detect_unattended_objects(frame, all_detections)
        
        # Perform improved gender classification using enhanced detection
        for track_id, bbox, area in tracked_detections:
            if track_id in analytics.people:
                person = analytics.people[track_id]
                if not hasattr(person, 'gender') or not person.gender:
                    # Use the enhanced gender detection function
                    gender, gender_confidence = detect_gender(frame, bbox, face_cascade, profile_cascade, body_cascade)
                    
                    if gender and gender_confidence > config['model_settings']['gender_classification_confidence']:
                        person.gender = gender.lower()
                        if gender == 'Male' and not hasattr(person, 'counted_gender'):
                            analytics.male_visitors += 1
                            person.counted_gender = True
                        elif gender == 'Female' and not hasattr(person, 'counted_gender'):
                            analytics.female_visitors += 1
                            person.counted_gender = True
                        
                        # Draw ID label if bounding boxes are enabled
                        if config['display_settings']['show_bounding_boxes']:
                            x1, y1, x2, y2 = [int(b) for b in bbox[:4]]
                            # Use same color scheme as bounding boxes
                            label_color = (255, 0, 0) if gender == 'Male' else (255, 192, 203)  # Blue for male, Pink for female
                            # Show only ID number
                            label = f"ID{track_id}"
                            cv2.putText(frame, label, (x1, y1-10),
                                      cv2.FONT_HERSHEY_SIMPLEX, 0.5, label_color, 1)
        
        # Draw visualizations if enabled
        if config['display_settings']['show_bounding_boxes']:
            frame = analytics.draw_visualizations(frame, tracked_detections)
            
            # Draw suspicious activity detections with red boxes
            for detection in suspicious_detections:
                x1, y1, x2, y2 = [int(coord) for coord in detection['bbox']]
                label = detection['label']
                confidence = detection['confidence']
                
                # Draw red bounding box for suspicious activities
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 3)  # Red box, thicker line
                
                # Create label with confidence
                label_text = f"{label} {confidence:.2f}"
                
                # Get text size for background
                (text_width, text_height), baseline = cv2.getTextSize(
                    label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
                )
                
                # Draw red background for label
                cv2.rectangle(frame, (x1, y1 - text_height - 10), 
                             (x1 + text_width + 10, y1), (0, 0, 255), -1)
                
                # Draw white text on red background
                cv2.putText(frame, label_text, (x1 + 5, y1 - 5),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                
                # Add alert icon/symbol
                cv2.putText(frame, "⚠️", (x1 - 25, y1 + 20),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            
            # Draw subtle area boundaries (optional)
            if config.get('display_settings', {}).get('show_areas', True):
                for area_name, area_info in analytics.mall_areas.items():
                    x, y, w, h = area_info['x'], area_info['y'], area_info['w'], area_info['h']
                    # Draw very subtle area rectangles
                    area_color = {
                        'entrance': (0, 100, 0),      # Dark Green
                        'food_court': (100, 80, 0),   # Dark Orange
                        'clothing_store': (100, 0, 100), # Dark Magenta
                        'electronics': (0, 100, 100)   # Dark Cyan
                    }.get(area_name, (64, 64, 64))
                    
                    cv2.rectangle(frame, (x, y), (x + w, y + h), area_color, 1)
                    # Smaller area labels
                    cv2.putText(frame, area_name.replace('_', ' ')[:8], 
                               (x + 5, y + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.35, area_color, 1)
            
            # Create a more compact semi-transparent overlay for stats
            overlay = frame.copy()
            cv2.rectangle(overlay, (5, 5), (300, 160), (0, 0, 0), -1)
            cv2.addWeighted(overlay, 0.4, frame, 0.6, 0, frame)
            
            # Add compact overlay text
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(frame, f"Total: {len(analytics.people)}", 
                       (10, 25), font, 0.5, (255, 255, 255), 1)
            cv2.putText(frame, f"M: {analytics.male_visitors} | F: {analytics.female_visitors}", 
                       (10, 45), font, 0.5, (255, 255, 255), 1)
            cv2.putText(frame, f"Bags: {len(analytics.unattended_bags)} | Kids: {len(analytics.unattended_children)}", 
                       (10, 65), font, 0.4, (255, 255, 0), 1)
            
            # Add suspicious activity count
            if len(suspicious_detections) > 0:
                cv2.putText(frame, f"🚨 SUSPICIOUS: {len(suspicious_detections)}", 
                           (10, 85), font, 0.5, (0, 0, 255), 2)
            
            # Add area-specific visitor counts
            cv2.putText(frame, "Areas:", (10, 110), font, 0.4, (150, 255, 150), 1)
            area_text = f"Entrance: {len([p for p in analytics.people.values() if hasattr(p, 'current_area') and p.current_area == 'entrance'])}"
            cv2.putText(frame, area_text, (10, 130), font, 0.35, (200, 200, 255), 1)
            area_text2 = f"Food: {len([p for p in analytics.people.values() if hasattr(p, 'current_area') and p.current_area == 'food_court'])}"
            cv2.putText(frame, area_text2, (140, 130), font, 0.35, (200, 200, 255), 1)
            area_text3 = f"Shop: {len([p for p in analytics.people.values() if hasattr(p, 'current_area') and p.current_area in ['clothing_store', 'electronics']])}"
            cv2.putText(frame, area_text3, (10, 150), font, 0.35, (200, 200, 255), 1)
        
        # Save video if enabled
        if config['output_settings']['save_video']:
            out.write(frame)
        
        # Display frame
        cv2.imshow('Mall CCTV Analytics', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        
        frame_count += 1
        
        # Print progress
        if frame_count % 30 == 0:  # Update every 30 frames
            elapsed = time.time() - start_time
            fps = frame_count / elapsed
            print(f"\rProcessing: {frame_count} frames, {fps:.1f} FPS", end='')
    
    # Final analytics
    print("\n\nGenerating final analytics...")
    analytics.analyze_shopping_behavior()
    analytics.analyze_area_popularity()
    analytics.print_analytics_summary()
    
    # Cleanup
    cap.release()
    if config['output_settings']['save_video']:
        out.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
