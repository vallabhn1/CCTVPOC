#!/usr/bin/env python3
"""
Test script for suspicious activity detection functionality using a video file
"""

import cv2
import os
from ultralytics import YOLO
import numpy as np

# Suspicious activity classes
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

def test_suspicious_activity_detection():
    """Test the suspicious activity detection model with a video file"""
    print("🔍 Testing Suspicious Activity Detection...")
    
    # Check if model file exists
    model_path = "Suspicious_Activities_nano.pt"
    if not os.path.exists(model_path):
        print(f"❌ Model file not found: {model_path}")
        print("Please download the model from:")
        print("https://huggingface.co/Accurateinfosolution/Suspicious_activity_detection_Yolov11_Custom")
        return False
    
    # Check if video file exists
    video_path = "Gang burgle.mp4"
    if not os.path.exists(video_path):
        print(f"❌ Video file not found: {video_path}")
        return False
    
    try:
        # Load the model
        print("📦 Loading suspicious activity model...")
        model = YOLO(model_path)
        model.conf = 0.6
        model.iou = 0.5
        print("✅ Model loaded successfully")
        
        # Test with the video file
        print(f"🎥 Testing with video: {video_path} (press 'q' to quit)...")
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            print(f"❌ Cannot open video file: {video_path}")
            return False
            
        while True:
            ret, frame = cap.read()
            if not ret:
                print("ℹ️ End of video reached")
                break
                
            # Run detection
            results = model(frame)
            
            # Process results
            suspicious_count = 0
            for r in results:
                boxes = r.boxes
                for box in boxes:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    conf = box.conf[0].cpu().numpy()
                    cls = int(box.cls[0].cpu().numpy())
                    
                    if conf > 0.6 and cls in SUSPICIOUS_CLASSES:
                        suspicious_count += 1
                        label = SUSPICIOUS_CLASSES[cls]
                        
                        # Draw red bounding box
                        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 3)
                        
                        # Draw label
                        label_text = f"{label} {conf:.2f}"
                        cv2.putText(frame, label_text, (int(x1), int(y1) - 10),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                        
                        print(f"🚨 Detected: {label} (Confidence: {conf:.2f})")
            
            # Add status text
            cv2.putText(frame, f"Suspicious Activities: {suspicious_count}", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            cv2.putText(frame, "Press 'q' to quit", 
                       (10, frame.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            cv2.imshow('Suspicious Activity Detection Test', frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        cap.release()
        cv2.destroyAllWindows()
        print("✅ Test completed successfully")
        return True
        
    except Exception as e:
        print(f"❌ Error during testing: {e}")
        return False

def main():
    print("="*60)
    print("   SUSPICIOUS ACTIVITY DETECTION - TEST")
    print("="*60)
    
    success = test_suspicious_activity_detection()
    
    print("\n" + "="*60)
    if success:
        print("🎉 Suspicious activity detection is working!")
        print("\nNext steps:")
        print("1. Run: python mall_surveillance.py")
        print("2. The system will now detect suspicious activities")
        print("3. Red boxes will highlight detected suspicious activities")
    else:
        print("⚠️ Suspicious activity detection test failed")
        print("Please check the model file, video file, and dependencies")
    print("="*60)

if __name__ == "__main__":
    main()