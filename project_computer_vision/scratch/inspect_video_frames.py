import cv2
import sys
import os
from ultralytics import YOLO

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def inspect_video():
    detector = YOLO("models/best.pt")
    cap = cv2.VideoCapture("dataset/videos/test.mp4")
    if not cap.isOpened():
        print("Could not open video file.")
        return
        
    print("--- Inspecting first 50 frames of test.mp4 ---")
    frame_idx = 0
    detections_found = 0
    
    # Load COCO model
    coco = YOLO("yolov8n.pt")
    
    print("--- Scanning entire video for vehicles ---")
    vehicle_frames = []
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        res_coco = coco(frame, verbose=False)[0]
        has_vehicle = False
        for box in res_coco.boxes:
            cls_id = int(box.cls[0].item())
            cls_name = res_coco.names[cls_id]
            if cls_name in ['car', 'truck', 'bus', 'motorcycle']:
                has_vehicle = True
                break
                
        if has_vehicle:
            vehicle_frames.append(frame_idx)
            if len(vehicle_frames) <= 10:
                print(f"Frame {frame_idx}: Found vehicle!")
            
        # Detect plates at very low confidence threshold (0.1)
        results = detector(frame, conf=0.1, verbose=False)[0]
        if len(results.boxes) > 0:
            print(f"Frame {frame_idx}: Found plate with conf {results.boxes[0].conf[0].item():.2f}")
            detections_found += len(results.boxes)
                
        frame_idx += 1
        
    cap.release()
    print(f"Total frames scanned: {frame_idx}")
    print(f"Total frames containing vehicles: {len(vehicle_frames)}")
    print(f"Total plates detected: {detections_found}")

if __name__ == "__main__":
    inspect_video()
