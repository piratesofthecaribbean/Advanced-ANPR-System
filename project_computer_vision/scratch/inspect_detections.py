import cv2
import sys
import os
from ultralytics import YOLO

# Add path resolution
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def inspect():
    # Load COCO model to detect vehicles
    coco = YOLO("yolov8n.pt")
    
    # Load our plate detector
    plate_detector = YOLO("models/best.pt")
    
    # Test images
    img_dir = "/Users/ayushverma/Downloads"
    images = [
        "WhatsApp Image 2025-03-28 at 13.36.23.jpeg",
        "WhatsApp Image 2025-03-28 at 13.36.25.jpeg",
        "newer.jpg"
    ]
    
    for img_name in images:
        path = os.path.join(img_dir, img_name)
        if not os.path.exists(path):
            print(f"File not found: {path}")
            continue
            
        print(f"\n--- Analyzing: {img_name} ---")
        img = cv2.imread(path)
        
        # 1. Check with COCO
        coco_res = coco(img, verbose=False)[0]
        vehicles = []
        for box in coco_res.boxes:
            cls_id = int(box.cls[0].item())
            cls_name = coco_res.names[cls_id]
            conf = box.conf[0].item()
            if cls_name in ['car', 'truck', 'bus', 'motorcycle']:
                vehicles.append(f"{cls_name} ({conf:.2f})")
        print(f"Vehicles found: {', '.join(vehicles) if vehicles else 'None'}")
        
        # 2. Check plate detector with low thresholds
        for th in [0.5, 0.2, 0.05, 0.01]:
            plate_res = plate_detector(img, conf=th, verbose=False)[0]
            boxes = len(plate_res.boxes)
            if boxes > 0:
                confs = [f"{b.conf[0].item():.3f}" for b in plate_res.boxes]
                print(f"Plate detector at conf={th}: Found {boxes} plate(s) with conf: {', '.join(confs)}")
                break
        else:
            print("Plate detector: Found 0 plates at all thresholds down to 0.01")

if __name__ == "__main__":
    inspect()
