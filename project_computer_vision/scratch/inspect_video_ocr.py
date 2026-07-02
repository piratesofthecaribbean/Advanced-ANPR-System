import cv2
import sys
import os
import easyocr
from ultralytics import YOLO
from src.preprocess import resize_image, perspective_correction
from src.utils import correct_ocr, validate_plate

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def inspect_video_ocr():
    detector = YOLO("models/best.pt")
    reader = easyocr.Reader(['en'], gpu=True)
    
    cap = cv2.VideoCapture("dataset/videos/test.mp4")
    if not cap.isOpened():
        print("Could not open video file.")
        return
        
    target_frames = [58, 117, 128, 157]
    frame_idx = 0
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        if frame_idx in target_frames:
            print(f"\n--- Analyzing Frame {frame_idx} ---")
            
            # Detect
            results = detector(frame, conf=0.15, verbose=False)[0]
            if len(results.boxes) == 0:
                print("No plate detected in this frame at conf=0.15")
            else:
                for idx, box in enumerate(results.boxes):
                    x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                    conf = box.conf[0].item()
                    print(f"Detected Plate box: {x1, y1, x2, y2} | Conf: {conf:.3f}")
                    
                    # Crop
                    crop = frame[y1:y2, x1:x2]
                    cv2.imwrite(f"outputs/video_crop_{frame_idx}_{idx}.jpg", crop)
                    
                    # OCR on raw crop
                    ocr_res = reader.readtext(crop)
                    print(f"Raw OCR results (found {len(ocr_res)} text blocks):")
                    for r in ocr_res:
                        text = r[1]
                        score = r[2]
                        corrected = correct_ocr(text)
                        valid = validate_plate(corrected)
                        print(f"  - '{text}' (conf: {score:.3f}) | Corrected: '{corrected}' | Valid: {valid}")
                        
                    # OCR on deskewed crop
                    deskewed = perspective_correction(crop)
                    cv2.imwrite(f"outputs/video_crop_deskewed_{frame_idx}_{idx}.jpg", deskewed)
                    ocr_res_desk = reader.readtext(deskewed)
                    print(f"Deskewed OCR results (found {len(ocr_res_desk)} text blocks):")
                    for r in ocr_res_desk:
                        text = r[1]
                        score = r[2]
                        corrected = correct_ocr(text)
                        valid = validate_plate(corrected)
                        print(f"  - '{text}' (conf: {score:.3f}) | Corrected: '{corrected}' | Valid: {valid}")
                        
        frame_idx += 1
        if frame_idx > max(target_frames):
            break
            
    cap.release()

if __name__ == "__main__":
    inspect_video_ocr()
