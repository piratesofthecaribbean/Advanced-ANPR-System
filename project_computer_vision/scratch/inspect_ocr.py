import cv2
import sys
import os
import easyocr
from ultralytics import YOLO
from src.preprocess import resize_image, enhance_contrast, remove_noise, perspective_correction
from src.utils import correct_ocr, validate_plate

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def inspect_ocr():
    # Load model
    detector = YOLO("models/best.pt")
    reader = easyocr.Reader(['en'], gpu=True)
    
    img_path = "/Users/ayushverma/Downloads/WhatsApp Image 2025-03-28 at 13.36.25.jpeg"
    img = cv2.imread(img_path)
    resized = resize_image(img, 1280)
    
    # Get bounding boxes
    results = detector(resized, conf=0.2, verbose=False)[0]
    if len(results.boxes) == 0:
        print("No plates detected!")
        return
        
    for idx, box in enumerate(results.boxes):
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
        print(f"\n--- Plate {idx+1} Bounding Box: {x1, y1, x2, y2} ---")
        
        # Crop plate
        crop = resized[y1:y2, x1:x2]
        cv2.imwrite("outputs/crop_raw.jpg", crop)
        
        # 1. OCR on Raw crop
        res_raw = reader.readtext(crop)
        print("OCR on Raw crop:")
        for r in res_raw:
            print(f"  Raw: '{r[1]}' | Conf: {r[2]:.3f} | Corrected: '{correct_ocr(r[1])}' | Valid Indian: {validate_plate(correct_ocr(r[1]))}")
            
        # 2. OCR on Grayscale crop
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        res_gray = reader.readtext(gray)
        print("OCR on Grayscale crop:")
        for r in res_gray:
            print(f"  Raw: '{r[1]}' | Conf: {r[2]:.3f} | Corrected: '{correct_ocr(r[1])}'")
            
        # 3. OCR on CLAHE enhanced crop
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)
        res_enh = reader.readtext(enhanced)
        print("OCR on CLAHE crop:")
        for r in res_enh:
            print(f"  Raw: '{r[1]}' | Conf: {r[2]:.3f} | Corrected: '{correct_ocr(r[1])}'")
            
        # 4. OCR on Deskewed crop
        deskewed = perspective_correction(crop)
        cv2.imwrite("outputs/crop_deskewed.jpg", deskewed)
        res_desk = reader.readtext(deskewed)
        print("OCR on Deskewed crop:")
        for r in res_desk:
            print(f"  Raw: '{r[1]}' | Conf: {r[2]:.3f} | Corrected: '{correct_ocr(r[1])}' | Valid Indian: {validate_plate(correct_ocr(r[1]))}")

if __name__ == "__main__":
    inspect_ocr()
