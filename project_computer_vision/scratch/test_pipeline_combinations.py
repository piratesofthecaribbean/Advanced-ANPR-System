import cv2
import sys
import os
from ultralytics import YOLO
from src.preprocess import resize_image, enhance_contrast, remove_noise

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def test_combinations():
    detector = YOLO("models/best.pt")
    img_path = "/Users/ayushverma/Downloads/WhatsApp Image 2025-03-28 at 13.36.25.jpeg"
    img = cv2.imread(img_path)
    
    print("--- Testing combinations on WhatsApp Image 2025-03-28 at 13.36.25.jpeg ---")
    
    # Raw Image
    res_raw = detector(img, conf=0.1, verbose=False)[0]
    print(f"Raw image: Found {len(res_raw.boxes)} plate(s)")
    
    # Resized to 1280
    resized = resize_image(img, 1280)
    res_res = detector(resized, conf=0.1, verbose=False)[0]
    print(f"Resized only: Found {len(res_res.boxes)} plate(s)")
    
    # Resized + CLAHE
    enhanced = enhance_contrast(resized)
    res_enh = detector(enhanced, conf=0.1, verbose=False)[0]
    print(f"Resized + CLAHE: Found {len(res_enh.boxes)} plate(s)")
    
    # Resized + Bilateral
    denoised = remove_noise(resized)
    res_den = detector(denoised, conf=0.1, verbose=False)[0]
    print(f"Resized + Bilateral: Found {len(res_den.boxes)} plate(s)")
    
    # Resized + CLAHE + Bilateral
    both = remove_noise(enhanced)
    res_both = detector(both, conf=0.1, verbose=False)[0]
    print(f"Resized + CLAHE + Bilateral: Found {len(res_both.boxes)} plate(s)")

if __name__ == "__main__":
    test_combinations()
