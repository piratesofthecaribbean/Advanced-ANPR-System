import os
import sys
import time
import argparse

# Ensure project root is in the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.preprocess import resize_image, enhance_contrast, remove_noise, perspective_correction
from src.detector import LicensePlateDetector
from src.ocr import LicensePlateOCR
from src.utils import validate_plate, correct_ocr, crop_plate, draw_boxes, save_results, log_event
from src.main import process_image

IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}

def collect_images(dataset_dir):
    images = []
    for root, _, files in os.walk(dataset_dir):
        for f in files:
            if os.path.splitext(f.lower())[1] in IMAGE_EXTENSIONS:
                images.append(os.path.join(root, f))
    return sorted(images)

def main():
    parser = argparse.ArgumentParser(description="Batch ANPR pipeline over entire dataset")
    parser.add_argument("--dataset", default="dataset/images", help="Root folder of images")
    parser.add_argument("--yolo-conf", type=float, default=0.5)
    parser.add_argument("--ocr-conf",  type=float, default=0.6)
    parser.add_argument("--no-deskew",  action="store_true")
    parser.add_argument("--no-enhance", action="store_true")
    parser.add_argument("--no-denoise", action="store_true")
    parser.add_argument("--limit", type=int, default=0, help="Process only first N images (0 = all)")
    args = parser.parse_args()

    os.makedirs("outputs", exist_ok=True)

    images = collect_images(args.dataset)
    if args.limit:
        images = images[:args.limit]

    total = len(images)
    log_event("info", f"Batch run started: {total} images found in '{args.dataset}'")

    # Load models once
    log_event("info", "Loading YOLO detector...")
    detector = LicensePlateDetector(model_path="models/best.pt")
    log_event("info", "Loading EasyOCR...")
    ocr = LicensePlateOCR()
    log_event("info", "Models loaded. Starting batch processing...")

    batch_start = time.time()
    success = 0
    failed  = 0

    for idx, img_path in enumerate(images, 1):
        try:
            process_image(img_path, detector, ocr, args)
            success += 1
        except Exception as e:
            log_event("error", f"[{idx}/{total}] Failed on {os.path.basename(img_path)}: {e}")
            failed += 1

        if idx % 50 == 0 or idx == total:
            elapsed = time.time() - batch_start
            rate = idx / elapsed
            eta  = (total - idx) / rate if rate > 0 else 0
            log_event("info",
                f"Progress: {idx}/{total} ({idx/total*100:.1f}%) | "
                f"✓ {success}  ✗ {failed} | "
                f"Rate: {rate:.1f} img/s | ETA: {eta/60:.1f} min")

    total_time = time.time() - batch_start
    log_event("info",
        f"Batch complete: {success} succeeded, {failed} failed, "
        f"total time: {total_time/60:.1f} min. "
        f"Results saved in outputs/results.csv and outputs/")

if __name__ == "__main__":
    main()
