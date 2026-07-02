import os
import cv2
import time
import argparse
import sys

# Ensure project root is in the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.preprocess import resize_image, enhance_contrast, remove_noise, perspective_correction
from src.detector import LicensePlateDetector
from src.ocr import LicensePlateOCR
from src.utils import validate_plate, correct_ocr, crop_plate, draw_boxes, save_results, log_event

def process_image(img_path, detector, ocr, args):
    """
    Processes a single image through the ANPR pipeline.
    """
    image_name = os.path.basename(img_path)
    log_event("info", f"Loading image: {image_name}")
    
    img = cv2.imread(img_path)
    if img is None:
        log_event("error", f"Could not read image: {img_path}")
        return
        
    start_time = time.time()
    
    # 1. Resize input image
    processed_img = resize_image(img, target_width=1280)
    
    # 2. Optional Contrast Enhancement
    if not args.no_enhance:
        processed_img = enhance_contrast(processed_img)
        
    # 3. Optional Noise Removal
    if not args.no_denoise:
        processed_img = remove_noise(processed_img)
        
    # 4. YOLOv9 Detection
    det_start = time.time()
    detections = detector.detect(processed_img, conf_threshold=args.yolo_conf)
    det_time = (time.time() - det_start) * 1000
    log_event("info", f"YOLO Detection Time: {det_time:.1f} ms | Found {len(detections)} plate(s)")
    
    annotated_img = processed_img.copy()
    
    for idx, det in enumerate(detections):
        x1, y1, x2, y2, det_conf = det
        
        # Crop plate
        plate = crop_plate(processed_img, [x1, y1, x2, y2])
        
        # 5. Optional Perspective Correction
        if not args.no_deskew:
            plate = perspective_correction(plate)
            
        # 6. OCR Text Extraction
        ocr_start = time.time()
        ocr_results = ocr.extract_text(plate)
        ocr_time = (time.time() - ocr_start) * 1000
        
        best_text = ""
        best_score = 0.0
        is_valid = False
        
        # Parse OCR results and find the best candidate
        for _, text, score in ocr_results:
            # Clean and correct text
            corrected = correct_ocr(text)
            valid = validate_plate(corrected)
            
            # Prioritize valid Indian plates, or fallback to highest confidence text
            if valid:
                best_text = corrected
                best_score = score
                is_valid = True
                break
            elif score > best_score:
                best_text = corrected
                best_score = score
                is_valid = False
                
        # Check confidence threshold
        if best_score >= args.ocr_conf and best_text:
            log_event("info", f"Plate {idx+1}: '{best_text}' | Conf: {best_score:.2f} | Valid Indian: {is_valid} | OCR Time: {ocr_time:.1f} ms")
            
            # Save CSV & Logging
            save_results(image_name, best_text, best_score)
            
            # Draw annotation
            annotated_img = draw_boxes(annotated_img, [x1, y1, x2, y2], best_text, best_score, is_valid)
        else:
            log_event("warning", f"Plate {idx+1} rejected (Conf: {best_score:.2f} < {args.ocr_conf} or no text detected)")
            annotated_img = draw_boxes(annotated_img, [x1, y1, x2, y2], "REJECTED", best_score, False)
            
    # Save output image
    out_path = os.path.join("outputs", f"annotated_{image_name}")
    cv2.imwrite(out_path, annotated_img)
    log_event("info", f"Finished image processing. Total pipeline time: {(time.time() - start_time)*1000:.1f} ms. Saved output to {out_path}")

def process_video(video_path, detector, ocr, args):
    """
    Processes a video file frame-by-frame and writes annotated output.
    """
    video_name = os.path.basename(video_path)
    log_event("info", f"Loading video: {video_name}")
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        log_event("error", f"Could not open video: {video_path}")
        return
        
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # We will output a scaled version of video (1280 wide) to keep aspect ratio
    target_width = 1280
    aspect_ratio = height / width
    target_height = int(target_width * aspect_ratio)
    
    out_path = os.path.join("outputs", f"annotated_{video_name}")
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out_writer = cv2.VideoWriter(out_path, fourcc, fps, (target_width, target_height))
    
    log_event("info", f"Video Details: {width}x{height} @ {fps:.2f} FPS | Processing total: {total_frames} frames")
    
    frame_idx = 0
    start_time = time.time()
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        # Resize frame
        processed_frame = resize_image(frame, target_width=target_width)
        
        # Preprocess frame
        if not args.no_enhance:
            processed_frame = enhance_contrast(processed_frame)
        if not args.no_denoise:
            processed_frame = remove_noise(processed_frame)
            
        # Detect
        detections = detector.detect(processed_frame, conf_threshold=args.yolo_conf)
        annotated_frame = processed_frame.copy()
        
        for det in detections:
            x1, y1, x2, y2, det_conf = det
            plate = crop_plate(processed_frame, [x1, y1, x2, y2])
            
            if not args.no_deskew:
                plate = perspective_correction(plate)
                
            ocr_results = ocr.extract_text(plate)
            
            best_text = ""
            best_score = 0.0
            is_valid = False
            
            for _, text, score in ocr_results:
                corrected = correct_ocr(text)
                valid = validate_plate(corrected)
                
                if valid:
                    best_text = corrected
                    best_score = score
                    is_valid = True
                    break
                elif score > best_score:
                    best_text = corrected
                    best_score = score
                    is_valid = False
                    
            if best_score >= args.ocr_conf and best_text:
                # Save result with frame info
                save_results(f"{video_name} (F:{frame_idx})", best_text, best_score)
                # Draw boxes
                annotated_frame = draw_boxes(annotated_frame, [x1, y1, x2, y2], best_text, best_score, is_valid)
            else:
                annotated_frame = draw_boxes(annotated_frame, [x1, y1, x2, y2], "", det_conf, False)
                
        out_writer.write(annotated_frame)
        
        frame_idx += 1
        if frame_idx % 30 == 0:
            log_event("info", f"Processed {frame_idx}/{total_frames} frames ({(frame_idx / total_frames)*100:.1f}%)")
            
    cap.release()
    out_writer.release()
    total_time = time.time() - start_time
    log_event("info", f"Finished video processing in {total_time:.1f} seconds. Output saved to {out_path}")

def main():
    parser = argparse.ArgumentParser(description="Advanced Automatic Number Plate Recognition (ANPR) Pipeline")
    parser.add_argument("--input", required=True, help="Path to input image or video file")
    parser.add_argument("--yolo-conf", type=float, default=0.5, help="YOLO plate detection confidence threshold")
    parser.add_argument("--ocr-conf", type=float, default=0.6, help="EasyOCR text confidence threshold")
    parser.add_argument("--no-deskew", action="store_true", help="Disable perspective correction (deskewing)")
    parser.add_argument("--no-enhance", action="store_true", help="Disable contrast enhancement (CLAHE)")
    parser.add_argument("--no-denoise", action="store_true", help="Disable noise filtering (Bilateral filter)")
    
    args = parser.parse_args()
    
    # Initialize models
    detector = LicensePlateDetector(model_path="models/best.pt")
    ocr = LicensePlateOCR()
    
    # Check input type
    input_path = args.input
    if not os.path.exists(input_path):
        log_event("error", f"Input path does not exist: {input_path}")
        return
        
    ext = os.path.splitext(input_path.lower())[1]
    image_exts = ['.jpg', '.jpeg', '.png', '.bmp', '.webp']
    video_exts = ['.mp4', '.avi', '.mov', '.mkv']
    
    if ext in image_exts:
        process_image(input_path, detector, ocr, args)
    elif ext in video_exts:
        process_video(input_path, detector, ocr, args)
    else:
        log_event("error", f"Unsupported file format: {ext}")

if __name__ == "__main__":
    main()
