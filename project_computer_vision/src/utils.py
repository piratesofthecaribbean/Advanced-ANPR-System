import re
import cv2
import csv
import os
import datetime

# Character conversion maps for OCR correction
char_to_digit = {
    'O': '0', 'I': '1', 'J': '3', 'A': '4', 'S': '5', 'G': '6', 'B': '8', 'Z': '2', 'T': '7', 'D': '0'
}

digit_to_char = {
    '0': 'O', '1': 'I', '3': 'J', '4': 'A', '5': 'S', '6': 'G', '8': 'B', '2': 'Z', '7': 'T'
}

def validate_plate(text):
    """
    Validates if the text matches the standard Indian license plate format.
    Example: UP32AB1234 or MH12DE1234
    Format: State Code (2 letters) + RTO Code (2 digits) + Series (1-2 letters) + Number (4 digits)
    """
    if not text:
        return False
    # Clean text to uppercase and remove all non-alphanumeric chars
    cleaned = re.sub(r'[^A-Z0-9]', '', text.upper())
    # Regex pattern: State (2 letters), RTO (2 digits), Letters (1-2 letters), Number (4 digits)
    pattern = r'^[A-Z]{2}[0-9]{2}[A-Z]{1,2}[0-9]{4}$'
    return bool(re.match(pattern, cleaned))

def correct_ocr(text):
    """
    Applies position-aware OCR correction based on Indian number plate structures:
    - 9 characters: LL DD L DDDD
    - 10 characters: LL DD LL DDDD
    (where L is letter, D is digit)
    """
    if not text:
        return ""
        
    # Clean text: uppercase, remove spaces/hyphens
    cleaned = re.sub(r'[^A-Z0-9]', '', text.upper())
    length = len(cleaned)
    
    corrected = []
    if length == 9:
        # Expected: LL DD L DDDD
        # Indices: 01 23 4 5678
        for i in range(9):
            char = cleaned[i]
            if i in [0, 1, 4]:  # Should be letters
                corrected.append(digit_to_char.get(char, char))
            else:  # Should be digits
                corrected.append(char_to_digit.get(char, char))
        return "".join(corrected)
        
    elif length == 10:
        # Expected: LL DD LL DDDD
        # Indices: 01 23 45 6789
        for i in range(10):
            char = cleaned[i]
            if i in [0, 1, 4, 5]:  # Should be letters
                corrected.append(digit_to_char.get(char, char))
            else:  # Should be digits
                corrected.append(char_to_digit.get(char, char))
        return "".join(corrected)
        
    else:
        # Best-effort general correction
        # Let's fix first two characters as letters, last 4 as digits
        for i in range(length):
            char = cleaned[i]
            if i in [0, 1]:  # Usually letters
                corrected.append(digit_to_char.get(char, char))
            elif i >= length - 4:  # Usually digits
                corrected.append(char_to_digit.get(char, char))
            else:
                corrected.append(char)
        return "".join(corrected)

def crop_plate(image, bbox):
    """
    Crops the license plate area from the image given a bounding box.
    """
    x1, y1, x2, y2 = map(int, bbox[:4])
    h, w = image.shape[:2]
    # Restrict boundaries
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)
    return image[y1:y2, x1:x2]

def draw_boxes(image, bbox, text, score, is_valid=False):
    """
    Draws custom styled bounding boxes and transparent labels on the image.
    Uses Green for validated Indian plates, Orange/Red for others.
    """
    img = image.copy()
    x1, y1, x2, y2 = map(int, bbox[:4])
    
    # Choose color scheme: Green for valid Indian plate, Red for invalid/generic
    color = (0, 200, 80) if is_valid else (0, 50, 255) # BGR
    
    # Draw bounding box
    cv2.rectangle(img, (x1, y1), (x2, y2), color, 3)
    
    # Create text label
    label = f"{text} ({score:.2f})" if text else "Plate"
    
    # Determine text size
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.6
    thickness = 2
    (text_w, text_h), baseline = cv2.getTextSize(label, font, font_scale, thickness)
    
    # Draw label background box (semi-transparent overlay)
    label_y1 = max(0, y1 - text_h - 10)
    label_y2 = y1
    label_x1 = x1
    label_x2 = x1 + text_w + 10
    
    # Blend semi-transparent rectangle
    overlay = img.copy()
    cv2.rectangle(overlay, (label_x1, label_y1), (label_x2, label_y2), color, -1)
    cv2.addWeighted(overlay, 0.7, img, 0.3, 0, img)
    
    # Draw text
    cv2.putText(img, label, (label_x1 + 5, label_y2 - 5), font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)
    
    return img

def save_results(image_name, plate_text, confidence, output_csv="outputs/results.csv"):
    """
    Saves detection results to a CSV file.
    Header: Image Name, Detected Plate, Confidence, Timestamp
    """
    file_exists = os.path.exists(output_csv)
    
    # Ensure directory exists
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    
    with open(output_csv, mode='a', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["Image Name", "Detected Plate", "Confidence", "Timestamp"])
            
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        writer.writerow([image_name, plate_text, f"{confidence:.4f}", timestamp])

def log_event(event_type, message, log_path="outputs/log.txt"):
    """
    Writes a timestamped execution log statement.
    """
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_line = f"[{timestamp}] {event_type.upper()}: {message}\n"
    
    with open(log_path, mode='a', encoding='utf-8') as f:
        f.write(log_line)
    # Also print to console
    print(log_line.strip())
