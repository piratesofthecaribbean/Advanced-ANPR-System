import cv2
import numpy as np

def resize_image(img, target_width=1280):
    """
    Resize image to a standard width while maintaining the aspect ratio.
    """
    h, w = img.shape[:2]
    if w == target_width:
        return img
    aspect_ratio = h / w
    target_height = int(target_width * aspect_ratio)
    return cv2.resize(img, (target_width, target_height), interpolation=cv2.INTER_AREA)

def enhance_contrast(img):
    """
    Applies CLAHE (Contrast Limited Adaptive Histogram Equalization) to improve brightness balance.
    """
    if len(img.shape) == 3:
        # Convert to YUV to enhance only the luminance channel
        yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        yuv[:, :, 0] = clahe.apply(yuv[:, :, 0])
        return cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR)
    else:
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        return clahe.apply(img)

def remove_noise(img):
    """
    Applies Bilateral Filtering to reduce noise while maintaining sharp edges.
    """
    return cv2.bilateralFilter(img, d=9, sigmaColor=75, sigmaSpace=75)

def adaptive_threshold(img):
    """
    Converts image to grayscale and applies adaptive thresholding.
    """
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img
    return cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                 cv2.THRESH_BINARY_INV, 11, 2)

def perspective_correction(img):
    """
    Finds the plate boundaries and performs a perspective transform to flatten/deskew the plate.
    """
    if img is None or img.size == 0:
        return img

    h, w = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Thresholding
    thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                 cv2.THRESH_BINARY_INV, 11, 2)
    
    # Find contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        return img
        
    # Sort contours by area in descending order
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    
    for cnt in contours[:5]:
        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
        
        # We check if we have approximated a 4-point polygon
        if len(approx) == 4:
            pts = approx.reshape(4, 2)
            
            # Sort points in order: top-left, top-right, bottom-right, bottom-left
            rect = np.zeros((4, 2), dtype="float32")
            s = pts.sum(axis=1)
            rect[0] = pts[np.argmin(s)]  # top-left
            rect[2] = pts[np.argmax(s)]  # bottom-right
            
            diff = np.diff(pts, axis=1)
            rect[1] = pts[np.argmin(diff)]  # top-right
            rect[3] = pts[np.argmax(diff)]  # bottom-left
            
            (tl, tr, br, bl) = rect
            
            # Compute widths
            widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
            widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
            maxWidth = max(int(widthA), int(widthB))
            
            # Compute heights
            heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
            heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
            maxHeight = max(int(heightA), int(heightB))
            
            # Ensure sanity of size
            if maxWidth < int(w * 0.4) or maxHeight < int(h * 0.4) or maxWidth > int(w * 1.5):
                continue
                
            dst = np.array([
                [0, 0],
                [maxWidth - 1, 0],
                [maxWidth - 1, maxHeight - 1],
                [0, maxHeight - 1]], dtype="float32")
                
            # Apply warp perspective
            M = cv2.getPerspectiveTransform(rect, dst)
            warped = cv2.warpPerspective(img, M, (maxWidth, maxHeight))
            
            # Resize warped image back to reasonable dimensions if needed
            return warped
            
    return img
