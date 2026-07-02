import easyocr
import torch

class LicensePlateOCR:
    def __init__(self, gpu=None):
        """
        Initialize the EasyOCR English reader.
        
        Args:
            gpu (bool, optional): Whether to use GPU acceleration.
                                  Defaults to True if Apple MPS is available, else False.
        """
        if gpu is None:
            # Check if MPS or CUDA is available for acceleration
            gpu = torch.backends.mps.is_available() or torch.cuda.is_available()
            
        print(f"[OCR] Initializing EasyOCR (GPU={gpu})...")
        self.reader = easyocr.Reader(['en'], gpu=gpu)
        
    def extract_text(self, cropped_plate):
        """
        Extracts text from a cropped plate image.
        
        Args:
            cropped_plate (numpy.ndarray): Cropped license plate image.
            
        Returns:
            list: A list of detections, where each detection is a tuple of (bbox, text, confidence).
        """
        if cropped_plate is None or cropped_plate.size == 0:
            return []
            
        # EasyOCR readtext returns list of: [([x1,y1,x2,y2], text, confidence)]
        return self.reader.readtext(cropped_plate)
