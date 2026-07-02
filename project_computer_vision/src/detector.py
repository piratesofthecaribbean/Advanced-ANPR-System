import torch
from ultralytics import YOLO

class LicensePlateDetector:
    def __init__(self, model_path='models/best.pt'):
        """
        Initialize the YOLO license plate detector.
        Automatically uses Apple Silicon GPU (MPS) if available.
        """
        self.device = 'mps' if torch.backends.mps.is_available() else 'cpu'
        print(f"[Detector] Initializing YOLO model from {model_path} on device: {self.device}")
        self.model = YOLO(model_path)
        
    def detect(self, image, conf_threshold=0.5):
        """
        Detects license plates in the input image.
        
        Args:
            image (numpy.ndarray): Input image in BGR format.
            conf_threshold (float): Confidence threshold for detections.
            
        Returns:
            list: List of detections, where each detection is [x1, y1, x2, y2, score].
        """
        # Run inference using Ultralytics
        results = self.model(image, conf=conf_threshold, device=self.device, verbose=False)
        
        detections = []
        if len(results) > 0:
            # results[0].boxes contains details of bounding boxes
            for box in results[0].boxes:
                coords = box.xyxy[0].tolist()  # [x1, y1, x2, y2]
                score = box.conf[0].item()
                detections.append([coords[0], coords[1], coords[2], coords[3], score])
                
        return detections
