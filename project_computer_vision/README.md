# Advanced Automatic Number Plate Recognition (ANPR) System

A modular, high-performance Automatic Number Plate Recognition (ANPR) system featuring **YOLOv8/v9 (via Ultralytics)**, **EasyOCR**, and **OpenCV** with a beautiful **CustomTkinter** dark mode GUI dashboard. 

Optimized to run with **PyTorch MPS (Metal Performance Shaders)** GPU acceleration on Apple Silicon (M1/M2/M3 MacBooks).

---

## Key Enhancements

1. **CustomTkinter Dark Mode Dashboard:** A modern desktop dashboard replacing traditional Jupyter Notebook outputs. Includes sliders for tuning YOLO and OCR confidences, checkboxes for preprocessing toggles, and live visualization of video streams.
2. **Indian Number Plate Validation:** Uses regex patterns (`^[A-Z]{2}[0-9]{2}[A-Z]{1,2}[0-9]{4}$`) to recognize standard Indian license plates (e.g. `UP32AB1234`) and filters out random noise/generic texts.
3. **Context-Aware OCR Correction:** Automatically corrects OCR mistakes like swapping `O` ↔ `0`, `I`/`l` ↔ `1`, `S` ↔ `5`, `B` ↔ `8`, `Z` ↔ `2`, `A` ↔ `4` depending on character position constraints in Indian license plates.
4. **Perspective Deskewing (Warp):** Detects contours of the cropped license plate and applies a perspective warp to flatten tilted or angled plates, heavily increasing EasyOCR extraction accuracy.
5. **Apple Silicon Acceleration:** Natively detects and utilizes macOS GPU (`device='mps'`) for near real-time YOLO object detection.
6. **Unified Image & Video Pipelines:** Processes single-image files or handles MP4/MOV/AVI video frame streams synchronously or via responsive GUI queues.
7. **Comprehensive Logging:** Exports timestamped event entries to `outputs/log.txt` and annotated results to structured CSV tables.

---

## Directory Structure

```
ANPR_Project/
├── dataset/
│     ├── images/         # Place input test images here
│     └── videos/         # Place input test videos here
├── models/
│     └── best.pt         # Pre-trained license plate YOLO model
├── outputs/
│     ├── log.txt         # Event and execution timing logs
│     ├── results.csv     # Extracted plate CSV sheets
│     └── annotated_*     # Saved output visual files
├── src/
│     ├── detector.py     # YOLO detection class wrapper (MPS ready)
│     ├── ocr.py          # EasyOCR reader extraction logic
│     ├── preprocess.py   # CLAHE contrast, Bilateral filter, and Perspective warp
│     ├── utils.py        # Indian validation, OCR correction, CSV & log writers
│     ├── main.py         # CLI entry point
│     └── gui.py          # CustomTkinter Dark Mode GUI Dashboard
├── requirements.txt      # List of dependencies
└── README.md             # Documentation
```

---

## Installation & Setup

### Step 1: Initialize Virtual Environment
Configure virtual environment using Python 3.11:
```bash
python3.11 -m venv venv
source venv/bin/activate
```

### Step 2: Install PyTorch (Apple Silicon native)
```bash
pip install torch torchvision torchaudio
```

### Step 3: Install Remaining Requirements
```bash
pip install -r requirements.txt
```

### Step 4: Add the Pre-trained YOLO weights
Make sure your custom license plate detector model is placed inside the `models/` directory:
- Name: `models/best.pt`

---

## CLI Usage

Run prediction on a single image or video using the CLI interface:
```bash
# Process an image file
python -m src.main --input dataset/images/sample.jpg

# Process a video file
python -m src.main --input dataset/videos/sample.mp4

# Run with custom detection and OCR thresholds
python -m src.main --input dataset/images/sample.jpg --yolo-conf 0.65 --ocr-conf 0.70

# Disable preprocessing features
python -m src.main --input dataset/images/sample.jpg --no-deskew --no-enhance
```

---

## GUI Dashboard Usage

Launch the modern desktop application:
```bash
python -m src.gui
```

### Dashboard Layout & Controls:
* **Sidebar Controls:**
  * Toggle mode between **Image** and **Video**.
  * Adjust sliders for **YOLO Detection** and **OCR Text** confidence cutoffs.
  * Toggle individual preprocessing stages (Contrast CLAHE, Bilateral filtering, Perspective deskewing).
  * Run processing with **Process File**.
* **Visualizer Frame:** Click to select a file or view live visual feedback of detections.
* **Results Panel:** Displays cropped plate crops (original and binarized), extracted text, and validation badges.
* **Export Options:** Click **Export CSV** to write output reports.
