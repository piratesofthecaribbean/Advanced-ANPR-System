import os
import cv2
import time
import threading
import queue
import datetime
import tkinter as tk
from tkinter import filedialog
import customtkinter as ctk
from PIL import Image, ImageTk
import sys

# Ensure project root is in the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import ANPR modules
from src.preprocess import resize_image, enhance_contrast, remove_noise, perspective_correction
from src.detector import LicensePlateDetector
from src.ocr import LicensePlateOCR
from src.utils import validate_plate, correct_ocr, crop_plate, draw_boxes, save_results

# Set CustomTkinter settings
ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")

class ANPRDashboard(ctk.CTk):
    def __init__(self):
        super().__init__()
        
        # Window configuration
        self.title("AI-Powered Automatic Number Plate Recognition (ANPR) Dashboard")
        self.geometry("1280x800")
        self.minsize(1100, 700)
        
        # Pipeline model states
        self.detector = None
        self.ocr = None
        self.models_loaded = False
        
        # State variables
        self.input_file_path = None
        self.is_video = False
        self.video_cap = None
        self.video_playing = False
        self.processed_results = []
        self.annotated_image = None
        self.original_image = None
        
        # Multi-threading communication queue
        self.video_queue = queue.Queue(maxsize=10)
        
        # GUI Layout setup
        self._build_sidebar()
        self._build_main_grid()
        
        # Initialize background loading of AI models
        self.status_bar.configure(text="System Status: Loading AI Models in background...")
        threading.Thread(target=self._load_models, daemon=True).start()
        
    def _load_models(self):
        try:
            self.detector = LicensePlateDetector(model_path="models/best.pt")
            self.ocr = LicensePlateOCR()
            self.models_loaded = True
            self.after(0, lambda: self.status_bar.configure(text="System Status: Ready. AI Models Loaded successfully."))
            self.after(0, lambda: self.btn_process.configure(state="normal"))
        except Exception as e:
            self.after(0, lambda: self.status_bar.configure(text=f"System Status: Error loading models: {str(e)}"))
            
    def _build_sidebar(self):
        # Sidebar Frame
        self.sidebar_frame = ctk.CTkFrame(self, width=280, corner_radius=0)
        self.sidebar_frame.pack(side="left", fill="y", padx=0, pady=0)
        
        # App Title
        self.title_label = ctk.CTkLabel(self.sidebar_frame, text="ANPR SYSTEM", font=ctk.CTkFont(size=22, weight="bold"))
        self.title_label.pack(padx=20, pady=(25, 5))
        self.subtitle_label = ctk.CTkLabel(self.sidebar_frame, text="Apple Silicon Optimized", font=ctk.CTkFont(size=12, slant="italic"))
        self.subtitle_label.pack(padx=20, pady=(0, 25))
        
        # Mode selector
        self.mode_label = ctk.CTkLabel(self.sidebar_frame, text="Input Mode", font=ctk.CTkFont(size=14, weight="bold"))
        self.mode_label.pack(padx=20, pady=(10, 5), anchor="w")
        
        self.mode_selector = ctk.CTkSegmentedButton(self.sidebar_frame, values=["Image", "Video"], command=self._on_mode_change)
        self.mode_selector.set("Image")
        self.mode_selector.pack(padx=20, pady=5, fill="x")
        
        # Sliders block
        self.sliders_label = ctk.CTkLabel(self.sidebar_frame, text="Confidence Settings", font=ctk.CTkFont(size=14, weight="bold"))
        self.sliders_label.pack(padx=20, pady=(20, 5), anchor="w")
        
        # YOLO Slider
        self.yolo_lbl = ctk.CTkLabel(self.sidebar_frame, text="YOLO Detection Conf: 0.50", font=ctk.CTkFont(size=12))
        self.yolo_lbl.pack(padx=20, pady=(5, 0), anchor="w")
        self.yolo_slider = ctk.CTkSlider(self.sidebar_frame, from_=0.1, to=1.0, number_of_steps=18, command=self._update_yolo_label)
        self.yolo_slider.set(0.5)
        self.yolo_slider.pack(padx=20, pady=5, fill="x")
        
        # OCR Slider
        self.ocr_lbl = ctk.CTkLabel(self.sidebar_frame, text="OCR Text Conf: 0.60", font=ctk.CTkFont(size=12))
        self.ocr_lbl.pack(padx=20, pady=(5, 0), anchor="w")
        self.ocr_slider = ctk.CTkSlider(self.sidebar_frame, from_=0.1, to=1.0, number_of_steps=18, command=self._update_ocr_label)
        self.ocr_slider.set(0.6)
        self.ocr_slider.pack(padx=20, pady=5, fill="x")
        
        # Checkboxes block
        self.preprocess_lbl = ctk.CTkLabel(self.sidebar_frame, text="Preprocessing Pipeline", font=ctk.CTkFont(size=14, weight="bold"))
        self.preprocess_lbl.pack(padx=20, pady=(20, 5), anchor="w")
        
        self.cb_clahe = ctk.CTkCheckBox(self.sidebar_frame, text="Contrast Enhance (CLAHE)")
        self.cb_clahe.select()
        self.cb_clahe.pack(padx=25, pady=5, anchor="w")
        
        self.cb_denoise = ctk.CTkCheckBox(self.sidebar_frame, text="Bilateral Noise Filter")
        self.cb_denoise.select()
        self.cb_denoise.pack(padx=25, pady=5, anchor="w")
        
        self.cb_deskew = ctk.CTkCheckBox(self.sidebar_frame, text="Perspective Deskewing")
        self.cb_deskew.select()
        self.cb_deskew.pack(padx=25, pady=5, anchor="w")
        
        # Run Button
        self.btn_process = ctk.CTkButton(self.sidebar_frame, text="PROCESS FILE", state="disabled", fg_color="#3b82f6", hover_color="#2563eb", font=ctk.CTkFont(weight="bold"), command=self._process_file)
        self.btn_process.pack(padx=20, pady=(40, 10), fill="x")
        
        # Clear Button
        self.btn_clear = ctk.CTkButton(self.sidebar_frame, text="CLEAR DATA", fg_color="#ef4444", hover_color="#dc2626", font=ctk.CTkFont(weight="bold"), command=self._clear_data)
        self.btn_clear.pack(padx=20, pady=10, fill="x")
        
        # Theme toggle
        self.theme_label = ctk.CTkLabel(self.sidebar_frame, text="Theme Mode", font=ctk.CTkFont(size=12))
        self.theme_label.pack(side="bottom", padx=20, pady=(5, 5))
        self.theme_toggle = ctk.CTkOptionMenu(self.sidebar_frame, values=["Dark", "Light"], command=self._on_theme_toggle)
        self.theme_toggle.pack(side="bottom", padx=20, pady=(5, 20))
        
    def _build_main_grid(self):
        # Container frame
        self.main_container = ctk.CTkFrame(self)
        self.main_container.pack(side="left", fill="both", expand=True, padx=15, pady=15)
        
        # Upper area: Display & Results Grid
        self.upper_frame = ctk.CTkFrame(self.main_container, fg_color="transparent")
        self.upper_frame.pack(fill="both", expand=True, padx=0, pady=(0, 10))
        
        # Left Panel: Video/Frame Player Box
        self.player_frame = ctk.CTkFrame(self.upper_frame, corner_radius=10)
        self.player_frame.pack(side="left", fill="both", expand=True, padx=(0, 10), pady=0)
        
        self.player_title = ctk.CTkLabel(self.player_frame, text="Input Stream Visualizer", font=ctk.CTkFont(size=15, weight="bold"))
        self.player_title.pack(padx=15, pady=10, anchor="w")
        
        # Interactive Image Canvas/Label
        self.visualizer_lbl = ctk.CTkLabel(self.player_frame, text="Click 'Browse File' to load an Image or Video", fg_color="#18181b", corner_radius=8)
        self.visualizer_lbl.pack(fill="both", expand=True, padx=15, pady=(0, 15))
        self.visualizer_lbl.bind("<Button-1>", lambda e: self._browse_file())
        
        # Right Panel: Detections Box
        self.detections_frame = ctk.CTkFrame(self.upper_frame, width=420, corner_radius=10)
        self.detections_frame.pack(side="right", fill="both", padx=0, pady=0)
        self.detections_frame.pack_propagate(False)
        
        self.detections_title = ctk.CTkLabel(self.detections_frame, text="Detected License Plates", font=ctk.CTkFont(size=15, weight="bold"))
        self.detections_title.pack(padx=15, pady=10, anchor="w")
        
        # Scrollable area for plate items
        self.scroll_plates = ctk.CTkScrollableFrame(self.detections_frame, fg_color="transparent")
        self.scroll_plates.pack(fill="both", expand=True, padx=10, pady=(0, 10))
        
        # Bottom area: Log Console & Controls
        self.bottom_frame = ctk.CTkFrame(self.main_container, height=180, corner_radius=10)
        self.bottom_frame.pack(fill="x", side="bottom", padx=0, pady=0)
        self.bottom_frame.pack_propagate(False)
        
        self.log_header = ctk.CTkFrame(self.bottom_frame, fg_color="transparent")
        self.log_header.pack(fill="x", padx=15, pady=5)
        
        self.console_title = ctk.CTkLabel(self.log_header, text="System Log Console", font=ctk.CTkFont(size=14, weight="bold"))
        self.console_title.pack(side="left", pady=5)
        
        # Log export controls
        self.btn_export_csv = ctk.CTkButton(self.log_header, text="Export CSV", width=100, fg_color="#10b981", hover_color="#059669", font=ctk.CTkFont(weight="bold"), command=self._export_csv)
        self.btn_export_csv.pack(side="right", padx=(5, 0))
        
        # Text Console box
        self.console_text = ctk.CTkTextbox(self.bottom_frame, state="disabled", font=ctk.CTkFont(family="Courier", size=12), fg_color="#111827", text_color="#10b981")
        self.console_text.pack(fill="both", expand=True, padx=15, pady=(0, 10))
        
        # Status Bar
        self.status_bar = ctk.CTkLabel(self, text="System Status: Initializing...", font=ctk.CTkFont(size=11), anchor="w", fg_color="#1e293b", text_color="#94a3b8")
        self.status_bar.pack(side="bottom", fill="x", padx=0, pady=0)
        
    def _update_yolo_label(self, val):
        self.yolo_lbl.configure(text=f"YOLO Detection Conf: {val:.2f}")
        
    def _update_ocr_label(self, val):
        self.ocr_lbl.configure(text=f"OCR Text Conf: {val:.2f}")
        
    def _on_mode_change(self, val):
        self.is_video = (val == "Video")
        self._clear_data()
        
    def _on_theme_toggle(self, val):
        ctk.set_appearance_mode(val.lower())
        
    def _log(self, text):
        timestamp = datetime.datetime.now().strftime("%H:%M:%S")
        log_line = f"[{timestamp}] {text}\n"
        self.console_text.configure(state="normal")
        self.console_text.insert("end", log_line)
        self.console_text.see("end")
        self.console_text.configure(state="disabled")
        
    def _browse_file(self):
        if self.is_video:
            file_types = [("Video Files", "*.mp4 *.avi *.mov *.mkv")]
        else:
            file_types = [("Image Files", "*.jpg *.jpeg *.png *.bmp *.webp")]
            
        selected_file = filedialog.askopenfilename(filetypes=file_types)
        if selected_file:
            self.input_file_path = selected_file
            self._log(f"Selected file: {os.path.basename(selected_file)}")
            self.status_bar.configure(text=f"File selected: {os.path.basename(selected_file)}")
            
            if not self.is_video:
                self.video_playing = False
                self.original_image = cv2.imread(selected_file)
                self._display_opencv_image(self.original_image)
            else:
                self.video_playing = False
                if self.video_cap:
                    self.video_cap.release()
                self.video_cap = cv2.VideoCapture(selected_file)
                ret, frame = self.video_cap.read()
                if ret:
                    self._display_opencv_image(frame)
                    
    def _display_opencv_image(self, img):
        if img is None:
            return
        # Resize image to fit visualizer frame dimensions
        visualizer_w = self.visualizer_lbl.winfo_width()
        visualizer_h = self.visualizer_lbl.winfo_height()
        
        if visualizer_w <= 1 or visualizer_h <= 1:
            # Fallback if label has not rendered fully yet
            visualizer_w, visualizer_h = 750, 450
            
        h, w = img.shape[:2]
        
        # Calculate aspect ratio scaling
        scale = min(visualizer_w / w, visualizer_h / h)
        new_w = int(w * scale)
        new_h = int(h * scale)
        
        resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
        rgb_img = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        
        # Convert to CTkImage
        pil_img = Image.fromarray(rgb_img)
        ctk_img = ctk.CTkImage(light_image=pil_img, dark_image=pil_img, size=(new_w, new_h))
        
        self.visualizer_lbl.configure(image=ctk_img, text="")
        self.visualizer_lbl.image = ctk_img  # Keep reference
        
    def _clear_data(self):
        self.input_file_path = None
        self.video_playing = False
        if self.video_cap:
            self.video_cap.release()
            self.video_cap = None
        self.visualizer_lbl.configure(image=None, text="Click 'Browse File' to load an Image or Video")
        self.processed_results.clear()
        
        # Clear detected plates list
        for child in self.scroll_plates.winfo_children():
            child.destroy()
            
        self._log("Cleared loaded assets and result states.")
        
    def _export_csv(self):
        if not self.processed_results:
            self._log("Warning: No results to export.")
            return
        
        output_csv = filedialog.asksaveasfilename(defaultextension=".csv", filetypes=[("CSV Files", "*.csv")])
        if output_csv:
            try:
                with open(output_csv, mode='w', newline='', encoding='utf-8') as f:
                    writer = csv.writer(f)
                    writer.writerow(["Source File", "Plate Text", "Confidence", "Timestamp", "Valid Indian"])
                    for row in self.processed_results:
                        writer.writerow(row)
                self._log(f"Exported {len(self.processed_results)} entries to CSV: {output_csv}")
            except Exception as e:
                self._log(f"Error exporting CSV: {str(e)}")
                
    def _process_file(self):
        if not self.models_loaded:
            self._log("AI models are not fully loaded yet.")
            return
        if not self.input_file_path:
            self._log("Please select a file to process first.")
            return
            
        self._log("Starting ANPR Processing Pipeline...")
        self.btn_process.configure(state="disabled")
        
        # Clear past plates visual lists
        for child in self.scroll_plates.winfo_children():
            child.destroy()
            
        if not self.is_video:
            # Threaded Image processing
            threading.Thread(target=self._process_single_image_thread, daemon=True).start()
        else:
            # Threaded Video processing
            self.video_playing = True
            threading.Thread(target=self._process_video_thread, daemon=True).start()
            self.after(100, self._consume_video_queue)
            
    def _process_single_image_thread(self):
        start_time = time.time()
        img = self.original_image.copy()
        
        # Load sliders values
        yolo_thresh = self.yolo_slider.get()
        ocr_thresh = self.ocr_slider.get()
        
        # 1. Scaling
        processed = resize_image(img, target_width=1280)
        
        # 2. Contrast Enhancement
        if self.cb_clahe.get():
            self.after(0, lambda: self._log("Enhancing contrast using CLAHE..."))
            processed = enhance_contrast(processed)
            
        # 3. Noise Removal
        if self.cb_denoise.get():
            self.after(0, lambda: self._log("Smoothing noise with Bilateral Filter..."))
            processed = remove_noise(processed)
            
        # 4. YOLO Detection
        self.after(0, lambda: self._log(f"Running license plate detection (Conf: {yolo_thresh:.2f})..."))
        detections = self.detector.detect(processed, conf_threshold=yolo_thresh)
        self.after(0, lambda: self._log(f"Found {len(detections)} plate(s)."))
        
        annotated = processed.copy()
        
        for idx, det in enumerate(detections):
            x1, y1, x2, y2, det_conf = det
            
            # Crop Plate
            plate_crop = crop_plate(processed, [x1, y1, x2, y2])
            orig_plate_crop = plate_crop.copy()
            
            # 5. Perspective Deskewing
            if self.cb_deskew.get():
                plate_crop = perspective_correction(plate_crop)
                
            # Preprocess crop for display
            preprocessed_crop = cv2.cvtColor(plate_crop, cv2.COLOR_BGR2GRAY)
            _, preprocessed_crop = cv2.threshold(preprocessed_crop, 64, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
            
            # 6. OCR Text Extraction
            ocr_results = self.ocr.extract_text(plate_crop)
            
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
            
            # Save results
            timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            self.processed_results.append([os.path.basename(self.input_file_path), best_text, f"{best_score:.4f}", timestamp, str(is_valid)])
            
            if best_score >= ocr_thresh and best_text:
                self.after(0, lambda t=best_text, s=best_score, v=is_valid: self._log(f"Detected Plate: '{t}' (Conf: {s:.2f}) | Valid Indian: {v}"))
                annotated = draw_boxes(annotated, [x1, y1, x2, y2], best_text, best_score, is_valid)
                self.after(0, lambda op=orig_plate_crop, pp=preprocessed_crop, t=best_text, s=best_score, v=is_valid: 
                           self._add_plate_ui_card(op, pp, t, s, v))
            else:
                annotated = draw_boxes(annotated, [x1, y1, x2, y2], "", det_conf, False)
                self.after(0, lambda op=orig_plate_crop, pp=preprocessed_crop, t="REJECTED / LOW CONF", s=best_score, v=False:
                           self._add_plate_ui_card(op, pp, t, s, v))
                
        # Update canvas image
        self.after(0, lambda: self._display_opencv_image(annotated))
        self.after(0, lambda: self.btn_process.configure(state="normal"))
        
        # Save output file
        out_path = os.path.join("outputs", f"annotated_{os.path.basename(self.input_file_path)}")
        cv2.imwrite(out_path, annotated)
        self.after(0, lambda: self._log(f"Annotated output saved to {out_path}"))
        self.after(0, lambda: self.status_bar.configure(text=f"Finished processing in {(time.time()-start_time)*1000:.0f} ms."))

    def _process_video_thread(self):
        cap = cv2.VideoCapture(self.input_file_path)
        
        yolo_thresh = self.yolo_slider.get()
        ocr_thresh = self.ocr_slider.get()
        
        frame_idx = 0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        while cap.isOpened() and self.video_playing:
            ret, frame = cap.read()
            if not ret:
                break
                
            # Resize
            processed_frame = resize_image(frame, target_width=1280)
            
            # Preprocess
            if self.cb_clahe.get():
                processed_frame = enhance_contrast(processed_frame)
            if self.cb_denoise.get():
                processed_frame = remove_noise(processed_frame)
                
            # Detect
            detections = self.detector.detect(processed_frame, conf_threshold=yolo_thresh)
            annotated_frame = processed_frame.copy()
            
            frame_plate_results = []
            
            for det in detections:
                x1, y1, x2, y2, det_conf = det
                plate_crop = crop_plate(processed_frame, [x1, y1, x2, y2])
                orig_plate_crop = plate_crop.copy()
                
                if self.cb_deskew.get():
                    plate_crop = perspective_correction(plate_crop)
                    
                ocr_results = self.ocr.extract_text(plate_crop)
                
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
                        
                timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                self.processed_results.append([f"{os.path.basename(self.input_file_path)} (F:{frame_idx})", best_text, f"{best_score:.4f}", timestamp, str(is_valid)])
                
                if best_score >= ocr_thresh and best_text:
                    annotated_frame = draw_boxes(annotated_frame, [x1, y1, x2, y2], best_text, best_score, is_valid)
                    preprocessed_crop = cv2.cvtColor(plate_crop, cv2.COLOR_BGR2GRAY)
                    _, preprocessed_crop = cv2.threshold(preprocessed_crop, 64, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
                    frame_plate_results.append((orig_plate_crop, preprocessed_crop, best_text, best_score, is_valid))
                else:
                    annotated_frame = draw_boxes(annotated_frame, [x1, y1, x2, y2], "", det_conf, False)
            
            # Put output frame into GUI queue
            try:
                # Blocks if queue is full until UI thread consumes
                self.video_queue.put((annotated_frame, frame_plate_results, frame_idx, total_frames), timeout=2.0)
            except queue.Full:
                pass
                
            frame_idx += 1
            
        cap.release()
        self.video_playing = False
        self.after(0, lambda: self.btn_process.configure(state="normal"))
        self.after(0, lambda: self._log("Video Processing Finished."))
        self.after(0, lambda: self.status_bar.configure(text="Video Processing Complete."))
        
    def _consume_video_queue(self):
        if not self.video_playing and self.video_queue.empty():
            return
            
        try:
            while True:
                # Read all available items in queue to avoid lag, but only render the last one
                frame, plate_results, frame_idx, total_frames = self.video_queue.get_nowait()
                
                # Render frame
                self._display_opencv_image(frame)
                
                # Update status bar percentage
                pct = (frame_idx / total_frames) * 100 if total_frames > 0 else 0
                self.status_bar.configure(text=f"Processing Video Frame: {frame_idx}/{total_frames} ({pct:.1f}%)")
                
                # Render UI cards for detected plates in this frame
                if plate_results:
                    for op, pp, t, s, v in plate_results:
                        self._add_plate_ui_card(op, pp, t, s, v)
                        self._log(f"Frame {frame_idx} Plate: '{t}' (Conf: {s:.2f})")
                        
                self.video_queue.task_done()
        except queue.Empty:
            pass
            
        if self.video_playing:
            self.after(30, self._consume_video_queue)

    def _add_plate_ui_card(self, original_crop, preprocessed_crop, text, score, is_valid):
        # Create a container frame for this card
        card_frame = ctk.CTkFrame(self.scroll_plates, corner_radius=6, border_width=1, border_color="#374151")
        card_frame.pack(fill="x", padx=5, pady=5)
        
        # Display image previews
        # Original Crop preview
        h_op, w_op = original_crop.shape[:2]
        new_w = int(w_op * 40 / h_op) if h_op > 0 else 80
        orig_resized = cv2.resize(original_crop, (new_w, 40), interpolation=cv2.INTER_AREA)
        orig_rgb = cv2.cvtColor(orig_resized, cv2.COLOR_BGR2RGB)
        orig_pil = Image.fromarray(orig_rgb)
        orig_ctk = ctk.CTkImage(light_image=orig_pil, dark_image=orig_pil, size=(new_w, 40))
        
        img_lbl_orig = ctk.CTkLabel(card_frame, image=orig_ctk, text="")
        img_lbl_orig.image = orig_ctk
        img_lbl_orig.grid(row=0, column=0, rowspan=2, padx=10, pady=10)
        
        # Text details
        lbl_text = ctk.CTkLabel(card_frame, text=text, font=ctk.CTkFont(size=14, weight="bold"))
        lbl_text.grid(row=0, column=1, padx=(5, 10), pady=(8, 2), anchor="w")
        
        # Color code validation output
        status_text = "VALID INDIAN PLATE" if is_valid else "GENERIC / INVALID"
        status_color = "#10b981" if is_valid else "#f59e0b"
        lbl_status = ctk.CTkLabel(card_frame, text=status_text, font=ctk.CTkFont(size=10, weight="bold"), text_color=status_color)
        lbl_status.grid(row=1, column=1, padx=(5, 10), pady=(0, 8), anchor="w")
        
        # Confidence score
        lbl_score = ctk.CTkLabel(card_frame, text=f"Conf: {score:.2f}", font=ctk.CTkFont(size=11, slant="italic"))
        lbl_score.grid(row=0, column=2, rowspan=2, padx=10, pady=10, anchor="e")
        
        # Force scrolling to the bottom to display new results
        self.scroll_plates._parent_canvas.yview_moveto(1.0)

    def destroy(self):
        self.video_playing = False
        if self.video_cap:
            self.video_cap.release()
        super().destroy()

if __name__ == "__main__":
    app = ANPRDashboard()
    app.mainloop()
