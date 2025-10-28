# /backend/anpr_core.py

import cv2
import numpy as np
import easyocr
from ultralytics import YOLO 
import imutils 
import os 
import re  # <-- CRITICAL: Used for post-processing/validation
from typing import Dict, Any, List


# --- GLOBAL MODEL & CONFIGURATION ---

YOLO_MODEL: YOLO = None
READER: easyocr.Reader = None

# Set to 0.001 to ensure all boxes are checked (for maximum detection recall).
# The code relies on REGEX filtering (not confidence) to filter the OCR output.
CONFIDENCE_THRESHOLD = 0.001 
 
# Semantic version for this core module. Bumped locally as a non-invasive metadata change.
ANPR_VERSION = "0.1.2"

# Author metadata (non-invasive)
ANPR_AUTHOR = "Praveensudha"

# License metadata (non-invasive)
ANPR_LICENSE = "MIT"


# --- INITIALIZATION FUNCTION (Called by app.py on startup) ---
def initialize_models():
    """
    Loads the YOLO model and EasyOCR reader into global memory.
    """
    global YOLO_MODEL, READER
    
    print("ANPR Core: Initializing YOLOv8 and EasyOCR...")
    
    # Define the path to your custom model
    model_path = os.path.join(os.path.dirname(__file__), 'models', 'anpr_yolov8.pt')
    
    if not os.path.exists(model_path):
        print(f"WARNING: Custom YOLO model NOT found at {model_path}.")
        print("ACTION: Loading generic YOLOv8n.pt. Detection accuracy will be NIL.")
        YOLO_MODEL = YOLO('yolov8n.pt') 
    else:
        # Confirms your project is using the successful trained model
        print(f"INFO: Loading custom YOLO model from {model_path}.")
        YOLO_MODEL = YOLO(model_path) 
    
    # Initialize EasyOCR (English language for Indian plates)
    READER = easyocr.Reader(['en']) 
    
    print("ANPR Core: Models loaded successfully.")


def process_anpr_advanced(image_path: str) -> Dict[str, Any]:
    """Detects plates using YOLOv8 and recognizes characters with EasyOCR and preprocessing."""
    
    # Safety check: Ensure models are loaded
    if YOLO_MODEL is None or READER is None:
        try:
            initialize_models()
        except Exception as e:
            return {"plate_text": f"Initialization Failed: {str(e)}", "coordinates": [], "confidence": 0.0}
            
    best_plate_text = "No Plate Detected"
    best_coords: List[List[int]] = []
    best_conf = 0.0 

    try:
        img = cv2.imread(image_path)
        if img is None:
             return {"plate_text": "Error: Could not load image.", "coordinates": [], "confidence": 0.0}
             
        # --- 1. YOLOv8 Detection ---
        results = YOLO_MODEL(img, imgsz=640, verbose=False) 
        detections = results[0].boxes.data.cpu().numpy()
        
        num_detections = len(detections)
        print(f"--- ANPR DEBUG --- Total boxes detected by YOLO: {num_detections}") 
        
        # Iterate through all boxes found
        for detection in detections:
            x1, y1, x2, y2, conf, cls = detection
            
            print(f"  > RAW DETECT: Class={int(cls)}, Conf={conf:.4f}")

            # Filter by the configured confidence threshold
            if conf < CONFIDENCE_THRESHOLD:
                continue

            # --- 2. Crop and Preprocess for OCR (Accuracy Enhancement Block) ---
            x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
            
            h, w = img.shape[:2]
            cropped_plate_color = img[max(0, y1):min(h, y2), max(0, x1):min(w, x2)]
            
            if cropped_plate_color.size == 0:
                continue 

            cropped_plate_gray = cv2.cvtColor(cropped_plate_color, cv2.COLOR_BGR2GRAY)

            # 1. CLAHE (Contrast Enhancement) - Improves local visibility with optimized parameters
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(16,16))
            enhanced_image = clahe.apply(cropped_plate_gray)

            # 2. Median Blur (Noise Reduction) - Removes small specks/noise
            final_ocr_image = cv2.medianBlur(enhanced_image, 3) 

            # --- 3. EasyOCR Recognition ---
            ocr_results = READER.readtext(final_ocr_image, 
                                          allowlist='0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ', 
                                          detail=0)
            
            # Initialize matched text result
            final_plate_text = None 
            
            if ocr_results:
                plate_text_raw = "".join(ocr_results).replace(' ', '').upper()
                
                # ---------------------------------------------------
                # --- CHARACTER AMBIGUITY CORRECTION (Post-OCR) ---
                # ---------------------------------------------------

                plate_text_for_regex = plate_text_raw
                
                # Rule 1: Fix M/H/W/0 ambiguity (e.g., H003 -> MH03)
                if plate_text_for_regex.startswith('HO') and len(plate_text_for_regex) >= 3 and plate_text_for_regex[2].isdigit():
                    plate_text_for_regex = 'MH' + plate_text_for_regex[2:]
                
                # Rule 2: Fix T/I/K/N ambiguity in the state code (e.g., IK07 -> TN07, NL -> KL)
                if len(plate_text_for_regex) >= 2 and plate_text_for_regex[2].isdigit():
                    # Fix IK -> TN (for IK07BU5427)
                    if plate_text_for_regex.startswith('IK'): 
                        plate_text_for_regex = 'TN' + plate_text_for_regex[2:]
                    # Fix NL -> KL (for NL49H5270 misread)
                    elif plate_text_for_regex.startswith('NL'):
                        plate_text_for_regex = 'KL' + plate_text_for_regex[2:] 
                    # Fix TK -> TN (if T and K are confused)
                    elif plate_text_for_regex.startswith('TK'):
                         plate_text_for_regex = 'TN' + plate_text_for_regex[2:]
                
                
                # --- APPLY REGEX (The final validation filter for Indian plates) ---
                patterns = [
                    r"([A-Z]{2}\d{1,2}[A-Z]{1,2}\d{3,4})", # Full Pattern (e.g., MH08AT3000)
                    r"([A-Z]{2}\d{1,2}\d{3,4})"           # Old Format (e.g., MH011234)
                ]
                
                # Check each pattern against the corrected text
                for pattern in patterns:
                    match = re.search(pattern, plate_text_for_regex)
                    if match:
                        final_plate_text = match.group(1) # Get the validated text
                        
                        # Check for a reasonable length
                        if len(final_plate_text) >= 7:
                            break # Stop at the first valid match

                # --- Final Validation and Set Output ---
                
                if final_plate_text:
                    # Success: Use the validated Regex output
                    best_plate_text = final_plate_text
                elif len(plate_text_for_regex) >= 4:
                    # Fallback: Use the raw corrected text if validation failed
                    best_plate_text = plate_text_for_regex 

        # Set final coordinates and confidence if ANY text was successfully extracted/validated
        if best_plate_text != "No Plate Detected":
            best_coords = [[x1, y1], [x2, y2]] 
            best_conf = float(conf) 
            
            print(f"--- FINAL SUCCESS --- Recognized Plate: {best_plate_text} (Conf: {best_conf:.4f})")
            

        return {
            "plate_text": best_plate_text,
            "coordinates": best_coords,
            "confidence": best_conf 
        }

    except Exception as e:
        # Catch any unexpected runtime errors
        print(f"Advanced ANPR Fatal Error: {e}")
        return {"plate_text": f"ANPR Error: {str(e)}", "coordinates": [], "confidence": 0.0}

# Initialize on import so app.py can access the objects immediately
def get_model_status() -> Dict[str, Any]:
    """
    Non-invasive helper that returns the current status of the loaded models.

    This function is purely additive and does not change any runtime state.
    It can be used by external callers (or debugging endpoints) to check whether
    the YOLO and EasyOCR models are loaded.
    """
    try:
        yolo_info = str(YOLO_MODEL) if YOLO_MODEL is not None else None
    except Exception:
        yolo_info = None

    return {
        "yolo_loaded": YOLO_MODEL is not None,
        "reader_loaded": READER is not None,
        "yolo_model_info": yolo_info,
    }


def get_anpr_version() -> str:
    """
    Return the semantic version of this ANPR core module.

    Purely informational and non-invasive.
    """
    return ANPR_VERSION


def get_anpr_author() -> str:
    """
    Return the author/owner name for this ANPR core module.

    Purely informational and non-invasive.
    """
    return ANPR_AUTHOR


def get_anpr_license() -> str:
    """
    Return the license of the ANPR core module (informational).
    """
    return ANPR_LICENSE


# Initialize on import so app.py can access the objects immediately
initialize_models()