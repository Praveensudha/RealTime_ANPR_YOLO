# /backend/app.py

from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import cv2
import numpy as np # Used for image processing conversion within video logic
from werkzeug.utils import secure_filename

# Import core ANPR function, DB manager, and initialization logic
from anpr_core import process_anpr_advanced
from anpr_core import initialize_models
from db_manager import log_detection, get_detection_history

app = Flask(__name__)
# Enable CORS for React frontend with additional security headers
CORS(app, resources={r"/api/*": {
    "origins": "http://localhost:3000",
    "methods": ["GET", "POST"],
    "allow_headers": ["Content-Type"]
}})

# --- CONFIGURATION ---
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# --- HELPER: VIDEO PROCESSING FUNCTION ---
def process_video_anpr(video_path, filename):
    """Processes a video file frame-by-frame using the ANPR core."""
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError("Error opening video file.")
        
    log_data = []
    # Process every 8th frame for better detection rate while maintaining performance
    FRAME_SKIP = 8 
    frame_count = 0
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        frame_count += 1
        
        # Only process every Nth frame
        if frame_count % FRAME_SKIP != 0:
            continue
            
        # 1. Save the current frame temporarily for processing 
        temp_img_path = os.path.join(os.path.dirname(video_path), "temp_frame.jpg")
        
        # Use JPEG compression to ensure the file can be read and is small
        cv2.imwrite(temp_img_path, frame, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
        
        # 2. Run the existing high-accuracy ANPR core
        # We assume process_anpr_advanced returns dict with 'plate_text', 'confidence', etc.
        result = process_anpr_advanced(temp_img_path)
        os.remove(temp_img_path) # Delete temporary file
        
        # 3. Log and save unique detections
        if result['plate_text'] != 'No Plate Detected':
            plate_num = result['plate_text']
            
            # Check for plate uniqueness in this video stream (basic tracking)
            if not any(d['plate_number'] == plate_num for d in log_data):
                
                # Log to the permanent history database
                log_detection(plate_num, result["confidence"], f"{filename}_frame_{frame_count}")
                
                # Append to the results list returned to the frontend
                log_data.append({
                    "plate_number": plate_num, 
                    "confidence": result["confidence"], 
                    "frame": frame_count,
                    "media_type": "video"
                })

    cap.release()
    return log_data

# --- API ROUTES ---

@app.route('/api/recognize_media', methods=['POST'])
def recognize_media():
    """Handles both image and video file uploads for ANPR processing."""
    
    if 'media' not in request.files:
        return jsonify({"status": "error", "message": "No file provided"}), 400

    file = request.files['media']
    original_filename = secure_filename(file.filename) # CAPTURE ORIGINAL FILENAME
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], original_filename)

    if file.filename == '':
        return jsonify({"status": "error", "message": "No selected file"}), 400
        
    media_type = None
    
    try:
        file.save(filepath)
        
        # Determine file type and process
        if original_filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            media_type = "image"
            # Process as a single image
            result = process_anpr_advanced(filepath)
            
            log_detection(result["plate_text"], result["confidence"], original_filename)
            
            response = [{"plate_number": result["plate_text"], "coords": result["coordinates"], 
                         "confidence": result["confidence"], "media_type": media_type}]
        
        elif original_filename.lower().endswith(('.mp4', '.avi', '.mov', '.webm')):
            media_type = "video"
            # Process as video (calls the helper function)
            response = process_video_anpr(filepath, original_filename)
        
        else:
            os.remove(filepath)
            return jsonify({"status": "error", "message": "Unsupported file type. Use image or video files."}), 400

        # Clean up the file after processing is complete
        os.remove(filepath)
        
        # Return the filename and the array of results
        return jsonify({
            "status": "success", 
            "filename": original_filename, 
            "results": response
        }), 200

    except Exception as e:
        print(f"Server Error during ANPR processing of {media_type}: {e}")
        # Ensure file cleanup happens even on error
        if os.path.exists(filepath):
            os.remove(filepath) 
        return jsonify({"status": "error", "message": f"Processing failed: {str(e)}"}), 500

@app.route('/api/history', methods=['GET'])
def get_history():
    """Serves the complete history of ANPR detections."""
    try:
        history = get_detection_history()
        return jsonify(history), 200
    except Exception as e:
        print(f"History API Error: {e}")
        return jsonify({"message": "Failed to retrieve history.", "error": str(e)}), 500


if __name__ == '__main__':
    # Initialize ANPR models and database manager before starting the server
    initialize_models()
    
    # Run the server
    app.run(debug=True, port=5000)