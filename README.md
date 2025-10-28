# 🚗 Real-Time Automatic Number Plate Recognition (ANPR) System

## Project Overview

This project implements a robust, high-accuracy **Automatic Number Plate Recognition (ANPR)** system capable of detecting and recognizing Indian vehicle license plates from both **static images** and **video files**. It leverages a modern, two-stage deep learning pipeline (YOLOv8 + Enhanced OCR) deployed via a seamless React frontend and a Python Flask backend.

### Key Features Achieved

* **High Accuracy Detection:** Utilizes a **custom-trained YOLOv8 model** to accurately locate license plates.
* **Enhanced OCR:** Implements **OpenCV preprocessing (CLAHE/Median Blur)** and **Post-OCR Regular Expressions** to overcome common character ambiguities (M/H, T/I confusion).
* **Video Processing:** Processes uploaded video files frame-by-frame, identifying and logging unique plates found in the stream.
* **Full-Stack Architecture:** Separates the presentation layer (React) from the deep learning processing layer (Flask).
* **Data Persistence:** Records a history of all recognized plates, their confidence scores, and timestamps to an SQLite database.

## 📸 Final Demo and Results

The system accurately localizes and recognizes plates, even with challenging fonts.

| Test Image | Actual Plate | Detected Result | Confidence |
| :---: | :---: | :---: | :---: |
| Motorcycle (MH08) | MH 08 AT 3000 | **MH08AT3000** | 89% |
| White Car (MH20) | MH 20 EE 7598 | **MH20EE7598** | 87% |
| Bike (KL) | KL 49 H 5270 | **KL49H5270** | 87% |

**(NOTE: After you push this file, you can upload the actual screenshots to your repository and replace the text in this section with image links for a truly professional look.)**

### Detection History (Database Log)

All successful detections are logged to the `anpr_log.db` database and displayed on the frontend:


---

## 🛠️ Technology Stack

| Component | Technology | Role |
| :--- | :--- | :--- |
| **Frontend** | **React.js** (JavaScript) | User interface, media upload, and **Bounding Box visualization**. |
| **Backend API** | **Python / Flask** | RESTful API and WebSocket handler (for future live video integration). |
| **Deep Learning** | **YOLOv8** (Custom Trained) | High-accuracy object detection (License Plate Localization). |
| **Computer Vision** | **OpenCV, EasyOCR** | Image preprocessing, character recognition, and video frame handling. |
| **Database** | **SQLite3** | Lightweight database for persisting detection history (`anpr_log.db`). |

---

## ⚙️ Project Setup and Installation

### 1. Clone the Repository and Setup Structure

```bash
# 1. Clone the repository
git clone [https://github.com/Praveensudha/RealTime_ANPR_YOLO.git](https://github.com/Praveensudha/RealTime_ANPR_YOLO.git)
cd RealTime_ANPR_YOLO
```

### 2. Backend Setup (Python / Flask)
```
# Create and activate virtual environment
python -m venv venv
.\venv\Scripts\activate  # For Windows
# OR
source venv/bin/activate  # For Linux/macOS

# Install dependencies
pip install -r backend/requirements.txt
```
### 3. Model Deployment

The model weights (anpr_yolov8.pt) have already been committed to the backend/models/ folder. No manual download is required.

### 4. Frontend Setup (React)

```
# 1. Navigate to the frontend directory
cd frontend

# 2. Install Node.js dependencies
npm install
```
## 🚀 Running the Application

You must run the backend and frontend in separate terminal windows.

### 1. Start the Backend API (Terminal 1)

This initializes the YOLO model and starts the Flask server on http://127.0.0.1:5000.
```
# Ensure your VENV is active
cd backend
python app.py
```

### 2. Start the React Frontend (Terminal 2)

This launches the web interface on http://localhost:3000.

```
cd frontend
npm start
```
### Usage:

1. Analyze Image/Video: Use the main interface to upload a JPEG, PNG, or MP4 file. The backend will process it, and the frontend will display the result (with the bounding box if it's an image).
2. View History: Click the "View Detection History" button to see all logged results in the database.

## Next Steps (Advanced Future Development)

The next feature would be integrating WebSockets (using Flask-SocketIO) to enable true Live Webcam Streaming rather than just file upload processing. This would require modifying the ANPRUploader to capture frames directly from the camera onto an HTML Canvas and stream the data to the server.




