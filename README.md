# RealTime ANPR YOLO 🚗

A **Real-Time Automatic Number Plate Recognition (ANPR)** system built using **YOLOv8**, **Flask**, and **OpenCV**.  
The system detects license plates from live camera feeds or video files and extracts vehicle registration numbers efficiently.

---

## 🧠 Features

- 🚘 Real-time number plate detection using **YOLOv8**
- 🧩 Flask backend for live streaming and result visualization
- 🎞️ Support for both webcam and uploaded video inputs
- 🔤 Optical Character Recognition (OCR) for plate text extraction
- 📁 Modular structure for easy integration with other apps
- 🧾 Logging and result saving

---

## 🧰 Technologies Used

| Category | Tools / Frameworks |
|-----------|--------------------|
| Deep Learning | YOLOv8, PyTorch |
| Backend | Flask (Python) |
| Computer Vision | OpenCV |
| OCR | EasyOCR / Tesseract |
| Frontend | HTML, CSS, JavaScript (optional) |

---

## ⚙️ Installation

```bash
# 1️⃣ Clone the repository
git clone https://github.com/Praveensudha/RealTime_ANPR_YOLO.git
cd RealTime_ANPR_YOLO

# 2️⃣ Create virtual environment
python -m venv venv
venv\Scripts\activate   # (Windows)
# or
source venv/bin/activate  # (Linux/Mac)

# 3️⃣ Install dependencies
pip install -r requirements.txt

# 4️⃣ Run the Flask server
python app.py
```

## Then open your browser and visit:
```bash
http://127.0.0.1:5000/

```

## 📸 Project Structure
```bash
RealTime_ANPR_YOLO/
│
├── ml_training/               # YOLO training dataset and models
├── static/                    # Static assets (CSS, JS, images)
├── templates/                 # HTML templates
├── app.py                     # Flask application
├── requirements.txt           # Dependencies
└── README.md                  # Project overview

```

## 🎯 Future Enhancements

- Add vehicle make & color detection

- Support multi-lane camera feeds

- Deploy as a cloud-based API

- Integrate database for result storage