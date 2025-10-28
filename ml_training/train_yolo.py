# /ANPR_Project/ml_training/train_yolo.py

from ultralytics import YOLO
import os

# --- Configuration ---
# Path is relative to where this script (train_yolo.py) is located
DATA_YAML_PATH = 'YOLO_dataset/data.yaml'
MODEL_BASE = 'yolov8s.pt'  # Start with the small pre-trained model
EPOCHS = 150
IMG_SIZE = 640
BATCH_SIZE = 8
MODEL_NAME = 'anpr_final_production_v3'

def start_training():
    print("--- YOLOv8 Training Initialized ---")
    
    if not os.path.exists(DATA_YAML_PATH):
        print(f"FATAL ERROR: data.yaml not found at {DATA_YAML_PATH}")
        print("Ensure your conversion and data organization are complete.")
        return

    # Load the base model
    model = YOLO(MODEL_BASE)
    
    # Start training
    results = model.train(
        data=DATA_YAML_PATH,
        epochs=EPOCHS,
        imgsz=IMG_SIZE,
        batch=BATCH_SIZE,
        name=MODEL_NAME
    )
    
    print("--- Training Complete! ---")
    print(f"Integration File: runs/detect/{MODEL_NAME}/weights/best.pt")

if __name__ == '__main__':
    start_training()