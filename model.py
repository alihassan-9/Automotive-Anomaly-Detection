import os
import torch
from roboflow import Roboflow
from ultralytics import YOLO

ROBOFLOW_API_KEY = os.environ["ROBOFLOW_API_KEY"]

EPOCHS = 100
IMAGE_SIZE = 640
BATCH_SIZE = 8
MODEL_ARCH = "yolov8n-seg.pt"
TRAIN_PROJECT_NAME = "car_damage_segmentation_v1"
TRAIN_RUN_NAME = f"{MODEL_ARCH}_e{EPOCHS}_b{BATCH_SIZE}"

def train_model():
    print("--- YOLOv8 Instance Segmentation Training Pipeline ---")

    print("\n[Phase 1/3] Downloading Dataset from Roboflow...")
    try:
        roboflow_client = Roboflow(api_key=ROBOFLOW_API_KEY)
        project = roboflow_client.workspace("alis-project").project("car-crash-damage-detection")
        dataset = project.version(1).download("yolov8")
        data_yaml_path = os.path.join(dataset.location, "data.yaml")
        print(f"Dataset downloaded to: {dataset.location}")
    except Exception as e:
        print(f"Error during dataset download: {e}")
        return

    print(f"\n[Phase 2/3] Initializing YOLOv8 Model: {MODEL_ARCH}...")
    model = YOLO(MODEL_ARCH)

    print("\n[Phase 3/3] Starting Training Session...")
    print(f"  Configuration:")
    print(f"    - Epochs: {EPOCHS}")
    print(f"    - Batch Size: {BATCH_SIZE}")
    print(f"    - Image Size: {IMAGE_SIZE}")

    device = 0 if torch.cuda.is_available() else 'cpu'
    print(f"  Using device: {'GPU' if device == 0 else 'CPU'}")
    print("-" * 40)

    try:
        model.train(
            data=data_yaml_path,
            epochs=EPOCHS,
            imgsz=IMAGE_SIZE,
            batch=BATCH_SIZE,
            project=TRAIN_PROJECT_NAME,
            name=TRAIN_RUN_NAME,
            exist_ok=True,
            patience=25,
            device=device
        )
        print("\n--- Training Session Finished ---")
        print("The best model weights are saved as 'best.pt' in the project run directory.")

    except Exception as e:
        print(f"An error occurred during the training process: {e}")

if __name__ == '__main__':
    train_model()
