import random
import os
from pathlib import Path

from ultralytics import YOLO
from PIL import Image

# Path to your trained model
MODEL_PATH = "runs_foodgroups/yolov8n_foodgroups/weights/best.pt"

# Choose any directory of images (training, validation, or test)
DATASET_DIR = Path("..") / "data" / "archive" / "evaluation"

# Load YOLO model
model = YOLO(MODEL_PATH)

# Collect all images in this split
image_paths = []
for root, _, files in os.walk(DATASET_DIR):
    for f in files:
        if f.lower().endswith((".jpg", ".jpeg", ".png")):
            image_paths.append(Path(root) / f)

if not image_paths:
    raise RuntimeError("No images found in dataset folder!")

# Pick a random image
img_path = random.choice(image_paths)
print("Selected image:", img_path)

# Run inference
results = model(str(img_path))

# Show YOLO's plotted output
results[0].show()

# Print detections
print("\nDetections:")
for box in results[0].boxes:
    cls_id = int(box.cls.item())
    cls_name = model.names[cls_id]
    conf = float(box.conf.item())
    print(f"{cls_name:20s}  conf={conf:.3f}")
