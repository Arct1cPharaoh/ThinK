# flutter_yolo_converter.py (inside ThinK/ml/)
from pathlib import Path
from shutil import copy2
from ultralytics import YOLO

YOLO_WEIGHTS_PATH = Path("seg/runs_foodgroups/yolov8n_foodgroups/weights/best.pt")

# __file__ = ThinK/ml/seg/flutter_converter.py
ML_ROOT = Path(__file__).parent.resolve()     # → ThinK/ml/seg
PROJECT_ROOT = ML_ROOT.parent.parent          # → ThinK/

FLUTTER_ONNX_PATH = (
    PROJECT_ROOT / "calorie_counting" / "assets" / "models" / "foodgroups_yolo.onnx"
)

def main():
    print("[INFO] Loading YOLO model:", YOLO_WEIGHTS_PATH)
    model = YOLO(str(YOLO_WEIGHTS_PATH))

    FLUTTER_ONNX_PATH.parent.mkdir(parents=True, exist_ok=True)

    print("[INFO] Exporting ONNX…")
    exported_path = Path(
        model.export(
            format="onnx",
            opset=17,
            imgsz=640,
            dynamic=True,
            optimize=True,
            simplify=True,
            half=False,
        )
    )

    copy2(exported_path, FLUTTER_ONNX_PATH)

    print("\n[OK] Copied to:", FLUTTER_ONNX_PATH)

if __name__ == "__main__":
    main()
