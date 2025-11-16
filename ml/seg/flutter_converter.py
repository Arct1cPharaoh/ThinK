
# seg/flutter_converter.py  (run from ThinK/ml)

from pathlib import Path
from ultralytics import YOLO

SCRIPT_DIR = Path(__file__).resolve().parent   # .../ThinK/ml/seg
ML_ROOT = SCRIPT_DIR.parent                    # .../ThinK/ml

YOLO_WEIGHTS_PATH = SCRIPT_DIR / "runs_foodgroups/yolov8n_foodgroups/weights/best.pt"
OUT_PATH = ML_ROOT / "foodgroups_yolo.onnx"


def main():
    print("[INFO] Loading YOLO model:", YOLO_WEIGHTS_PATH)
    model = YOLO(str(YOLO_WEIGHTS_PATH))

    print("[INFO] Exporting ONNX to:", OUT_PATH)
    exported = model.export(
        format="onnx",
        opset=12,       # keep IR version compatible with mobile ORT
        dynamic=True,
        optimize=False,
        simplify=False,
        nms=True,
        half=False,
        imgsz=640,
    )

    # exported is the path Ultralytics wrote to; move/rename to OUT_PATH if different
    exported_path = Path(exported)
    if exported_path.resolve() != OUT_PATH.resolve():
        exported_path.replace(OUT_PATH)

    print("[OK] YOLO ONNX saved at:", OUT_PATH)


if __name__ == "__main__":
    main()
