#!/usr/bin/env python
"""
Fine-tune YOLOv8 on Food-11–style food-group dataset.

Assumes layout like:

  ../data/archive/
      training/
          Bread/
          Dairy product/
          Dessert/
          ...
      validation/
      evaluation/

We turn each image into a single YOLO box covering the whole image with
class = folder name (Bread, Dessert, etc.).

You need: pip install ultralytics pillow pyyaml
"""

import os
from pathlib import Path

from PIL import Image
from ultralytics import YOLO
import yaml


DATASET_ROOT = Path("..") / "data" / "archive"
YOLO_ROOT = DATASET_ROOT / "yolo_foodgroups"  # will be created


def load_categories():
    """Infer category names from training subfolders, return as list[id] = name."""
    train_root = DATASET_ROOT / "training"
    class_names = sorted(
        d
        for d in os.listdir(train_root)
        if (train_root / d).is_dir()
    )
    return class_names  # index in this list is class id


def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


def make_yolo_structure():
    for split in ("train", "val", "test"):
        for sub in ("images", "labels"):
            ensure_dir(YOLO_ROOT / split / sub)


def convert_to_yolo():
    """
    Build YOLOv8 detection-style dataset:
      - one box per image covering the whole image
      - class id from folder name
    """
    categories = load_categories()
    name_to_id = {name: i for i, name in enumerate(categories)}
    make_yolo_structure()

    split_map = {
        "training": "train",
        "validation": "val",
        "evaluation": "test",
    }

    exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}

    for src_split, dst_split in split_map.items():
        src_root = DATASET_ROOT / src_split
        if not src_root.is_dir():
            continue

        for class_name in os.listdir(src_root):
            class_dir = src_root / class_name
            if not class_dir.is_dir():
                continue

            cid = name_to_id.get(class_name)
            if cid is None:
                continue

            dst_img_root = YOLO_ROOT / dst_split / "images" / class_name
            dst_lbl_root = YOLO_ROOT / dst_split / "labels" / class_name
            ensure_dir(dst_img_root)
            ensure_dir(dst_lbl_root)

            for fname in os.listdir(class_dir):
                src_img_path = class_dir / fname
                if not src_img_path.is_file():
                    continue
                if src_img_path.suffix.lower() not in exts:
                    continue

                # open to get size
                try:
                    img = Image.open(src_img_path)
                    w, h = img.size
                except Exception:
                    continue

                # copy image
                dst_img_path = dst_img_root / fname
                if not dst_img_path.exists():
                    dst_img_path.write_bytes(src_img_path.read_bytes())

                # full-image box in YOLO format (normalized cx, cy, w, h)
                x_c, y_c, bw, bh = 0.5, 0.5, 1.0, 1.0
                lbl_lines = [f"{cid} {x_c:.6f} {y_c:.6f} {bw:.6f} {bh:.6f}"]

                dst_lbl_path = dst_lbl_root / (dst_img_path.stem + ".txt")
                with dst_lbl_path.open("w", encoding="utf-8") as f:
                    f.write("\n".join(lbl_lines))

    # data.yaml for YOLO
    data_yaml = {
        "path": str(YOLO_ROOT.resolve()),
        "train": "train/images",
        "val": "val/images",
        "test": "test/images",
        "names": categories,
    }
    with (YOLO_ROOT / "data.yaml").open("w", encoding="utf-8") as f:
        yaml.safe_dump(data_yaml, f)

    # quick sanity check
    train_imgs = list((YOLO_ROOT / "train" / "images").rglob("*.*"))
    print("num train images:", len(train_imgs))


def train_yolo():
    data_yaml_path = YOLO_ROOT / "data.yaml"
    model = YOLO("yolov8n.pt")  # detection model
    model.train(
        data=str(data_yaml_path),
        epochs=50,
        imgsz=640,
        batch=16,
        project="runs_foodgroups",
        name="yolov8n_foodgroups",
        exist_ok=True,  # won't delete old runs
    )


if __name__ == "__main__":
    convert_to_yolo()
    train_yolo()
