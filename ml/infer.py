import json
import random
from pathlib import Path

import torch
from PIL import Image
import matplotlib.pyplot as plt

from train import CalorieRegressor, build_transforms  # make sure this matches your train file name


DATA_ROOT = Path("data/mm_food_100k")
IMG_DIR = DATA_ROOT / "images"
META_PATH = DATA_ROOT / "meta.jsonl"
CHECKPOINT = Path("artifacts/calorie_regressor_efficientnet_b0.pt")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_model() -> CalorieRegressor:
    model = CalorieRegressor()
    state = torch.load(CHECKPOINT, map_location=DEVICE)
    model.load_state_dict(state)
    model.to(DEVICE)
    model.eval()
    return model


def load_meta():
    meta = []
    with META_PATH.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                ex = json.loads(line)
            except json.JSONDecodeError:
                continue
            meta.append(ex)

    # filter to entries that actually have calories and existing images
    clean = []
    for ex in meta:
        if ex.get("calories_kcal") is None:
            continue
        img_path = IMG_DIR / ex["filename"]
        if not img_path.is_file():
            continue
        clean.append(ex)

    if not clean:
        raise RuntimeError("No valid entries found in metadata.")

    return clean


def pick_random_example(meta):
    return random.choice(meta)


@torch.no_grad()
def predict_kcal(model: CalorieRegressor, img: Image.Image) -> float:
    tfm = build_transforms(train=False)
    x = tfm(img).unsqueeze(0).to(DEVICE)
    log_kcal = model(x)
    kcal = CalorieRegressor.log_to_kcal(log_kcal)[0].item()
    return float(kcal)


if __name__ == "__main__":
    model = load_model()
    meta = load_meta()
    ex = pick_random_example(meta)

    true_kcal = float(ex["calories_kcal"])
    img_path = IMG_DIR / ex["filename"]

    img = Image.open(img_path).convert("RGB")
    pred_kcal = predict_kcal(model, img)

    print(f"File: {img_path}")
    print(f"True calories:     {true_kcal:.2f} kcal")
    print(f"Predicted calories:{pred_kcal:.2f} kcal")

    plt.figure(figsize=(4, 6))
    plt.imshow(img)
    plt.axis("off")
    plt.title(f"Pred: {pred_kcal:.1f} kcal\nTrue: {true_kcal:.1f} kcal")
    plt.tight_layout()
    plt.show()
