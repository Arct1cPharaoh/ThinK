import json
from io import BytesIO
from pathlib import Path
import requests
from datasets import load_dataset
from PIL import Image
from tqdm.auto import tqdm


DATASET_NAME = "Codatta/MM-Food-100K"
SPLIT = "train"

OUT_ROOT = Path("data/mm_food_100k")
OUT_IMG_DIR = OUT_ROOT / "images"
OUT_META_PATH = OUT_ROOT / "meta.jsonl"

OUT_IMG_DIR.mkdir(parents=True, exist_ok=True)


def extract_calories(ex):
    raw = ex.get("nutritional_profile")
    if isinstance(raw, str):
        try:
            return json.loads(raw).get("calories_kcal")
        except Exception:
            return None
    if isinstance(raw, dict):
        return raw.get("calories_kcal")
    return None


def main():
    ds = load_dataset(DATASET_NAME, split=SPLIT)

    with OUT_META_PATH.open("w", encoding="utf-8") as meta_file:
        for i, ex in enumerate(tqdm(ds, desc="Downloading MM-Food-100K")):
            url = ex.get("image_url")
            if not url:
                continue

            # download image
            try:
                r = requests.get(url, timeout=10)
                r.raise_for_status()
                img = Image.open(BytesIO(r.content)).convert("RGB")
            except Exception:
                continue

            # save image
            fname = f"{i:06d}.jpg"
            img.save(OUT_IMG_DIR / fname, format="JPEG")

            # write metadata line immediately
            entry = {
                "id": i,
                "filename": fname,
                "dish_name": ex.get("dish_name"),
                "calories_kcal": extract_calories(ex),
            }
            meta_file.write(json.dumps(entry) + "\n")
            meta_file.flush()


if __name__ == "__main__":
    main()
