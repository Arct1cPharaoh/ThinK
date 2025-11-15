import json
from typing import Optional

from datasets import load_dataset, Dataset, DatasetDict, Image


DATASET_NAME = "Codatta/MM-Food-100K"


def _parse_nutrition_and_portion(example: dict) -> dict:
    # nutritional_profile: '{"fat_g":25.0,"protein_g":30.0,"calories_kcal":400,"carbohydrate_g":15.0}'
    try:
        prof = json.loads(example.get("nutritional_profile", "{}"))
    except Exception:
        prof = {}

    example["calories_kcal"] = prof.get("calories_kcal")
    example["protein_g"] = prof.get("protein_g")
    example["fat_g"] = prof.get("fat_g")
    example["carbohydrate_g"] = prof.get("carbohydrate_g")

    # portion_size: '["chicken:300g"]'
    example["portion_grams"] = None
    try:
        portions = json.loads(example.get("portion_size", "[]"))
        if portions:
            first = portions[0]
            if ":" in first and first.endswith("g"):
                grams_str = first.split(":", 1)[1].strip().rstrip("g")
                example["portion_grams"] = float(grams_str)
    except Exception:
        pass

    return example


def _prepare_split(split: Dataset) -> Dataset:
    split = split.rename_column("image_url", "image")
    split = split.cast_column("image", Image())
    split = split.map(_parse_nutrition_and_portion)
    return split


def load_mm_food(
    split: Optional[str] = None,
    **load_kwargs,
) -> Dataset | DatasetDict:
    """
    Load MM-Food-100K with:
      - image_url -> image (PIL Image via datasets.Image)
      - parsed nutrition fields (calories_kcal, protein_g, fat_g, carbohydrate_g)
      - parsed portion_grams (best-effort from portion_size)
    """
    ds: DatasetDict = load_dataset(DATASET_NAME, **load_kwargs)

    for key in ds.keys():
        ds[key] = _prepare_split(ds[key])

    if split is not None:
        return ds[split]
    return ds


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    ds_train: Dataset = load_mm_food(split="train")
    df = ds_train.to_pandas()

    print("Columns:", df.columns.tolist())
    print("\nNutrition summary:")
    print(df[["calories_kcal", "protein_g", "fat_g", "carbohydrate_g", "portion_grams"]].describe())

    # Histogram of calories
    plt.figure()
    df["calories_kcal"].dropna().hist(bins=50)
    plt.xlabel("Calories (kcal)")
    plt.ylabel("Count")
    plt.title("MM-Food-100K: Calories distribution")
    plt.tight_layout()
    plt.show()

    # Top 20 most common dishes
    top_dishes = df["dish_name"].value_counts().head(20)

    plt.figure()
    top_dishes.sort_values().plot(kind="barh")
    plt.xlabel("Count")
    plt.ylabel("Dish name")
    plt.title("MM-Food-100K: Top 20 dishes")
    plt.tight_layout()
    plt.show()
