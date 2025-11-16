#!/usr/bin/env python
import os
from collections import Counter, defaultdict

import matplotlib.pyplot as plt


DATASET_DIR = os.path.join("..", "data", "archive")


def load_categories(train_dir=os.path.join(DATASET_DIR, "training")):
    """
    Infer category ids from subdirectories in the training split.

    Returns:
        dict[int, str]: category_id -> category_name
    """
    cat_id_to_name = {}
    if not os.path.isdir(train_dir):
        return cat_id_to_name

    # stable ordering
    class_names = sorted(
        d for d in os.listdir(train_dir)
        if os.path.isdir(os.path.join(train_dir, d))
    )
    for cid, name in enumerate(class_names):
        cat_id_to_name[cid] = name
    return cat_id_to_name


def load_splits(dataset_dir=DATASET_DIR):
    """
    Build image id sets per split from folder structure.

    Returns:
        dict[str, set[str]]: split_name -> {image_id, ...}
    """
    splits = {}
    mapping = {
        "training": "train",
        "validation": "val",
        "evaluation": "test",
    }

    for split_folder, split_name in mapping.items():
        root = os.path.join(dataset_dir, split_folder)
        if not os.path.isdir(root):
            continue

        ids = set()
        for class_name in os.listdir(root):
            class_dir = os.path.join(root, class_name)
            if not os.path.isdir(class_dir):
                continue
            for fname in os.listdir(class_dir):
                fpath = os.path.join(class_dir, fname)
                if not os.path.isfile(fpath):
                    continue
                # image id: split/class/file
                rel_id = os.path.join(split_name, class_name, fname)
                ids.add(rel_id)
        splits[split_name] = ids

    return splits


def load_annotations(dataset_dir=DATASET_DIR):
    """
    For this classification-style dataset, treat each image as having
    exactly one label (its class folder).

    Returns:
        class_counts: Counter(category_id -> #images)
        boxes_per_image: Counter(image_id -> 1)
        img_to_cats: dict[image_id] -> set{category_id}
    """
    class_counts = Counter()
    boxes_per_image = Counter()
    img_to_cats = defaultdict(set)

    cat_id_to_name = load_categories()
    name_to_id = {name: cid for cid, name in cat_id_to_name.items()}

    mapping = {
        "training": "train",
        "validation": "val",
        "evaluation": "test",
    }

    for split_folder, split_name in mapping.items():
        root = os.path.join(dataset_dir, split_folder)
        if not os.path.isdir(root):
            continue

        for class_name in os.listdir(root):
            class_dir = os.path.join(root, class_name)
            if not os.path.isdir(class_dir):
                continue
            cid = name_to_id.get(class_name)
            if cid is None:
                continue

            for fname in os.listdir(class_dir):
                fpath = os.path.join(class_dir, fname)
                if not os.path.isfile(fpath):
                    continue
                img_id = os.path.join(split_name, class_name, fname)

                class_counts[cid] += 1
                boxes_per_image[img_id] += 1  # always 1 for classification
                img_to_cats[img_id].add(cid)

    return class_counts, boxes_per_image, img_to_cats


def plot_split_sizes(splits):
    names = []
    counts = []
    for split_name in ["train", "val", "test"]:
        if split_name in splits:
            names.append(split_name)
            counts.append(len(splits[split_name]))

    if not names:
        return

    plt.figure()
    plt.bar(names, counts)
    plt.title("Number of images per split")
    plt.xlabel("Split")
    plt.ylabel("Images")


def plot_class_distribution(class_counts, cat_id_to_name, top_k=20):
    if not class_counts:
        return

    most_common = class_counts.most_common(top_k)
    ids = [cid for cid, _ in most_common]
    names = [cat_id_to_name.get(cid, str(cid)) for cid in ids]
    counts = [class_counts[cid] for cid in ids]

    plt.figure(figsize=(10, 5))
    plt.bar(range(len(names)), counts)
    plt.xticks(range(len(names)), names, rotation=45, ha="right")
    plt.title(f"Top {top_k} categories by number of images")
    plt.xlabel("Category")
    plt.ylabel("Image count")
    plt.tight_layout()


def plot_boxes_per_image_hist(boxes_per_image):
    if not boxes_per_image:
        return

    counts = list(boxes_per_image.values())
    plt.figure()
    plt.hist(counts, bins=range(1, max(counts) + 2))
    plt.title("Distribution of labels per image")
    plt.xlabel("Labels per image")
    plt.ylabel("Number of images")


if __name__ == "__main__":
    cat_id_to_name = load_categories()
    splits = load_splits()
    class_counts, boxes_per_image, img_to_cats = load_annotations()

    plot_split_sizes(splits)
    plot_class_distribution(class_counts, cat_id_to_name, top_k=20)
    plot_boxes_per_image_hist(boxes_per_image)

    plt.show()
