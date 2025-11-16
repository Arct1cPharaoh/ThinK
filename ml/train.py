import json
from pathlib import Path
from typing import List, Tuple

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image
from tqdm.auto import tqdm
from ultralytics import YOLO


MIN_CALORIES = 0.0

DATA_ROOT = Path("data/mm_food_100k")
IMG_DIR = DATA_ROOT / "images"
META_PATH = DATA_ROOT / "meta.jsonl"

YOLO_MODEL_PATH = Path(
    "seg/runs_foodgroups/yolov8n_foodgroups/weights/best.pt"
)


# -----------------------
# Dataset / loading
# -----------------------
class LocalFoodDataset(Dataset):
    def __init__(self, meta: List[dict], transform=None):
        self.meta = meta
        self.transform = transform

    def __len__(self) -> int:
        return len(self.meta)

    def __getitem__(self, idx: int):
        ex = self.meta[idx]

        img_path = IMG_DIR / ex["filename"]
        img = Image.open(img_path).convert("RGB")
        if self.transform:
            img = self.transform(img)

        calories = float(ex["calories_kcal"])
        calories = max(calories, MIN_CALORIES)

        # NOTE: we now return image path so YOLO can run on original image
        return img, str(img_path), torch.tensor(calories, dtype=torch.float32)


def load_meta(meta_path: Path, max_samples: int | None = None) -> List[dict]:
    meta: List[dict] = []
    if not meta_path.is_file():
        raise FileNotFoundError(f"Metadata file not found: {meta_path}")

    with meta_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                ex = json.loads(line)
            except json.JSONDecodeError:
                continue
            meta.append(ex)

    cleaned: List[dict] = []
    for ex in meta:
        if ex.get("calories_kcal") is None:
            continue
        img_path = IMG_DIR / ex["filename"]
        if not img_path.is_file():
            continue
        cleaned.append(ex)

    if max_samples is not None:
        cleaned = cleaned[:max_samples]

    return cleaned


def build_transforms(train: bool):
    if train:
        return transforms.Compose(
            [
                transforms.Resize(256),
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
            ]
        )
    else:
        return transforms.Compose(
            [
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
            ]
        )


def build_datasets_from_meta(
    meta: List[dict],
    val_ratio: float = 0.1,
) -> Tuple[Dataset, Dataset]:
    n = len(meta)
    if n == 0:
        raise ValueError("No valid samples found in metadata.")

    perm = torch.randperm(n).tolist()
    val_size = max(1, int(n * val_ratio))
    train_size = n - val_size

    train_idx = perm[:train_size]
    val_idx = perm[train_size:]

    train_meta = [meta[i] for i in train_idx]
    val_meta = [meta[i] for i in val_idx]

    train_dataset = LocalFoodDataset(
        train_meta,
        transform=build_transforms(train=True),
    )
    val_dataset = LocalFoodDataset(
        val_meta,
        transform=build_transforms(train=False),
    )

    return train_dataset, val_dataset


def build_dataloaders(train_dataset, val_dataset, batch_size: int = 32):
    common_kwargs = dict(num_workers=0, pin_memory=False)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        **common_kwargs,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        **common_kwargs,
    )
    return train_loader, val_loader


# -----------------------
# Model: image + food-group vector
# -----------------------
class CalorieRegressorWithGroups(nn.Module):
    """
    EfficientNet-B0 backbone with a regression head that also
    conditions on a YOLO food-group vector (e.g. area fractions).
    Predicts log(calories + 1).
    """

    def __init__(self, n_groups: int):
        super().__init__()
        backbone = models.efficientnet_b0(
            weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1
        )
        in_features = backbone.classifier[1].in_features
        backbone.classifier = nn.Identity()

        self.backbone = backbone
        self.fc_reg = nn.Linear(in_features + n_groups, 1)

    def forward(self, x: torch.Tensor, group_vec: torch.Tensor) -> torch.Tensor:
        # x: (B, 3, H, W), group_vec: (B, n_groups)
        feats = self.backbone(x)                 # (B, in_features)
        z = torch.cat([feats, group_vec], dim=1) # (B, in_features + n_groups)
        log_kcal = self.fc_reg(z).squeeze(1)
        return log_kcal.clamp_min(0.0)

    @staticmethod
    def log_to_kcal(log_kcal: torch.Tensor) -> torch.Tensor:
        kcal = torch.expm1(log_kcal)
        return kcal.clamp_min(MIN_CALORIES)


def build_model(lr: float, n_groups: int):
    model = CalorieRegressorWithGroups(n_groups)
    reg_loss = nn.SmoothL1Loss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    return model, reg_loss, optimizer


# -----------------------
# YOLO-based food-group features
# -----------------------
def compute_group_vectors(detector: YOLO, img_paths: List[str], device) -> torch.Tensor:
    k = len(detector.names)

    # NOTE: disable per-image logging
    results = detector(img_paths, verbose=False)  # <--- add verbose=False

    group_vecs = []
    for res in results:
        h, w = res.orig_shape[:2]
        total_area = float(w * h) + 1e-6

        vec = torch.zeros(k, dtype=torch.float32)
        for box in res.boxes:
            cid = int(box.cls.item())
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            area = max(x2 - x1, 0.0) * max(y2 - y1, 0.0)
            frac = float(area / total_area)
            vec[cid] += frac

        vec = torch.clamp(vec, 0.0, 1.0)
        group_vecs.append(vec)

    group_vecs = torch.stack(group_vecs, dim=0).to(device)
    return group_vecs


# -----------------------
# Train / val
# -----------------------
def _step(
    model,
    imgs,
    img_paths,
    calories,
    reg_loss,
    device,
    detector: YOLO,
):
    imgs = imgs.to(device)
    calories = calories.to(device).clamp_min(MIN_CALORIES)

    # compute YOLO food-group features on the fly
    group_vecs = compute_group_vectors(detector, img_paths, device)

    target_log = torch.log1p(calories)
    pred_log = model(imgs, group_vecs)

    loss = reg_loss(pred_log, target_log)

    with torch.no_grad():
        pred_kcal = CalorieRegressorWithGroups.log_to_kcal(pred_log)
        mae = torch.abs(pred_kcal - calories).mean()

    return loss, mae


def train_one_epoch(
    model,
    train_loader,
    optimizer,
    reg_loss,
    device,
    detector: YOLO,
    epoch: int,
    num_epochs: int,
):
    model.train()
    total_loss = 0.0
    total_mae = 0.0
    n_train = len(train_loader.dataset)

    train_pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{num_epochs} [train]")
    for imgs, img_paths, calories in train_pbar:
        optimizer.zero_grad()

        loss, mae = _step(
            model,
            imgs,
            img_paths,
            calories,
            reg_loss,
            device,
            detector,
        )

        loss.backward()
        optimizer.step()

        batch_size = imgs.size(0)
        total_loss += loss.item() * batch_size
        total_mae += mae.item() * batch_size

        train_pbar.set_postfix(
            {
                "loss": f"{total_loss / n_train:.4f}",
                "mae_kcal": f"{total_mae / n_train:.2f}",
            }
        )

    return total_loss / n_train, total_mae / n_train


@torch.no_grad()
def validate(
    model,
    val_loader,
    reg_loss,
    device,
    detector: YOLO,
    epoch: int,
    num_epochs: int,
):
    model.eval()
    val_loss = 0.0
    val_mae = 0.0
    n_val = len(val_loader.dataset)

    val_pbar = tqdm(val_loader, desc=f"Epoch {epoch}/{num_epochs} [val]")
    for imgs, img_paths, calories in val_pbar:
        loss, mae = _step(
            model,
            imgs,
            img_paths,
            calories,
            reg_loss,
            device,
            detector,
        )

        batch_size = imgs.size(0)
        val_loss += loss.item() * batch_size
        val_mae += mae.item() * batch_size

        val_pbar.set_postfix(
            {
                "val_loss": f"{val_loss / n_val:.4f}",
                "val_mae_kcal": f"{val_mae / n_val:.2f}",
            }
        )

    return val_loss / n_val, val_mae / n_val


# -----------------------
# Entry point
# -----------------------
def train(
    num_epochs: int = 8,
    batch_size: int = 16,
    lr: float = 1e-4,
    max_samples: int | None = 10000,
    out_dir: Path = Path("artifacts"),
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("using", device)

    meta = load_meta(META_PATH, max_samples=max_samples)
    print(f"Loaded {len(meta)} local samples")

    # build YOLO once and reuse
    detector = YOLO(str(YOLO_MODEL_PATH))
    n_groups = len(detector.names)
    print("Using n_groups =", n_groups)

    train_dataset, val_dataset = build_datasets_from_meta(meta, val_ratio=0.1)
    train_loader, val_loader = build_dataloaders(train_dataset, val_dataset, batch_size)

    model, reg_loss, optimizer = build_model(lr, n_groups)
    model = model.to(device)

    for epoch in range(1, num_epochs + 1):
        train_loss, train_mae = train_one_epoch(
            model,
            train_loader,
            optimizer,
            reg_loss,
            device,
            detector,
            epoch,
            num_epochs,
        )
        print(
            f"\nEpoch {epoch}/{num_epochs} "
            f"train_loss={train_loss:.4f} train_mae_kcal={train_mae:.2f}"
        )

        val_loss, val_mae = validate(
            model,
            val_loader,
            reg_loss,
            device,
            detector,
            epoch,
            num_epochs,
        )
        print(f"  val_loss={val_loss:.4f} val_mae_kcal={val_mae:.2f}")

    out_dir.mkdir(parents=True, exist_ok=True)
    model_path = out_dir / "calorie_regressor_efficientnet_b0_groups.pt"
    torch.save(model.state_dict(), model_path)
    print("Saved model to", model_path)


if __name__ == "__main__":
    train()
