import json
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms, models
from tqdm.auto import tqdm

from data import load_mm_food


class FoodTorchDataset(Dataset):
    def __init__(self, hf_dataset, label2idx, transform=None):
        self.ds = hf_dataset
        self.label2idx = label2idx
        self.transform = transform
        self.indices = [
            i
            for i, ex in enumerate(self.ds)
            if ex.get("dish_name") and ex.get("calories_kcal") is not None
        ]

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        ex = self.ds[self.indices[idx]]
        img = ex["image"]
        if self.transform:
            img = self.transform(img)
        label = self.label2idx[ex["dish_name"]]
        calories = float(ex["calories_kcal"])
        return (
            img,
            torch.tensor(label, dtype=torch.long),
            torch.tensor(calories, dtype=torch.float32),
        )


class FoodModel(nn.Module):
    def __init__(self, num_classes: int):
        super().__init__()
        backbone = models.efficientnet_b0(
            weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1
        )
        in_features = backbone.classifier[1].in_features
        backbone.classifier = nn.Identity()
        self.backbone = backbone
        self.fc_cls = nn.Linear(in_features, num_classes)
        self.fc_reg = nn.Linear(in_features, 1)

    def forward(self, x):
        feats = self.backbone(x)
        logits = self.fc_cls(feats)
        calories = self.fc_reg(feats).squeeze(1)
        return logits, calories


def build_label_mapping(ds, out_dir: Path):
    dish_names = sorted(
        list(
            {
                ex["dish_name"]
                for ex in tqdm(ds, desc="Collect dish names")
                if ex.get("dish_name")
            }
        )
    )
    label2idx = {name: i for i, name in enumerate(dish_names)}
    idx2label = {i: name for name, i in label2idx.items()}

    out_dir.mkdir(exist_ok=True, parents=True)
    with (out_dir / "label_mapping.json").open("w") as f:
        json.dump({"label2idx": label2idx, "idx2label": idx2label}, f)

    return label2idx, idx2label


def build_dataloaders(ds_train_hf, label2idx, batch_size=32, val_ratio=0.1):
    transform = transforms.Compose(
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

    full_dataset = FoodTorchDataset(ds_train_hf, label2idx, transform=transform)

    val_size = int(len(full_dataset) * val_ratio)
    train_size = len(full_dataset) - val_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True
    )

    return train_loader, val_loader


def train_one_epoch(
    model,
    train_loader,
    optimizer,
    ce_loss,
    mse_loss,
    device,
    reg_lambda: float,
    epoch: int,
    num_epochs: int,
):
    model.train()
    total_loss = 0.0
    total_cls_loss = 0.0
    total_reg_loss = 0.0
    n_train = len(train_loader.dataset)

    train_pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{num_epochs} [train]")
    for imgs, labels, calories in train_pbar:
        imgs = imgs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        calories = calories.to(device, non_blocking=True)

        optimizer.zero_grad()
        logits, cal_pred = model(imgs)

        loss_cls = ce_loss(logits, labels)
        loss_reg = mse_loss(cal_pred, calories)
        loss = loss_cls + reg_lambda * loss_reg

        loss.backward()
        optimizer.step()

        batch_size = imgs.size(0)
        total_loss += loss.item() * batch_size
        total_cls_loss += loss_cls.item() * batch_size
        total_reg_loss += loss_reg.item() * batch_size

        train_pbar.set_postfix(
            {
                "loss": f"{total_loss / n_train:.4f}",
                "cls": f"{total_cls_loss / n_train:.4f}",
                "reg": f"{total_reg_loss / n_train:.4f}",
            }
        )

    return (
        total_loss / n_train,
        total_cls_loss / n_train,
        total_reg_loss / n_train,
    )


def validate(model, val_loader, ce_loss, mse_loss, device, reg_lambda: float, epoch, num_epochs):
    model.eval()
    val_loss = 0.0
    correct = 0
    total = 0
    n_val = len(val_loader.dataset)

    val_pbar = tqdm(val_loader, desc=f"Epoch {epoch}/{num_epochs} [val]")
    with torch.no_grad():
        for imgs, labels, calories in val_pbar:
            imgs = imgs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            calories = calories.to(device, non_blocking=True)

            logits, cal_pred = model(imgs)
            loss_cls = ce_loss(logits, labels)
            loss_reg = mse_loss(cal_pred, calories)
            loss = loss_cls + reg_lambda * loss_reg

            batch_size = imgs.size(0)
            val_loss += loss.item() * batch_size

            preds = logits.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

            val_pbar.set_postfix(
                {
                    "val_loss": f"{val_loss / n_val:.4f}",
                    "val_acc": f"{correct / total:.4f}",
                }
            )

    return val_loss / n_val, correct / total


def train(
    num_epochs: int = 1,
    batch_size: int = 32,
    lr: float = 1e-4,
    reg_lambda: float = 0.1,
    out_dir: Path = Path("artifacts"),
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ds_train_hf = load_mm_food(split="train")
    max_samples = 5000
    ds_train_hf = ds_train_hf.select(range(min(max_samples, len(ds_train_hf))))
    label2idx, _ = build_label_mapping(ds_train_hf, out_dir)

    train_loader, val_loader = build_dataloaders(
        ds_train_hf, label2idx, batch_size=batch_size
    )

    model = FoodModel(num_classes=len(label2idx)).to(device)
    ce_loss = nn.CrossEntropyLoss()
    mse_loss = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(1, num_epochs + 1):
        train_loss, train_cls, train_reg = train_one_epoch(
            model,
            train_loader,
            optimizer,
            ce_loss,
            mse_loss,
            device,
            reg_lambda,
            epoch,
            num_epochs,
        )
        print(
            f"\nEpoch {epoch}/{num_epochs} "
            f"train_loss={train_loss:.4f} cls={train_cls:.4f} reg={train_reg:.4f}"
        )

        val_loss, val_acc = validate(
            model,
            val_loader,
            ce_loss,
            mse_loss,
            device,
            reg_lambda,
            epoch,
            num_epochs,
        )
        print(f"  val_loss={val_loss:.4f} val_acc={val_acc:.4f}")

    model_path = out_dir / "food_model_efficientnet_b0.pt"
    torch.save(model.state_dict(), model_path)
    print("Saved model to", model_path)


if __name__ == "__main__":
    train()
