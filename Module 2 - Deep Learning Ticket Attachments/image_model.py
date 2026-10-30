"""Train and run an image classifier for simulated support attachments."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms

from generate_data import DEFAULT_OUTPUT, LABELS, ROOT

SEED = 42
IMAGE_SIZE = 128
ARTIFACTS = ROOT / "artifacts"


def set_seed(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(2)


def image_transform(arch: str, training: bool = False):
    steps = [transforms.Resize((IMAGE_SIZE, IMAGE_SIZE))]
    if training:
        steps.extend([
            transforms.RandomAffine(degrees=2, translate=(0.025, 0.025)),
            transforms.ColorJitter(brightness=0.08, contrast=0.08),
        ])
    steps.append(transforms.ToTensor())
    if arch == "mobilenet":
        steps.append(transforms.Normalize(mean=(0.485, 0.456, 0.406),
                                          std=(0.229, 0.224, 0.225)))
    return transforms.Compose(steps)


class AttachmentDataset(Dataset):
    def __init__(self, manifest: Path, split: str, arch: str, training: bool = False):
        self.manifest = Path(manifest)
        frame = pd.read_csv(manifest)
        required = {"path", "label", "split", "scenario_id"}
        if not required.issubset(frame.columns):
            raise ValueError(f"Manifest needs columns {sorted(required)}")
        self.frame = frame[frame["split"] == split].reset_index(drop=True)
        if len(self.frame) == 0:
            raise ValueError(f"No rows for split {split}")
        self.transform = image_transform(arch, training)

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, index: int):
        row = self.frame.iloc[index]
        image_path = self.manifest.parent / row["path"]
        with Image.open(image_path) as image:
            tensor = self.transform(image.convert("RGB"))
        return tensor, LABELS.index(row["label"]), row["path"]


class SmallCNN(nn.Module):
    def __init__(self, classes: int = 3):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1), nn.GroupNorm(4, 16), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.GroupNorm(4, 32), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.GroupNorm(8, 64), nn.ReLU(), nn.MaxPool2d(2),
            nn.AdaptiveAvgPool2d((4, 4)),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(), nn.Linear(64 * 16, 64), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(64, classes),
        )

    def forward(self, images):
        return self.classifier(self.features(images))


def make_model(arch: str, pretrained: bool = False) -> nn.Module:
    if arch == "scratch":
        return SmallCNN(len(LABELS))
    if arch == "mobilenet":
        if pretrained:
            torch.hub.set_dir(str(ARTIFACTS / "torch"))
        weights = models.MobileNet_V3_Small_Weights.DEFAULT if pretrained else None
        model = models.mobilenet_v3_small(weights=weights)
        if pretrained:
            for parameter in model.features.parameters():
                parameter.requires_grad = False
        model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, len(LABELS))
        return model
    raise ValueError(f"Unknown architecture: {arch}")


def run_epoch(model, loader, criterion, optimizer=None):
    training = optimizer is not None
    model.train(training)
    if training and hasattr(model, "features") and not any(
        parameter.requires_grad for parameter in model.features.parameters()
    ):
        model.features.eval()  # Keep pretrained BatchNorm statistics fixed.
    losses, actual, predicted = [], [], []
    for images, labels, _ in loader:
        if training:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(training):
            logits = model(images)
            loss = criterion(logits, labels)
            if training:
                loss.backward()
                optimizer.step()
        losses.append(float(loss.item()) * len(labels))
        actual.extend(labels.tolist())
        predicted.extend(logits.argmax(dim=1).tolist())
    return {
        "loss": sum(losses) / len(actual),
        "accuracy": float(np.mean(np.array(actual) == np.array(predicted))),
        "macro_f1": float(f1_score(actual, predicted, average="macro", zero_division=0)),
    }


def evaluate(model, loader):
    model.eval()
    actual, predicted, paths = [], [], []
    with torch.no_grad():
        for images, labels, batch_paths in loader:
            logits = model(images)
            actual.extend(labels.tolist())
            predicted.extend(logits.argmax(dim=1).tolist())
            paths.extend(batch_paths)
    return {
        "macro_f1": round(float(f1_score(actual, predicted, average="macro", zero_division=0)), 4),
        "classification_report": classification_report(
            actual, predicted, labels=list(range(len(LABELS))), target_names=LABELS,
            output_dict=True, zero_division=0,
        ),
        "confusion_matrix": confusion_matrix(actual, predicted, labels=list(range(len(LABELS)))).tolist(),
        "predictions": pd.DataFrame({
            "path": paths,
            "label": [LABELS[index] for index in actual],
            "predicted": [LABELS[index] for index in predicted],
            "correct": [a == p for a, p in zip(actual, predicted)],
        }),
    }


def plot_curves(history: pd.DataFrame, path: Path) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(10, 4))
    for split in ("train", "validation"):
        axes[0].plot(history["epoch"], history[f"{split}_loss"], label=split)
        axes[1].plot(history["epoch"], history[f"{split}_macro_f1"], label=split)
    axes[0].set_title("Cross-entropy loss")
    axes[1].set_title("Macro F1")
    for axis in axes:
        axis.set_xlabel("Epoch")
        axis.legend()
    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)


def train(manifest: Path = DEFAULT_OUTPUT / "manifest.csv", arch: str = "scratch",
          output_dir: Path | None = None, epochs: int = 25, batch_size: int = 16,
          learning_rate: float | None = None) -> dict:
    if epochs < 1 or batch_size < 1:
        raise ValueError("epochs and batch size must be positive")
    set_seed()
    manifest = Path(manifest)
    output_dir = Path(output_dir) if output_dir else ARTIFACTS / arch
    train_data = AttachmentDataset(manifest, "train", arch, training=True)
    valid_data = AttachmentDataset(manifest, "validation", arch)
    test_data = AttachmentDataset(manifest, "test", arch)
    split_frames = [data.frame for data in (train_data, valid_data, test_data)]
    scenario_sets = [set(frame["scenario_id"]) for frame in split_frames]
    if any(scenario_sets[i] & scenario_sets[j] for i, j in ((0, 1), (0, 2), (1, 2))):
        raise ValueError("Scenario IDs overlap across splits")
    loaders = [DataLoader(data, batch_size=batch_size, shuffle=(index == 0), num_workers=0)
               for index, data in enumerate((train_data, valid_data, test_data))]
    model = make_model(arch, pretrained=(arch == "mobilenet"))
    rate = learning_rate if learning_rate is not None else (0.0003 if arch == "scratch" else 0.0005)
    optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad),
                                  lr=rate, weight_decay=0.001)
    criterion = nn.CrossEntropyLoss()
    history, best_score, best_epoch, best_state, stale = [], -1.0, 0, None, 0
    for epoch in range(1, epochs + 1):
        train_metrics = run_epoch(model, loaders[0], criterion, optimizer)
        valid_metrics = run_epoch(model, loaders[1], criterion)
        history.append({"epoch": epoch, **{f"train_{key}": value for key, value in train_metrics.items()},
                        **{f"validation_{key}": value for key, value in valid_metrics.items()}})
        if valid_metrics["macro_f1"] > best_score + 1e-6:
            best_score, best_epoch, stale = valid_metrics["macro_f1"], epoch, 0
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        else:
            stale += 1
        if stale >= 10:
            break
    model.load_state_dict(best_state)
    test_result = evaluate(model, loaders[2])
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.save({"arch": arch, "classes": LABELS, "state_dict": model.state_dict(),
                "image_size": IMAGE_SIZE, "pretrained_features": arch == "mobilenet"},
               output_dir / "model.pt")
    pd.DataFrame(history).to_csv(output_dir / "history.csv", index=False)
    plot_curves(pd.DataFrame(history), output_dir / "training_curves.png")
    test_result.pop("predictions").to_csv(output_dir / "test_predictions.csv", index=False)
    report = {"arch": arch, "pretrained_features": arch == "mobilenet", "seed": SEED,
              "split_rows": {name: len(data) for name, data in zip(("train", "validation", "test"),
                                                                    (train_data, valid_data, test_data))},
              "best_epoch": best_epoch, "best_validation_macro_f1": round(best_score, 4),
              "test": test_result,
              "limitation": "Synthetic attachment images only; test score is not a production estimate."}
    (output_dir / "metrics.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def predict(model_path: Path, image_path: Path) -> dict:
    checkpoint = torch.load(model_path, map_location="cpu", weights_only=True)
    model = make_model(checkpoint["arch"], pretrained=False)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    with Image.open(image_path) as image:
        tensor = image_transform(checkpoint["arch"])(image.convert("RGB")).unsqueeze(0)
    with torch.no_grad():
        probabilities = torch.softmax(model(tensor), dim=1)[0].tolist()
    classes = list(checkpoint["classes"])
    return {"label": classes[int(np.argmax(probabilities))],
            "probabilities": {label: round(float(probability), 4)
                              for label, probability in zip(classes, probabilities)}}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    training = commands.add_parser("train")
    training.add_argument("--manifest", type=Path, default=DEFAULT_OUTPUT / "manifest.csv")
    training.add_argument("--arch", choices=("scratch", "mobilenet"), default="scratch")
    training.add_argument("--output", type=Path)
    training.add_argument("--epochs", type=int, default=25)
    training.add_argument("--batch-size", type=int, default=16)
    training.add_argument("--lr", type=float)
    inference = commands.add_parser("predict")
    inference.add_argument("image", type=Path)
    inference.add_argument("--model", type=Path, default=ARTIFACTS / "scratch" / "model.pt")
    args = parser.parse_args()
    if args.command == "train":
        print(json.dumps(train(args.manifest, args.arch, args.output, args.epochs,
                               args.batch_size, args.lr), indent=2))
    else:
        print(json.dumps(predict(args.model, args.image), indent=2))


if __name__ == "__main__":
    main()
