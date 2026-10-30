"""Week 7: train a compact variational autoencoder on Module 2 images."""

from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parent
DEFAULT_MANIFEST = ROOT.parent / "Module 2 - Deep Learning Ticket Attachments" / "data" / "generated" / "manifest.csv"
DEFAULT_OUTPUT = ROOT / "artifacts" / "vae"
SIZE = 64


class AttachmentDataset(Dataset):
    def __init__(self, manifest: Path, split: str):
        self.root = manifest.parent
        with manifest.open(newline="") as stream:
            self.rows = [row for row in csv.DictReader(stream) if row["split"] == split]
        if not self.rows:
            raise ValueError(f"No {split} images in {manifest}")

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        with Image.open(self.root / self.rows[index]["path"]) as source:
            image = source.convert("RGB").resize((SIZE, SIZE), Image.Resampling.BILINEAR)
        return torch.from_numpy(np.asarray(image).copy()).permute(2, 0, 1).float() / 255.0


class VAE(nn.Module):
    def __init__(self, latent_dim: int = 16):
        super().__init__()
        self.latent_dim = latent_dim
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, 4, 2, 1), nn.ReLU(),
            nn.Conv2d(32, 64, 4, 2, 1), nn.ReLU(),
            nn.Conv2d(64, 128, 4, 2, 1), nn.ReLU(),
            nn.Flatten(),
        )
        self.mu = nn.Linear(128 * 8 * 8, latent_dim)
        self.logvar = nn.Linear(128 * 8 * 8, latent_dim)
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 128 * 8 * 8), nn.ReLU(), nn.Unflatten(1, (128, 8, 8)),
            nn.ConvTranspose2d(128, 64, 4, 2, 1), nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 4, 2, 1), nn.ReLU(),
            nn.ConvTranspose2d(32, 3, 4, 2, 1), nn.Sigmoid(),
        )

    def encode(self, x):
        features = self.encoder(x)
        return self.mu(features), self.logvar(features)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = mu + torch.randn_like(mu) * torch.exp(0.5 * logvar)
        return self.decoder(z), mu, logvar

    def sample(self, count: int, device: torch.device):
        return self.decoder(torch.randn(count, self.latent_dim, device=device))


def loss_parts(reconstruction, image, mu, logvar, beta: float):
    reconstruction_loss = F.mse_loss(reconstruction, image, reduction="sum") / image.shape[0]
    kl = -0.5 * torch.sum(1 + logvar - mu.square() - logvar.exp()) / image.shape[0]
    return reconstruction_loss + beta * kl, reconstruction_loss, kl


def save_grid(tensors: torch.Tensor, path: Path, columns: int):
    images = tensors.detach().cpu().clamp(0, 1)
    rows = (len(images) + columns - 1) // columns
    canvas = Image.new("RGB", (SIZE * columns, SIZE * rows), "white")
    for index, tensor in enumerate(images):
        array = (tensor.permute(1, 2, 0).numpy() * 255).round().astype(np.uint8)
        canvas.paste(Image.fromarray(array), ((index % columns) * SIZE, (index // columns) * SIZE))
    canvas.save(path)


def train(manifest: Path, output: Path, epochs: int, batch_size: int, beta: float, seed: int):
    if epochs < 1 or batch_size < 1 or beta < 0:
        raise ValueError("epochs and batch size must be positive; beta must be nonnegative")
    torch.set_num_threads(min(4, torch.get_num_threads()))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    train_set = AttachmentDataset(manifest, "train")
    validation_set = AttachmentDataset(manifest, "validation")
    test_set = AttachmentDataset(manifest, "test")
    training = DataLoader(train_set, batch_size=batch_size, shuffle=True, generator=torch.Generator().manual_seed(seed))
    validation = DataLoader(validation_set, batch_size=batch_size)
    model = VAE().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    output.mkdir(parents=True, exist_ok=True)
    history = []
    best = float("inf")
    for epoch in range(1, epochs + 1):
        row = {"epoch": epoch}
        for split, loader in (("train", training), ("validation", validation)):
            model.train(split == "train")
            totals = np.zeros(3)
            for images in loader:
                images = images.to(device)
                with torch.set_grad_enabled(split == "train"):
                    reconstruction, mu, logvar = model(images)
                    losses = loss_parts(reconstruction, images, mu, logvar, beta)
                    if split == "train":
                        optimizer.zero_grad(set_to_none=True)
                        losses[0].backward()
                        optimizer.step()
                totals += np.array([part.item() for part in losses]) * len(images)
            row.update({f"{split}_{name}": float(value / len(loader.dataset))
                        for name, value in zip(("loss", "reconstruction", "kl"), totals)})
        history.append(row)
        print(f"epoch {epoch}: train={row['train_loss']:.3f} validation={row['validation_loss']:.3f}")
        if row["validation_loss"] < best:
            best = row["validation_loss"]
            torch.save({"state_dict": model.state_dict(), "latent_dim": model.latent_dim,
                        "seed": seed, "beta": beta}, output / "model.pt")
    with (output / "history.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=history[0].keys())
        writer.writeheader()
        writer.writerows(history)
    checkpoint = torch.load(output / "model.pt", map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    original = torch.stack([validation_set[index] for index in range(8)]).to(device)
    with torch.no_grad():
        mu, _ = model.encode(original)
        reconstructed = model.decoder(mu)
        torch.manual_seed(seed + 1)
        sampled = model.sample(16, device)
        test_losses = []
        for images in DataLoader(test_set, batch_size=batch_size):
            images = images.to(device)
            reconstruction, mu, logvar = model(images)
            test_losses.append((loss_parts(reconstruction, images, mu, logvar, beta)[0].item(), len(images)))
    save_grid(torch.cat((original, reconstructed)), output / "reconstructions.png", columns=8)
    save_grid(sampled, output / "samples.png", columns=4)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot([r["epoch"] for r in history], [r["train_loss"] for r in history], label="Train")
    ax.plot([r["epoch"] for r in history], [r["validation_loss"] for r in history], label="Validation")
    ax.set(xlabel="Epoch", ylabel="VAE loss", title="Reconstruction + beta × KL")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output / "training_curves.png", dpi=150)
    plt.close(fig)
    metrics = {"best_validation_loss": best, "test_loss": sum(v*n for v,n in test_losses)/len(test_set),
               "train_images": len(train_set), "validation_images": len(validation_set),
               "test_images": len(test_set), "beta": beta, "seed": seed, "epochs": epochs,
               "dataset": str(manifest.resolve())}
    (output / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--beta", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    print(json.dumps(train(args.manifest, args.output, args.epochs, args.batch_size, args.beta, args.seed), indent=2))
