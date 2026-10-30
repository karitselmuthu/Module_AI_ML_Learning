import json
import sys
from pathlib import Path

import pytest
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from diffusion_app import generate, review, validate_config
from vae import VAE, loss_parts, save_grid


def test_vae_shapes_and_nonnegative_losses(tmp_path):
    model = VAE(latent_dim=8)
    images = torch.rand(2, 3, 64, 64)
    reconstructed, mu, logvar = model(images)
    assert reconstructed.shape == images.shape
    assert mu.shape == logvar.shape == (2, 8)
    assert all(part.item() >= 0 for part in loss_parts(reconstructed, images, mu, logvar, 0.05))
    save_grid(model.sample(4, torch.device("cpu")), tmp_path / "samples.png", 2)
    assert Image.open(tmp_path / "samples.png").size == (128, 128)


class FakePipeline:
    def __init__(self):
        self.calls = []

    def __call__(self, **kwargs):
        self.calls.append(kwargs)
        color = (kwargs["num_inference_steps"] * 10, 0, 0)
        return type("Output", (), {"images": [Image.new("RGB", (32, 32), color)]})()


def test_generation_saves_settings_and_review(tmp_path):
    config = {"model_id": "example/test-model", "prompts": ["abstract support icon"],
              "seed": 5, "settings": [{"steps": 2, "guidance_scale": 1.0},
                                       {"steps": 4, "guidance_scale": 5.0}]}
    path = tmp_path / "config.json"
    path.write_text(json.dumps(config))
    fake = FakePipeline()
    run_dir = generate(path, tmp_path / "runs", pipeline=fake)
    records = json.loads((run_dir / "review.json").read_text())
    assert len(records) == 2
    assert all(row["seed"] == 5 and row["review"] == "pending" for row in records)
    assert fake.calls[0]["generator"].initial_seed() == fake.calls[1]["generator"].initial_seed()
    assert (run_dir / "contact_sheet.png").is_file()
    review(run_dir, records[0]["image"], "accepted", "Clear visual")
    assert json.loads((run_dir / "review.json").read_text())[0]["notes"] == "Clear visual"
    with pytest.raises(ValueError):
        review(run_dir, "../../outside.png", "accepted", "")


def test_invalid_config_rejected():
    with pytest.raises(ValueError, match="steps"):
        validate_config({"model_id": "x", "prompts": ["a"], "seed": 1,
                         "settings": [{"steps": 0, "guidance_scale": 7}]})
