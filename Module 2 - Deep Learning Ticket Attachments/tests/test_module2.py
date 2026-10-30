from hashlib import sha256
from pathlib import Path

import pandas as pd
import pytest
import torch
from PIL import Image

from generate_data import LABELS, generate
from image_model import SmallCNN, make_model, predict as predict_image
from text_attention import TicketTransformer, encode, make_vocab, predict as predict_text


def test_generated_scenarios_are_separate_and_reproducible(tmp_path):
    manifest = generate(tmp_path / "first")
    frame = pd.read_csv(manifest)
    assert len(frame) == 240
    assert set(frame["label"]) == set(LABELS)
    assert frame.groupby("split").size().to_dict() == {"train": 144, "validation": 48, "test": 48}
    groups = [set(frame.loc[frame["split"] == split, "scenario_id"])
              for split in ("train", "validation", "test")]
    assert not (groups[0] & groups[1] or groups[0] & groups[2] or groups[1] & groups[2])
    first_image = manifest.parent / frame.iloc[0]["path"]
    assert Image.open(first_image).size == (224, 160)
    second_manifest = generate(tmp_path / "second")
    second_image = second_manifest.parent / frame.iloc[0]["path"]
    assert sha256(first_image.read_bytes()).digest() == sha256(second_image.read_bytes()).digest()


def test_image_models_and_saved_inference(tmp_path):
    scratch = SmallCNN()
    assert tuple(scratch(torch.rand(2, 3, 128, 128)).shape) == (2, 3)
    transfer = make_model("mobilenet", pretrained=False)
    assert tuple(transfer(torch.rand(2, 3, 128, 128)).shape) == (2, 3)
    model_path = tmp_path / "model.pt"
    image_path = tmp_path / "image.png"
    torch.save({"arch": "scratch", "classes": LABELS, "state_dict": scratch.state_dict(),
                "image_size": 128}, model_path)
    Image.new("RGB", (224, 160), "white").save(image_path)
    result = predict_image(model_path, image_path)
    assert result["label"] in LABELS
    assert sum(result["probabilities"].values()) == pytest.approx(1, abs=0.0002)


def test_attention_masks_padding_and_rejects_blank_inference(tmp_path):
    vocab = make_vocab(["invoice paid", "reset password"])
    assert "unseen" not in vocab
    encoded = encode(["invoice unseen", "reset"], vocab, max_len=5)
    assert encoded.tolist()[0] == [vocab["invoice"], 1, 0, 0, 0]
    model = TicketTransformer(len(vocab), max_len=5)
    assert tuple(model(encoded).shape) == (2, 3)
    checkpoint = tmp_path / "model.pt"
    torch.save({"state_dict": model.state_dict(), "vocab": vocab,
                "classes": LABELS, "max_len": 5}, checkpoint)
    with pytest.raises(ValueError, match="must not be blank"):
        predict_text(checkpoint, "  ")
