"""Week 6 lab: compare a tiny self-attention ticket model with TF-IDF."""

from __future__ import annotations

import argparse
import json
import random
import re
from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from generate_data import LABELS, ROOT

MODULE1 = ROOT.parent / "Module 1 - Machine Learning Ticket Classifier"
LOCAL_TICKETS = MODULE1 / "data" / "tickets.csv"
DEFAULT_DATA = LOCAL_TICKETS if LOCAL_TICKETS.is_file() else MODULE1 / "sample_data" / "tickets.csv"
ARTIFACTS = ROOT / "artifacts" / "text_attention"
SEED = 42
MAX_LEN = 24
TOKEN_PATTERN = re.compile(r"[a-z0-9]+")


def tokens(text: str) -> list[str]:
    return TOKEN_PATTERN.findall(text.lower())


def make_vocab(texts) -> dict[str, int]:
    counts = Counter(token for text in texts for token in tokens(text))
    return {"<pad>": 0, "<unk>": 1, **{
        token: index + 2 for index, (token, _) in enumerate(sorted(counts.items()))
    }}


def encode(texts, vocab: dict[str, int], max_len: int = MAX_LEN) -> torch.Tensor:
    rows = []
    for text in texts:
        ids = [vocab.get(token, 1) for token in tokens(text)[:max_len]]
        rows.append(ids + [0] * (max_len - len(ids)))
    return torch.tensor(rows, dtype=torch.long)


class TicketTransformer(nn.Module):
    def __init__(self, vocab_size: int, classes: int = 3, max_len: int = MAX_LEN):
        super().__init__()
        self.word_embedding = nn.Embedding(vocab_size, 32, padding_idx=0)
        self.position_embedding = nn.Embedding(max_len, 32)
        layer = nn.TransformerEncoderLayer(
            d_model=32, nhead=4, dim_feedforward=64, dropout=0.2, batch_first=True
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=1, enable_nested_tensor=False)
        self.classifier = nn.Sequential(nn.Dropout(0.2), nn.Linear(32, classes))

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        padding = ids.eq(0)
        positions = torch.arange(ids.size(1), device=ids.device).unsqueeze(0)
        embedded = self.word_embedding(ids) + self.position_embedding(positions)
        encoded = self.encoder(embedded, src_key_padding_mask=padding)
        present = (~padding).unsqueeze(-1)
        pooled = (encoded * present).sum(dim=1) / present.sum(dim=1).clamp(min=1)
        return self.classifier(pooled)


def split_tickets(frame: pd.DataFrame):
    train_valid, test = train_test_split(
        frame, test_size=0.2, random_state=SEED, stratify=frame["label"]
    )
    train, valid = train_test_split(
        train_valid, test_size=0.25, random_state=SEED, stratify=train_valid["label"]
    )
    return train, valid, test


def score(truth, predicted) -> dict:
    return {
        "macro_f1": round(float(f1_score(truth, predicted, average="macro", zero_division=0)), 4),
        "classification_report": classification_report(
            truth, predicted, labels=LABELS, output_dict=True, zero_division=0
        ),
        "confusion_matrix": confusion_matrix(truth, predicted, labels=LABELS).tolist(),
    }


def train(data_path: Path = DEFAULT_DATA, output_dir: Path = ARTIFACTS,
          epochs: int = 25) -> dict:
    if epochs < 1:
        raise ValueError("epochs must be positive")
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.set_num_threads(2)
    frame = pd.read_csv(data_path)[["text", "label"]].dropna().drop_duplicates("text")
    if set(frame["label"]) != set(LABELS):
        raise ValueError("Expected account, billing, and technical labels")
    train_frame, valid_frame, test_frame = split_tickets(frame)
    vocab = make_vocab(train_frame["text"])
    train_x, valid_x, test_x = [encode(part["text"], vocab) for part in
                              (train_frame, valid_frame, test_frame)]
    train_y, valid_y, test_y = [torch.tensor([LABELS.index(label) for label in part["label"]])
                                  for part in (train_frame, valid_frame, test_frame)]
    loader = DataLoader(TensorDataset(train_x, train_y), batch_size=16, shuffle=True)
    model = TicketTransformer(len(vocab))
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.003, weight_decay=0.01)
    criterion = nn.CrossEntropyLoss()
    history, best_f1, best_epoch, best_state, stale = [], -1.0, 0, None, 0
    for epoch in range(1, epochs + 1):
        model.train()
        train_losses = []
        for ids, labels in loader:
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(ids), labels)
            loss.backward()
            optimizer.step()
            train_losses.append(float(loss.item()) * len(labels))
        model.eval()
        with torch.no_grad():
            train_logits = model(train_x)
            valid_logits = model(valid_x)
            valid_loss = float(criterion(valid_logits, valid_y).item())
        train_predictions = train_logits.argmax(1).tolist()
        valid_predictions = valid_logits.argmax(1).tolist()
        validation_f1 = f1_score(valid_y.tolist(), valid_predictions, average="macro", zero_division=0)
        history.append({
            "epoch": epoch,
            "train_loss": sum(train_losses) / len(train_frame),
            "validation_loss": valid_loss,
            "train_macro_f1": f1_score(train_y.tolist(), train_predictions, average="macro", zero_division=0),
            "validation_macro_f1": validation_f1,
        })
        if validation_f1 > best_f1 + 1e-6:
            best_f1, best_epoch, stale = validation_f1, epoch, 0
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        else:
            stale += 1
        if stale >= 6:
            break
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        transformer_prediction = [LABELS[index] for index in model(test_x).argmax(1).tolist()]
    baseline = Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1, 2))),
        ("classifier", LogisticRegression(max_iter=1000, random_state=SEED)),
    ])
    baseline.fit(train_frame["text"], train_frame["label"])
    baseline_validation = score(valid_frame["label"].tolist(),
                                baseline.predict(valid_frame["text"]).tolist())
    baseline_prediction = baseline.predict(test_frame["text"]).tolist()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "vocab": vocab,
                "classes": LABELS, "max_len": MAX_LEN}, output_dir / "model.pt")
    joblib.dump(baseline, output_dir / "tfidf_baseline.joblib")
    pd.DataFrame(history).to_csv(output_dir / "history.csv", index=False)
    figure, axes = plt.subplots(1, 2, figsize=(10, 4))
    for metric, axis in (("loss", axes[0]), ("macro_f1", axes[1])):
        for split in ("train", "validation"):
            axis.plot([row["epoch"] for row in history],
                      [row[f"{split}_{metric}"] for row in history], label=split)
        axis.set_title(metric)
        axis.set_xlabel("Epoch")
        axis.legend()
    figure.tight_layout()
    figure.savefig(output_dir / "training_curves.png", dpi=150)
    plt.close(figure)
    pd.DataFrame({"text": test_frame["text"].tolist(),
                  "label": test_frame["label"].tolist(),
                  "transformer_prediction": transformer_prediction,
                  "tfidf_prediction": baseline_prediction}).to_csv(
                      output_dir / "test_predictions.csv", index=False)
    result = {
        "source": str(data_path), "vocabulary_size": len(vocab),
        "split_rows": {"train": len(train_frame), "validation": len(valid_frame), "test": len(test_frame)},
        "best_epoch": best_epoch, "best_validation_macro_f1": round(float(best_f1), 4),
        "baseline_validation_macro_f1": baseline_validation["macro_f1"],
        "recommended_text_model": ("tfidf_baseline" if baseline_validation["macro_f1"] >= best_f1
                                   else "transformer"),
        "test": {
            "transformer": score(test_frame["label"].tolist(), transformer_prediction),
            "tfidf_baseline": score(test_frame["label"].tolist(), baseline_prediction),
        },
        "limitation": "Only 120 hand-written tickets; these are exploratory learning results.",
    }
    (output_dir / "metrics.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def predict(model_path: Path, text: str) -> dict:
    if not text.strip():
        raise ValueError("Ticket text must not be blank")
    checkpoint = torch.load(model_path, map_location="cpu", weights_only=True)
    model = TicketTransformer(len(checkpoint["vocab"]), len(checkpoint["classes"]),
                              checkpoint["max_len"])
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    ids = encode([text], checkpoint["vocab"], checkpoint["max_len"])
    with torch.no_grad():
        probabilities = torch.softmax(model(ids), dim=1)[0].tolist()
    classes = list(checkpoint["classes"])
    return {"issue_label": classes[int(np.argmax(probabilities))],
            "probabilities": {label: round(float(probability), 4)
                              for label, probability in zip(classes, probabilities)}}


def predict_baseline(model_path: Path, text: str) -> dict:
    if not text.strip():
        raise ValueError("Ticket text must not be blank")
    model = joblib.load(model_path)
    probabilities = model.predict_proba([text])[0]
    return {"issue_label": str(model.predict([text])[0]),
            "probabilities": {str(label): round(float(probability), 4)
                              for label, probability in zip(model.classes_, probabilities)}}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    training = commands.add_parser("train")
    training.add_argument("--data", type=Path, default=DEFAULT_DATA)
    training.add_argument("--output", type=Path, default=ARTIFACTS)
    training.add_argument("--epochs", type=int, default=25)
    inference = commands.add_parser("predict")
    inference.add_argument("text")
    inference.add_argument("--model", type=Path, default=ARTIFACTS / "model.pt")
    args = parser.parse_args()
    if args.command == "train":
        print(json.dumps(train(args.data, args.output, args.epochs), indent=2))
    else:
        print(json.dumps(predict(args.model, args.text), indent=2))


if __name__ == "__main__":
    main()
