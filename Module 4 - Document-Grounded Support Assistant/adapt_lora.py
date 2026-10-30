"""Week 10: small LoRA label-format adaptation on a pretrained FLAN-T5 model."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path

import torch
from peft import LoraConfig, TaskType, get_peft_model
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

ROOT = Path(__file__).resolve().parent
DEFAULT_DATA = ROOT / "eval" / "route_examples.jsonl"
DEFAULT_OUTPUT = ROOT / "artifacts" / "lora_route"
MODEL_ID = "google/flan-t5-small"
LABELS = {"account", "billing", "technical"}


def load_examples(path: Path) -> tuple[list[dict], list[dict]]:
    examples = [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]
    if any(row.get("label") not in LABELS or row.get("split") not in {"train", "validation"} for row in examples):
        raise ValueError("Each example needs a train/validation split and one supported label")
    train = [row for row in examples if row["split"] == "train"]
    validation = [row for row in examples if row["split"] == "validation"]
    if not train or not validation:
        raise ValueError("Need both training and validation examples")
    return train, validation


def instruction(text: str) -> str:
    return ("Route this support ticket. Reply with exactly one label: account, billing, or technical.\n"
            f"Ticket: {text}\nLabel:")


def predict_labels(model, tokenizer, examples: list[dict]) -> list[str]:
    model.eval()
    predictions = []
    for row in examples:
        tokens = tokenizer(instruction(row["text"]), return_tensors="pt", truncation=True, max_length=96)
        with torch.no_grad():
            generated = model.generate(**tokens, max_new_tokens=6, do_sample=False)
        predictions.append(tokenizer.decode(generated[0], skip_special_tokens=True).strip().lower())
    return predictions


def run(data: Path = DEFAULT_DATA, output: Path = DEFAULT_OUTPUT, steps: int = 8, seed: int = 42) -> dict:
    if steps < 1:
        raise ValueError("steps must be positive")
    train, validation = load_examples(data)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(min(4, torch.get_num_threads()))
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    base = AutoModelForSeq2SeqLM.from_pretrained(MODEL_ID, use_safetensors=True)
    baseline = predict_labels(base, tokenizer, validation)
    config = LoraConfig(r=4, lora_alpha=8, lora_dropout=0.05, target_modules=["q", "v"],
                        task_type=TaskType.SEQ_2_SEQ_LM)
    model = get_peft_model(base, config)
    trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    total = sum(parameter.numel() for parameter in model.parameters())
    optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=5e-4)
    losses = []
    for step in range(steps):
        row = train[step % len(train)]
        inputs = tokenizer(instruction(row["text"]), return_tensors="pt", truncation=True, max_length=96)
        target = tokenizer(text_target=row["label"], return_tensors="pt", max_length=8)
        model.train()
        loss = model(**inputs, labels=target["input_ids"]).loss
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        losses.append(float(loss.item()))
        print(f"step {step + 1}/{steps}: loss={losses[-1]:.4f}")
    adapted = predict_labels(model, tokenizer, validation)
    output.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(output / "adapter")
    report = {"base_model": MODEL_ID, "steps": steps, "seed": seed,
              "train_examples": len(train), "validation_examples": len(validation),
              "trainable_parameters": trainable, "total_parameters": total,
              "trainable_fraction": round(trainable / total, 6),
              "training_losses": losses,
              "baseline_accuracy": sum(p == r["label"] for p, r in zip(baseline, validation)) / len(validation),
              "adapted_accuracy": sum(p == r["label"] for p, r in zip(adapted, validation)) / len(validation),
              "validation": [{"text": row["text"], "label": row["label"], "baseline": before, "adapted": after}
                             for row, before, after in zip(validation, baseline, adapted)],
              "data_sha256": hashlib.sha256(Path(data).read_bytes()).hexdigest()}
    (output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--steps", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    print(json.dumps(run(args.data, args.output, args.steps, args.seed), indent=2))
