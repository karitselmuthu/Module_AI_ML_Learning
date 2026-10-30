"""Week 8: reproducible, reviewable text-to-image diffusion runs."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import torch
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parent
DEFAULT_CONFIG = ROOT / "configs" / "asset_run.json"
DEFAULT_RUNS = ROOT / "artifacts" / "diffusion_runs"


def validate_config(config: dict) -> None:
    if not isinstance(config.get("model_id"), str) or not config["model_id"].strip():
        raise ValueError("model_id is required")
    prompts = config.get("prompts")
    settings = config.get("settings")
    if not isinstance(prompts, list) or not prompts or not all(isinstance(p, str) and p.strip() for p in prompts):
        raise ValueError("prompts must be a nonempty list of nonempty strings")
    if not isinstance(settings, list) or not settings:
        raise ValueError("settings must be a nonempty list")
    if not isinstance(config.get("seed"), int) or config["seed"] < 0:
        raise ValueError("seed must be a nonnegative integer")
    for dimension in ("width", "height"):
        value = config.get(dimension, 256)
        if not isinstance(value, int) or value < 64 or value > 1024 or value % 8:
            raise ValueError(f"{dimension} must be a multiple of 8 from 64 to 1024")
    for setting in settings:
        if not isinstance(setting, dict):
            raise ValueError("each setting must be an object")
        if not isinstance(setting.get("steps"), int) or not 1 <= setting["steps"] <= 100:
            raise ValueError("steps must be an integer from 1 to 100")
        if not isinstance(setting.get("guidance_scale"), (int, float)) or not 0 <= setting["guidance_scale"] <= 20:
            raise ValueError("guidance_scale must be from 0 to 20")


def load_pipeline(model_id: str, revision: str | None = None):
    try:
        from diffusers import DiffusionPipeline
    except ImportError as exc:
        raise RuntimeError("Install requirements.txt to run diffusion generation") from exc
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    kwargs = {"dtype": torch.float16 if device == "cuda" else torch.float32}
    if revision:
        kwargs["revision"] = revision
    pipe = DiffusionPipeline.from_pretrained(model_id, **kwargs)
    pipe.to(device)
    if device == "cpu" and hasattr(pipe, "enable_attention_slicing"):
        pipe.enable_attention_slicing()
    return pipe, device


def contact_sheet(records: list[dict], run_dir: Path) -> None:
    thumb_width, thumb_height = 256, 256
    caption_height = 62
    sheet = Image.new("RGB", (thumb_width * 2, (thumb_height + caption_height) * ((len(records) + 1) // 2)), "white")
    draw = ImageDraw.Draw(sheet)
    for index, row in enumerate(records):
        with Image.open(run_dir / row["image"]) as source:
            image = source.convert("RGB")
            image.thumbnail((thumb_width, thumb_height))
        x = (index % 2) * thumb_width
        y = (index // 2) * (thumb_height + caption_height)
        sheet.paste(image, (x + (thumb_width-image.width)//2, y + (thumb_height-image.height)//2))
        caption = f"{row['image']}\n{row['steps']} steps · guidance {row['guidance_scale']}"
        draw.multiline_text((x + 5, y + thumb_height + 2), caption, fill="black", spacing=2)
    sheet.save(run_dir / "contact_sheet.png")


def generate(config_path: Path, runs_dir: Path = DEFAULT_RUNS, pipeline=None) -> Path:
    raw = config_path.read_bytes()
    config = json.loads(raw)
    validate_config(config)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    run_id = f"{timestamp}-{hashlib.sha256(raw).hexdigest()[:8]}"
    run_dir = runs_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    (run_dir / "config.json").write_bytes(raw)
    if pipeline is None:
        pipeline, device = load_pipeline(config["model_id"], config.get("revision"))
    else:
        device = "cpu"
    records = []
    try:
        for prompt_index, prompt in enumerate(config["prompts"]):
            # Same noise seed for a prompt across settings: comparison isolates setting changes.
            seed = config["seed"] + prompt_index
            for setting_index, setting in enumerate(config["settings"]):
                generator = torch.Generator(device=device).manual_seed(seed)
                image = pipeline(prompt=prompt, num_inference_steps=setting["steps"],
                                 guidance_scale=float(setting["guidance_scale"]),
                                 negative_prompt=config.get("negative_prompt") or None,
                                 width=config.get("width", 256), height=config.get("height", 256),
                                 generator=generator).images[0]
                filename = f"prompt-{prompt_index+1:02d}-setting-{setting_index+1:02d}.png"
                image.save(run_dir / filename)
                records.append({"image": filename, "prompt": prompt, "negative_prompt": config.get("negative_prompt", ""),
                                "seed": seed, "steps": setting["steps"],
                                "width": config.get("width", 256), "height": config.get("height", 256),
                                "guidance_scale": float(setting["guidance_scale"]),
                                "review": "pending", "notes": ""})
                (run_dir / "review.json").write_text(json.dumps(records, indent=2) + "\n")
        contact_sheet(records, run_dir)
    finally:
        if records and not (run_dir / "review.json").exists():
            (run_dir / "review.json").write_text(json.dumps(records, indent=2) + "\n")
    return run_dir


def review(run_dir: Path, image_name: str, decision: str, notes: str) -> Path:
    path = run_dir / "review.json"
    records = json.loads(path.read_text())
    matches = [row for row in records if row["image"] == image_name]
    if len(matches) != 1 or not (run_dir / image_name).is_file():
        raise ValueError("image must be a generated file listed in review.json")
    matches[0]["review"] = decision
    matches[0]["notes"] = notes.strip()
    path.write_text(json.dumps(records, indent=2) + "\n")
    return path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    generation = commands.add_parser("generate")
    generation.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    generation.add_argument("--runs-dir", type=Path, default=DEFAULT_RUNS)
    reviewing = commands.add_parser("review")
    reviewing.add_argument("run_dir", type=Path)
    reviewing.add_argument("image")
    reviewing.add_argument("--decision", choices=("accepted", "rejected", "pending"), required=True)
    reviewing.add_argument("--notes", default="")
    args = parser.parse_args()
    if args.command == "generate":
        print(generate(args.config, args.runs_dir))
    else:
        print(review(args.run_dir, args.image, args.decision, args.notes))
