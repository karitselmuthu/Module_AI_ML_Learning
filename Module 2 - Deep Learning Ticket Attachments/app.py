"""Classify a ticket's text or an attached image with Module 2 models."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from generate_data import ROOT


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="mode", required=True)
    image_command = commands.add_parser("image", help="classify an attached screenshot")
    image_command.add_argument("path", type=Path)
    image_command.add_argument("--model", type=Path,
                               default=ROOT / "artifacts" / "mobilenet" / "model.pt")
    text_command = commands.add_parser("text", help="classify ticket text with the attention model")
    text_command.add_argument("message")
    text_command.add_argument("--model", type=Path,
                              default=ROOT / "artifacts" / "text_attention" / "tfidf_baseline.joblib")
    args = parser.parse_args()
    if args.mode == "image":
        from image_model import predict as predict_image

        result = {"input_type": "image", **predict_image(args.model, args.path)}
    else:
        from text_attention import predict as predict_attention, predict_baseline

        predictor = predict_attention if args.model.suffix == ".pt" else predict_baseline
        result = {"input_type": "text", **predictor(args.model, args.message)}
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
