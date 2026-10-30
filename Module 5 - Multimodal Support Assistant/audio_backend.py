"""Run speech recognition in a separate process from OCR's OpenCV libraries."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from pathlib import Path

from faster_whisper import WhisperModel

ROOT = Path(__file__).resolve().parent


def duration(path: Path) -> float | None:
    if not shutil.which("ffprobe"):
        return None
    run = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration",
                          "-of", "default=noprint_wrappers=1:nokey=1", str(path)],
                         capture_output=True, text=True, check=True)
    return float(run.stdout.strip())


def transcribe(path: Path, model_name: str) -> list[dict]:
    model = WhisperModel(model_name, device="cpu", compute_type="int8", download_root=str(ROOT / "artifacts" / "models"))
    segments, _ = model.transcribe(str(path), language="en", beam_size=1, vad_filter=True)
    length = duration(path)
    return [{"timestamp": round(float(segment.start), 3),
             "end": round(min(float(segment.end), length), 3) if length else round(float(segment.end), 3),
             "text": segment.text.strip()} for segment in segments if segment.text.strip()]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", type=Path)
    parser.add_argument("--model", default="tiny.en")
    args = parser.parse_args()
    print(json.dumps(transcribe(args.path, args.model)))
