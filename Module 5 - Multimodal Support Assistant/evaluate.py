"""Run reproducible sample-media checks by input type and record elapsed time."""

from __future__ import annotations

import argparse
import json
import tempfile
import wave
from pathlib import Path

from PIL import Image

from multimodal import MEDIA, process_audio, process_image, process_text, process_video


def contains_all(text: str, phrases: list[str]) -> bool:
    return all(phrase.lower() in text.lower() for phrase in phrases)


def run(skip_audio: bool = False) -> dict:
    cases = [
        ("text", lambda: process_text(MEDIA / "guide.txt"), ["password", "refunds"]),
        ("image", lambda: process_image(MEDIA / "receipt.png"), ["BILLING RECEIPT", "$42.00"]),
        ("video", lambda: process_video(MEDIA / "receipt_walkthrough.mp4",
                                        transcript=MEDIA / "video_transcript.srt"), ["BILLING PAGE", "DOWNLOAD RECEIPT", "downloads a receipt"]),
    ]
    if not skip_audio:
        cases.append(("audio", lambda: process_audio(MEDIA / "billing_request.wav"), ["billing receipt"]))
    rows = []
    for kind, action, phrases in cases:
        response = action()
        rows.append({"input_type": kind, "status": response["status"],
                     "expected_phrases": phrases, "phrase_check": contains_all(response["summary"], phrases),
                     "processing_seconds": response["processing_seconds"]})
    with tempfile.TemporaryDirectory() as temp:
        empty = Path(temp) / "empty.txt"
        empty.write_text("  \n")
        blank = Path(temp) / "blank.png"
        Image.new("RGB", (200, 100), "white").save(blank)
        quiet = Path(temp) / "quiet.wav"
        with wave.open(str(quiet), "wb") as output:
            output.setnchannels(1)
            output.setsampwidth(2)
            output.setframerate(16000)
            output.writeframes(b"\x00\x00" * 16000)
        failures = {"empty_text": process_text(empty)["status"],
                    "blank_image": process_image(blank)["status"]}
        if not skip_audio:
            failures["silent_audio"] = process_audio(quiet)["status"]
    return {"cases": rows, "task_accuracy": sum(r["phrase_check"] and r["status"] == "ok" for r in rows) / len(rows),
            "failure_behavior": failures, "all_failure_checks_pass": all(value == "insufficient_input" for value in failures.values())}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--skip-audio", action="store_true", help="Avoid the first-use speech model download")
    parser.add_argument("--output", type=Path, default=Path("artifacts/evaluation.json"))
    args = parser.parse_args()
    report = run(args.skip_audio)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
