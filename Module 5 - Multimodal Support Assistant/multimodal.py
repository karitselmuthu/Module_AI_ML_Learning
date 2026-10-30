"""Small local multimodal processor with a consistent JSON result schema."""

from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parent
MEDIA = ROOT / "sample_data"
MODULE2 = ROOT.parent / "Module 2 - Deep Learning Ticket Attachments" / "image_model.py"
MAX_TEXT = 100_000
MAX_FRAMES = 8
MAX_VIDEO_SECONDS = 120


def result(kind: str, source: str, start: float, *, status: str, summary: str,
           evidence: list[dict] | None = None, details: dict | None = None) -> dict:
    return {"input_type": kind, "source": source, "status": status, "summary": summary,
            "evidence": evidence or [], "details": details or {},
            "processing_seconds": round(time.perf_counter() - start, 3)}


def check_file(path: Path) -> Path:
    path = Path(path)
    if not path.is_file():
        raise ValueError(f"Input file does not exist: {path}")
    return path


def summarize(text: str, max_sentences: int = 2) -> str:
    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+|\n+", text.strip()) if s.strip()]
    return " ".join(sentences[:max_sentences]) if sentences else ""


def process_text(path: Path) -> dict:
    start = time.perf_counter()
    path = check_file(path)
    if path.suffix.lower() == ".pdf":
        from pypdf import PdfReader
        reader = PdfReader(path)
        if reader.is_encrypted:
            raise ValueError("Encrypted PDF is unsupported")
        pages = [{"page": i, "text": page.extract_text() or ""}
                 for i, page in enumerate(reader.pages, 1)]
    elif path.suffix.lower() in {".txt", ".md"}:
        pages = [{"page": 1, "text": path.read_text(encoding="utf-8")[:MAX_TEXT]}]
    else:
        raise ValueError("Text input must be .txt, .md, or text-based .pdf")
    text = "\n".join(page["text"] for page in pages).strip()[:MAX_TEXT]
    return result("text", str(path), start, status="ok" if text else "insufficient_input",
                  summary=summarize(text, 3) if text else "No readable text found.",
                  evidence=[{"page": page["page"], "text": page["text"][:500]} for page in pages if page["text"].strip()],
                  details={"characters": len(text), "pages": len(pages)})


_ocr_engine = None


def ocr_image(path: Path) -> list[dict]:
    global _ocr_engine
    if _ocr_engine is None:
        from rapidocr import RapidOCR
        _ocr_engine = RapidOCR()
    output = _ocr_engine(str(path))
    texts = list(output.txts) if output.txts is not None else []
    scores = list(output.scores) if output.scores is not None else []
    return [{"text": str(text), "confidence": round(float(scores[i]), 3) if i < len(scores) else None}
            for i, text in enumerate(texts) if str(text).strip()]


def classify_attachment(path: Path, model_path: Path, python: Path | None = None) -> dict:
    if not Path(model_path).is_file():
        raise ValueError(f"Attachment model does not exist: {model_path}")
    if not MODULE2.is_file():
        raise ValueError("Module 2 image_model.py is missing")
    completed = subprocess.run([str(python or sys.executable), str(MODULE2), "predict",
                                str(path), "--model", str(model_path)], capture_output=True, text=True)
    if completed.returncode:
        raise RuntimeError(f"Module 2 image prediction failed: {completed.stderr.strip()[-500:]}")
    return json.loads(completed.stdout)


def process_image(path: Path, *, image_model: Path | None = None, image_python: Path | None = None) -> dict:
    start = time.perf_counter()
    path = check_file(path)
    if path.suffix.lower() not in {".png", ".jpg", ".jpeg", ".webp"}:
        raise ValueError("Image input must be PNG, JPEG, or WebP")
    with Image.open(path) as image:
        width, height = image.size
        image.verify()
    lines = ocr_image(path)
    text = " ".join(line["text"] for line in lines)
    category = classify_attachment(path, image_model, image_python) if image_model else None
    summary = summarize(text) if text else "No readable text detected in image."
    if category:
        summary += f" Attachment category: {category['label']}."
    return result("image", str(path), start, status="ok" if text or category else "insufficient_input",
                  summary=summary,
                  evidence=lines, details={"width": width, "height": height, "ocr_lines": len(lines),
                                          "attachment_category": category})


def require_binary(name: str) -> str:
    binary = shutil.which(name)
    if not binary:
        raise RuntimeError(f"{name} is required for this workflow; see README.md")
    return binary


def media_duration(path: Path) -> float:
    run = subprocess.run([require_binary("ffprobe"), "-v", "error", "-show_entries", "format=duration",
                          "-of", "default=noprint_wrappers=1:nokey=1", str(path)],
                         capture_output=True, text=True, check=True)
    duration = float(run.stdout.strip())
    return duration


def ffprobe_duration(path: Path) -> float:
    duration = media_duration(path)
    if not 0 < duration <= MAX_VIDEO_SECONDS:
        raise ValueError(f"Video duration must be 0–{MAX_VIDEO_SECONDS} seconds")
    return duration


def parse_srt(path: Path) -> list[dict]:
    content = check_file(path).read_text(encoding="utf-8-sig")
    cues = []
    for block in re.split(r"\n\s*\n", content.strip()):
        match = re.search(r"(\d{2}):(\d{2}):(\d{2})[,\.]([0-9]{3})\s+-->\s+", block)
        if not match:
            continue
        hour, minute, second, ms = map(int, match.groups())
        lines = block.splitlines()
        cue_line = next(i for i, line in enumerate(lines) if "-->" in line)
        text = " ".join(line.strip() for line in lines[cue_line + 1:])
        if text:
            cues.append({"timestamp": round(hour * 3600 + minute * 60 + second + ms / 1000, 3), "text": text})
    return cues


def transcribe_audio(path: Path, model_name: str = "tiny.en") -> list[dict]:
    completed = subprocess.run([sys.executable, str(ROOT / "audio_backend.py"), str(path), "--model", model_name],
                               capture_output=True, text=True)
    if completed.returncode:
        raise RuntimeError(f"Speech recognition failed: {completed.stderr.strip()[-500:]}")
    return json.loads(completed.stdout)


def process_audio(path: Path, model_name: str = "tiny.en") -> dict:
    start = time.perf_counter()
    path = check_file(path)
    if path.suffix.lower() not in {".wav", ".mp3", ".m4a", ".flac", ".aiff"}:
        raise ValueError("Audio input must be WAV, MP3, M4A, FLAC, or AIFF")
    segments = transcribe_audio(path, model_name)
    text = " ".join(segment["text"] for segment in segments)
    return result("audio", str(path), start, status="ok" if text else "insufficient_input",
                  summary=summarize(text) if text else "No speech detected.",
                  evidence=segments, details={"transcript": text, "segments": len(segments), "model": model_name})


def synthesize(text: str, output: Path) -> dict:
    start = time.perf_counter()
    if not text.strip() or len(text) > 500:
        raise ValueError("Speech text must contain 1–500 characters")
    output = Path(output)
    if output.suffix.lower() not in {".wav", ".aiff"}:
        raise ValueError("Speech output must be .wav or .aiff")
    output.parent.mkdir(parents=True, exist_ok=True)
    if shutil.which("say") and shutil.which("ffmpeg"):
        with tempfile.TemporaryDirectory() as temporary:
            intermediate = Path(temporary) / "speech.aiff"
            subprocess.run(["say", "-o", str(intermediate), text], check=True)
            if output.suffix.lower() == ".aiff":
                shutil.copyfile(intermediate, output)
            else:
                subprocess.run([require_binary("ffmpeg"), "-hide_banner", "-loglevel", "error", "-i", str(intermediate),
                                "-ac", "1", "-ar", "16000", "-y", str(output)], check=True)
    else:
        import pyttsx3
        engine = pyttsx3.init()
        engine.save_to_file(text, str(output))
        engine.runAndWait()
    if not output.is_file() or output.stat().st_size < 1000:
        raise RuntimeError("Speech engine did not create a usable audio file")
    if shutil.which("ffprobe"):
        run = subprocess.run([require_binary("ffprobe"), "-v", "error", "-show_entries", "format=duration",
                              "-of", "default=noprint_wrappers=1:nokey=1", str(output)],
                             capture_output=True, text=True)
        try:
            duration = float(run.stdout.strip())
        except ValueError:
            duration = 0
        if run.returncode or duration <= 0:
            output.unlink(missing_ok=True)
            raise RuntimeError("Speech engine created an audio file with no decodable duration")
    else:
        duration = None
    return result("speech_output", str(output), start, status="ok", summary=text,
                  details={"bytes": output.stat().st_size, "duration_seconds": duration})


def process_video(path: Path, *, transcript: Path | None = None, interval: float = 2,
                  transcribe: bool = False, model_name: str = "tiny.en") -> dict:
    start = time.perf_counter()
    path = check_file(path)
    if path.suffix.lower() not in {".mp4", ".mov", ".mkv", ".webm"}:
        raise ValueError("Video input must be MP4, MOV, MKV, or WebM")
    if interval <= 0:
        raise ValueError("Frame interval must be positive")
    duration = ffprobe_duration(path)
    timestamps = [round(i * interval, 3) for i in range(min(MAX_FRAMES, int(duration / interval) + 1))
                  if i * interval < duration]
    frames = []
    with tempfile.TemporaryDirectory() as temporary:
        for i, timestamp in enumerate(timestamps):
            frame = Path(temporary) / f"frame_{i:02d}.png"
            subprocess.run([require_binary("ffmpeg"), "-hide_banner", "-loglevel", "error", "-ss", str(timestamp),
                            "-i", str(path), "-frames:v", "1", "-y", str(frame)], check=True, capture_output=True)
            if not frame.is_file():
                continue
            lines = ocr_image(frame)
            frames.append({"timestamp": timestamp, "text": " ".join(x["text"] for x in lines),
                           "ocr_lines": lines})
        if transcript:
            cues = parse_srt(transcript) if transcript.suffix.lower() == ".srt" else [
                {"timestamp": 0.0, "text": check_file(transcript).read_text(encoding="utf-8")[:MAX_TEXT]}]
        elif transcribe:
            audio = Path(temporary) / "audio.wav"
            extract = subprocess.run([require_binary("ffmpeg"), "-hide_banner", "-loglevel", "error", "-i", str(path),
                                      "-vn", "-ac", "1", "-ar", "16000", "-y", str(audio)], capture_output=True)
            cues = transcribe_audio(audio, model_name) if extract.returncode == 0 and audio.is_file() else []
        else:
            cues = []
    evidence = ([{"kind": "frame", **frame} for frame in frames if frame["text"]] +
                [{"kind": "transcript", **cue} for cue in cues if cue["text"]])
    evidence.sort(key=lambda item: item["timestamp"])
    visual_steps = list(dict.fromkeys(frame["text"] for frame in frames if frame["text"]))
    narration = list(dict.fromkeys(cue["text"] for cue in cues if cue["text"]))
    parts = []
    if visual_steps:
        parts.append("Visual steps: " + "; ".join(visual_steps[:3]) + ".")
    if narration:
        parts.append("Narration: " + summarize(" ".join(narration), 2))
    summary = " ".join(parts)
    return result("video", str(path), start, status="ok" if evidence else "insufficient_input",
                  summary=summary if summary else "No readable frame text or transcript found.",
                  evidence=evidence, details={"duration_seconds": round(duration, 3), "sampled_frames": len(frames),
                                              "transcript_segments": len(cues), "interval_seconds": interval})


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="task", required=True)
    for task in ("text", "image", "audio", "video"):
        sub.add_parser(task).add_argument("path", type=Path)
    sub.choices["image"].add_argument("--image-model", type=Path)
    sub.choices["image"].add_argument("--image-python", type=Path)
    sub.choices["audio"].add_argument("--model", default="tiny.en")
    sub.choices["video"].add_argument("--transcript", type=Path)
    sub.choices["video"].add_argument("--transcribe", action="store_true")
    sub.choices["video"].add_argument("--interval", type=float, default=2)
    sub.choices["video"].add_argument("--model", default="tiny.en")
    speech = sub.add_parser("speak")
    speech.add_argument("text")
    speech.add_argument("--output", type=Path, default=ROOT / "artifacts" / "spoken_answer.wav")
    args = parser.parse_args()
    if args.task == "text":
        payload = process_text(args.path)
    elif args.task == "image":
        payload = process_image(args.path, image_model=args.image_model, image_python=args.image_python)
    elif args.task == "audio":
        payload = process_audio(args.path, args.model)
    elif args.task == "video":
        payload = process_video(args.path, transcript=args.transcript, interval=args.interval,
                                transcribe=args.transcribe, model_name=args.model)
    else:
        payload = synthesize(args.text, args.output)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
