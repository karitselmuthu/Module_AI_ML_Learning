"""Regenerate the fictional image and video samples; audio uses a local speech engine."""

from __future__ import annotations

import subprocess
import tempfile
import shutil
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / "sample_data"


def font(size: int):
    for candidate in ("/System/Library/Fonts/Supplemental/Arial.ttf", "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"):
        if Path(candidate).exists():
            return ImageFont.truetype(candidate, size)
    return ImageFont.load_default()


def slide(title: str, subtitle: str, output: Path) -> None:
    image = Image.new("RGB", (960, 360), "white")
    draw = ImageDraw.Draw(image)
    draw.rectangle((25, 25, 935, 335), outline="#264766", width=5)
    draw.text((65, 75), title, fill="black", font=font(50))
    draw.text((65, 175), subtitle, fill="black", font=font(38))
    image.save(output)


def main() -> None:
    OUTPUT.mkdir(exist_ok=True)
    slide("BILLING RECEIPT", "Amount: $42.00", OUTPUT / "receipt.png")
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg:
        with tempfile.TemporaryDirectory() as tmp:
            first = Path(tmp) / "first.png"
            second = Path(tmp) / "second.png"
            slide("BILLING PAGE", "Open billing profile", first)
            slide("DOWNLOAD RECEIPT", "Select completed payment", second)
            # Concat demuxer uses one file per still image and explicit durations.
            concat = Path(tmp) / "slides.txt"
            concat.write_text(f"file '{first}'\nduration 2.5\nfile '{second}'\nduration 2.5\nfile '{second}'\n")
            raw_video = Path(tmp) / "silent.mp4"
            subprocess.run([ffmpeg, "-hide_banner", "-loglevel", "error", "-f", "concat", "-safe", "0",
                            "-i", str(concat), "-vf", "fps=10,format=yuv420p", "-t", "5",
                            "-y", str(raw_video)], check=True)
            sound = OUTPUT / "billing_request.wav"
            if sound.is_file():
                subprocess.run([ffmpeg, "-hide_banner", "-loglevel", "error", "-i", str(raw_video),
                                "-i", str(sound), "-map", "0:v:0", "-map", "1:a:0", "-c:v", "copy", "-c:a", "aac",
                                "-af", "apad", "-t", "5", "-y", str(OUTPUT / "receipt_walkthrough.mp4")], check=True)
            else:
                shutil.copyfile(raw_video, OUTPUT / "receipt_walkthrough.mp4")


if __name__ == "__main__":
    main()
