from pathlib import Path

import pytest

import multimodal as app


def test_text_and_empty_text(tmp_path):
    text = tmp_path / "guide.txt"
    text.write_text("Refunds take five days. Ask billing for help.")
    result = app.process_text(text)
    assert result["input_type"] == "text"
    assert result["status"] == "ok"
    assert "Refunds" in result["summary"]
    assert result["details"]["characters"] > 0
    text.write_text(" \n")
    assert app.process_text(text)["status"] == "insufficient_input"


def test_sample_text_pdf():
    output = app.process_text(app.MEDIA / "guide.pdf")
    assert output["status"] == "ok"
    assert "five business days" in output["summary"]
    assert output["evidence"][0]["page"] == 1


def test_image_ocr_and_unreadable_image(tmp_path, monkeypatch):
    from PIL import Image
    path = tmp_path / "receipt.png"
    Image.new("RGB", (100, 50), "white").save(path)
    monkeypatch.setattr(app, "ocr_image", lambda _: [{"text": "Receipt 42", "confidence": 0.9}])
    result = app.process_image(path)
    assert result["details"]["width"] == 100
    assert result["evidence"][0]["text"] == "Receipt 42"
    monkeypatch.setattr(app, "classify_attachment", lambda *_: {"label": "billing", "probabilities": {"billing": 0.9}})
    classified = app.process_image(path, image_model=tmp_path / "demo.pt")
    assert classified["details"]["attachment_category"]["label"] == "billing"
    monkeypatch.setattr(app, "ocr_image", lambda _: [])
    assert app.process_image(path)["status"] == "insufficient_input"
    path.write_bytes(b"not an image")
    with pytest.raises(Exception):
        app.process_image(path)


def test_audio_transcript_and_silence(tmp_path, monkeypatch):
    path = tmp_path / "voice.wav"
    path.write_bytes(b"stub")
    monkeypatch.setattr(app, "transcribe_audio", lambda *_: [
        {"timestamp": 0.0, "end": 1.5, "text": "Billing receipt"}])
    result = app.process_audio(path)
    assert result["summary"] == "Billing receipt"
    assert result["details"]["segments"] == 1
    monkeypatch.setattr(app, "transcribe_audio", lambda *_: [])
    assert app.process_audio(path)["status"] == "insufficient_input"


def test_srt_timestamps_and_missing_file(tmp_path):
    srt = tmp_path / "clip.srt"
    srt.write_text("1\n00:01:02,500 --> 00:01:04,000\nReceipt downloaded.\n")
    assert app.parse_srt(srt) == [{"timestamp": 62.5, "text": "Receipt downloaded."}]
    with pytest.raises(ValueError, match="does not exist"):
        app.process_text(tmp_path / "missing.txt")


@pytest.mark.skipif(not app.shutil.which("ffmpeg") or not app.shutil.which("ffprobe"),
                    reason="FFmpeg is required for video")
def test_video_frames_and_transcript(monkeypatch):
    video = app.MEDIA / "receipt_walkthrough.mp4"
    if not video.is_file():
        pytest.skip("Sample video has not been generated")
    monkeypatch.setattr(app, "ocr_image", lambda _: [{"text": "BILLING PAGE", "confidence": 0.9}])
    output = app.process_video(video, transcript=app.MEDIA / "video_transcript.srt")
    assert output["input_type"] == "video"
    assert output["status"] == "ok"
    assert output["details"]["sampled_frames"] >= 2
    assert output["details"]["transcript_segments"] == 2
    assert any(item["kind"] == "transcript" and item["timestamp"] == 2.5 for item in output["evidence"])


def test_invalid_inputs(tmp_path):
    file = tmp_path / "wrong.csv"
    file.write_text("hello")
    with pytest.raises(ValueError):
        app.process_text(file)
    with pytest.raises(ValueError):
        app.process_image(file)
    with pytest.raises(ValueError):
        app.process_audio(file)
    with pytest.raises(ValueError):
        app.process_video(file)
    with pytest.raises(ValueError):
        app.synthesize("", tmp_path / "voice.wav")
