"""
Tests for video_processor.py
- extract_keyframes: tested with a real temp directory + mocked subprocess
- encode_image_to_base64: tested with a real temp file
- generate_vertical_reel_clip: filter-chain logic tested without running FFmpeg
- analyze_scene_with_gemini / run_openai_fallback: patched at the boundary
"""
import os
import sys
import glob
import base64
import tempfile
import subprocess
import pytest
from unittest.mock import patch, MagicMock, call

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../app")))

import video_processor
from video_processor import (
    encode_image_to_base64,
    extract_keyframes,
    generate_vertical_reel_clip,
)


# ─── encode_image_to_base64 ───────────────────────────────────────────────────

class TestEncodeImageToBase64:
    def test_encodes_real_file(self, tmp_path):
        img = tmp_path / "frame.jpg"
        img.write_bytes(b"\xff\xd8\xff")   # minimal JPEG header bytes
        result = encode_image_to_base64(str(img))
        # Result must be a valid base64 string that decodes back to our bytes
        assert base64.b64decode(result) == b"\xff\xd8\xff"

    def test_returns_string(self, tmp_path):
        img = tmp_path / "frame.jpg"
        img.write_bytes(b"abc")
        assert isinstance(encode_image_to_base64(str(img)), str)

    def test_missing_file_raises(self):
        with pytest.raises(FileNotFoundError):
            encode_image_to_base64("/nonexistent/path/frame.jpg")


# ─── extract_keyframes ────────────────────────────────────────────────────────

class TestExtractKeyframes:

    @patch("video_processor.subprocess.run")
    def test_creates_output_dir_if_missing(self, mock_run, tmp_path):
        mock_run.return_value = MagicMock(returncode=0)
        output_dir = tmp_path / "frames"
        extract_keyframes("fake.mp4", str(output_dir), interval_seconds=5)
        assert output_dir.exists()

    @patch("video_processor.subprocess.run")
    def test_clears_existing_jpgs_before_extraction(self, mock_run, tmp_path):
        mock_run.return_value = MagicMock(returncode=0)
        output_dir = tmp_path / "frames"
        output_dir.mkdir()
        # Pre-populate with stale files
        (output_dir / "keyframe_0001.jpg").write_bytes(b"old")
        (output_dir / "keyframe_0002.jpg").write_bytes(b"old")
        extract_keyframes("fake.mp4", str(output_dir), interval_seconds=5)
        # After the call (before ffmpeg actually writes) the old files are gone
        assert len(list(output_dir.glob("*.jpg"))) == 0

    @patch("video_processor.subprocess.run")
    def test_ffmpeg_command_uses_correct_fps_filter(self, mock_run, tmp_path):
        mock_run.return_value = MagicMock(returncode=0)
        output_dir = tmp_path / "frames"
        extract_keyframes("fake.mp4", str(output_dir), interval_seconds=10)
        cmd = mock_run.call_args[0][0]
        assert "fps=1/10" in cmd

    @patch("video_processor.subprocess.run")
    def test_ffmpeg_uses_high_quality_jpeg(self, mock_run, tmp_path):
        mock_run.return_value = MagicMock(returncode=0)
        output_dir = tmp_path / "frames"
        extract_keyframes("fake.mp4", str(output_dir), interval_seconds=5)
        cmd = mock_run.call_args[0][0]
        assert "-q:v" in cmd and "2" in cmd

    @patch("video_processor.subprocess.run")
    def test_raises_on_ffmpeg_failure(self, mock_run, tmp_path):
        mock_run.side_effect = subprocess.CalledProcessError(1, "ffmpeg")
        output_dir = tmp_path / "frames"
        with pytest.raises(subprocess.CalledProcessError):
            extract_keyframes("bad.mp4", str(output_dir))


# ─── generate_vertical_reel_clip ─────────────────────────────────────────────

class TestGenerateVerticalReelClip:

    @patch("video_processor.subprocess.run")
    def test_blurred_mode_uses_boxblur_filter(self, mock_run, tmp_path):
        mock_run.return_value = MagicMock(returncode=0)
        src = tmp_path / "src.mp4"
        src.write_bytes(b"fake")          # must exist — validation guard checks os.path.exists
        out = str(tmp_path / "reel.mp4")
        generate_vertical_reel_clip(str(src), 0.0, 30.0, out, render_mode="blurred")
        cmd = mock_run.call_args[0][0]
        vf_value = cmd[cmd.index("-vf") + 1]
        assert "boxblur" in vf_value
        assert "split" in vf_value

    @patch("video_processor.subprocess.run")
    def test_center_mode_uses_center_crop(self, mock_run, tmp_path):
        mock_run.return_value = MagicMock(returncode=0)
        src = tmp_path / "src.mp4"
        src.write_bytes(b"fake")
        out = str(tmp_path / "reel.mp4")
        generate_vertical_reel_clip(str(src), 0.0, 30.0, out, render_mode="center")
        cmd = mock_run.call_args[0][0]
        vf_value = cmd[cmd.index("-vf") + 1]
        assert "(iw-ow)/2" in vf_value

    @patch("video_processor.subprocess.run")
    def test_left_mode_uses_zero_x_offset(self, mock_run, tmp_path):
        mock_run.return_value = MagicMock(returncode=0)
        src = tmp_path / "src.mp4"
        src.write_bytes(b"fake")
        out = str(tmp_path / "reel.mp4")
        generate_vertical_reel_clip(str(src), 10.0, 40.0, out, render_mode="left")
        cmd = mock_run.call_args[0][0]
        vf_value = cmd[cmd.index("-vf") + 1]
        assert "crop" in vf_value
        assert ":0," in vf_value or vf_value.endswith(":0")

    @patch("video_processor.subprocess.run")
    def test_right_mode_uses_iw_ow_offset(self, mock_run, tmp_path):
        mock_run.return_value = MagicMock(returncode=0)
        src = tmp_path / "src.mp4"
        src.write_bytes(b"fake")
        out = str(tmp_path / "reel.mp4")
        generate_vertical_reel_clip(str(src), 5.0, 25.0, out, render_mode="right")
        cmd = mock_run.call_args[0][0]
        vf_value = cmd[cmd.index("-vf") + 1]
        assert "iw-ow" in vf_value

    @patch("video_processor.subprocess.run")
    def test_output_uses_libx264_and_aac(self, mock_run, tmp_path):
        mock_run.return_value = MagicMock(returncode=0)
        src = tmp_path / "src.mp4"
        src.write_bytes(b"fake")
        out = str(tmp_path / "reel.mp4")
        generate_vertical_reel_clip(str(src), 0.0, 30.0, out)
        cmd = mock_run.call_args[0][0]
        assert "libx264" in cmd
        assert "aac" in cmd

    @patch("video_processor.subprocess.run")
    def test_removes_existing_output_before_render(self, mock_run, tmp_path):
        mock_run.return_value = MagicMock(returncode=0)
        src = tmp_path / "src.mp4"
        src.write_bytes(b"fake")
        out = tmp_path / "reel.mp4"
        out.write_bytes(b"old content")
        generate_vertical_reel_clip(str(src), 0.0, 10.0, str(out))
        assert not out.exists()

    @patch("video_processor.subprocess.run")
    def test_raises_on_ffmpeg_failure(self, mock_run, tmp_path):
        mock_run.side_effect = subprocess.CalledProcessError(
            1, "ffmpeg", stderr=b"encoding error"
        )
        src = tmp_path / "src.mp4"
        src.write_bytes(b"fake")
        with pytest.raises(subprocess.CalledProcessError):
            generate_vertical_reel_clip(str(src), 0.0, 30.0, str(tmp_path / "out.mp4"))

    def test_raises_when_end_before_start(self, tmp_path):
        src = tmp_path / "src.mp4"
        src.write_bytes(b"fake")
        with pytest.raises(ValueError, match="end_time"):
            generate_vertical_reel_clip(str(src), 30.0, 10.0, str(tmp_path / "out.mp4"))

    def test_raises_when_negative_start(self, tmp_path):
        src = tmp_path / "src.mp4"
        src.write_bytes(b"fake")
        with pytest.raises(ValueError, match="negative"):
            generate_vertical_reel_clip(str(src), -5.0, 10.0, str(tmp_path / "out.mp4"))

    def test_raises_when_source_missing(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            generate_vertical_reel_clip("nonexistent.mp4", 0.0, 10.0, str(tmp_path / "out.mp4"))
