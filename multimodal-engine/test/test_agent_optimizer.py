"""
Tests for agent_optimizer.py
Covers:
- Layout style → render_mode routing logic (Blurred Stack vs AI Smart Crop)
- Slug sanitisation logic used for output filenames
- run_autonomous_editing_pipeline returns False when no highlights found
- run_autonomous_editing_pipeline iterates and calls generate_vertical_reel_clip per highlight
All Gemini API calls and FFmpeg calls are fully mocked.
"""
import os
import sys
import pytest
from unittest.mock import patch, MagicMock, call

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../app")))

from models import (
    StructuredTranscript,
    AudioSegment,
    WordTimestamp,
    ChronologicalVisualTimeline,
    VideoFrameMoment,
)


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _make_transcript():
    return StructuredTranscript(
        segments=[
            AudioSegment(
                start_time=0.0,
                end_time=10.0,
                text="Hello world",
                words=[WordTimestamp(word="Hello", start=0.0, end=0.5)],
            )
        ]
    )

def _make_timeline():
    return ChronologicalVisualTimeline(
        timeline=[VideoFrameMoment(timestamp_seconds=5.0, visual_description="Slide visible")]
    )

def _make_highlight(title="Great Hook", start=10.0, end=30.0, position="center"):
    """Return a mock ProductionHighlight object."""
    h = MagicMock()
    h.hook_title = title
    h.start_time = start
    h.end_time = end
    h.speaker_position = position
    return h


# ─── Slug sanitisation (inline logic extracted for testing) ───────────────────

def _sanitise_slug(title: str) -> str:
    """Mirrors the slug logic inside run_autonomous_editing_pipeline."""
    return (
        "".join(c for c in title if c.isalnum() or c in (" ", "_"))
        .rstrip()
        .replace(" ", "_")
        .lower()
    )

class TestSlugSanitisation:
    def test_spaces_become_underscores(self):
        assert _sanitise_slug("Hello World") == "hello_world"

    def test_special_chars_removed(self):
        assert _sanitise_slug("Top 5 Tips! #viral") == "top_5_tips_viral"

    def test_already_clean_title(self):
        assert _sanitise_slug("clean_title") == "clean_title"

    def test_empty_title(self):
        assert _sanitise_slug("") == ""

    def test_only_special_chars_becomes_empty(self):
        assert _sanitise_slug("!!!") == ""


# ─── Layout style → render mode routing ──────────────────────────────────────

def _resolve_render_mode(layout_style: str, speaker_position: str) -> str:
    """Mirrors the if/else in run_autonomous_editing_pipeline."""
    if "Blurred Stack" in layout_style:
        return "blurred"
    return speaker_position

class TestRenderModeRouting:
    def test_blurred_stack_overrides_speaker_position(self):
        assert _resolve_render_mode("Blurred Stack Mode (Presentation/Code)", "left") == "blurred"

    def test_ai_smart_crop_uses_speaker_position_center(self):
        assert _resolve_render_mode("AI Smart Face Crop (Podcast/Vlog)", "center") == "center"

    def test_ai_smart_crop_uses_speaker_position_left(self):
        assert _resolve_render_mode("AI Smart Face Crop (Podcast/Vlog)", "left") == "left"

    def test_ai_smart_crop_uses_speaker_position_right(self):
        assert _resolve_render_mode("AI Smart Face Crop (Podcast/Vlog)", "right") == "right"

    def test_blurred_partial_match_still_resolves(self):
        # Ensures substring matching works as coded
        assert _resolve_render_mode("Blurred Stack", "right") == "blurred"


# ─── run_autonomous_editing_pipeline ─────────────────────────────────────────

class TestRunAutonomousEditingPipeline:

    @patch("agent_optimizer.generate_vertical_reel_clip")
    @patch("agent_optimizer.discover_highlights_autonomously", return_value=[])
    def test_returns_false_when_no_highlights(self, mock_discover, mock_render):
        from agent_optimizer import run_autonomous_editing_pipeline
        result = run_autonomous_editing_pipeline(
            video_path="fake.mp4",
            audio_transcript=_make_transcript(),
            visual_breakdown=_make_timeline(),
            layout_style="AI Smart Face Crop",
        )
        assert result is False
        mock_render.assert_not_called()

    @patch("agent_optimizer.os.makedirs")
    @patch("agent_optimizer.generate_vertical_reel_clip")
    @patch("agent_optimizer.discover_highlights_autonomously")
    def test_returns_true_when_highlights_found(self, mock_discover, mock_render, mock_makedirs):
        mock_discover.return_value = [_make_highlight("Good Clip", 5.0, 25.0, "center")]
        from agent_optimizer import run_autonomous_editing_pipeline
        result = run_autonomous_editing_pipeline(
            video_path="fake.mp4",
            audio_transcript=_make_transcript(),
            visual_breakdown=_make_timeline(),
            layout_style="AI Smart Face Crop",
        )
        assert result is True

    @patch("agent_optimizer.os.makedirs")
    @patch("agent_optimizer.generate_vertical_reel_clip")
    @patch("agent_optimizer.discover_highlights_autonomously")
    def test_calls_render_once_per_highlight(self, mock_discover, mock_render, mock_makedirs):
        highlights = [
            _make_highlight("Clip One", 5.0, 25.0, "center"),
            _make_highlight("Clip Two", 40.0, 60.0, "left"),
        ]
        mock_discover.return_value = highlights
        from agent_optimizer import run_autonomous_editing_pipeline
        run_autonomous_editing_pipeline(
            video_path="fake.mp4",
            audio_transcript=_make_transcript(),
            visual_breakdown=_make_timeline(),
            layout_style="AI Smart Face Crop",
        )
        assert mock_render.call_count == 2

    @patch("agent_optimizer.os.makedirs")
    @patch("agent_optimizer.generate_vertical_reel_clip")
    @patch("agent_optimizer.discover_highlights_autonomously")
    def test_blurred_mode_overrides_render_mode(self, mock_discover, mock_render, mock_makedirs):
        mock_discover.return_value = [_make_highlight("Demo", 0.0, 20.0, "right")]
        from agent_optimizer import run_autonomous_editing_pipeline
        run_autonomous_editing_pipeline(
            video_path="fake.mp4",
            audio_transcript=_make_transcript(),
            visual_breakdown=_make_timeline(),
            layout_style="Blurred Stack Mode",
        )
        _, kwargs = mock_render.call_args
        assert kwargs["render_mode"] == "blurred"

    @patch("agent_optimizer.os.makedirs")
    @patch("agent_optimizer.generate_vertical_reel_clip")
    @patch("agent_optimizer.discover_highlights_autonomously")
    def test_ffmpeg_error_does_not_crash_pipeline(self, mock_discover, mock_render, mock_makedirs):
        """A single clip render failure should not stop remaining clips."""
        mock_discover.return_value = [
            _make_highlight("Bad Clip", 0.0, 20.0, "center"),
            _make_highlight("Good Clip", 30.0, 50.0, "center"),
        ]
        mock_render.side_effect = [Exception("ffmpeg failed"), None]
        from agent_optimizer import run_autonomous_editing_pipeline
        # Should not raise — the exception is caught internally
        result = run_autonomous_editing_pipeline(
            video_path="fake.mp4",
            audio_transcript=_make_transcript(),
            visual_breakdown=_make_timeline(),
            layout_style="AI Smart Face Crop",
        )
        assert result is True
        assert mock_render.call_count == 2
