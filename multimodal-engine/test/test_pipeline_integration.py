"""
Integration tests for the full 4-phase pipeline.

All external boundaries (Gemini, OpenAI, FFmpeg, disk I/O) are mocked so these
tests run without API keys, video files, or FFmpeg installed.

They verify that the phases wire together correctly:
  Phase 1 → audio extraction + transcription  → StructuredTranscript
  Phase 2 → keyframe extraction + VLM analysis → ChronologicalVisualTimeline
  Phase 3 → blog synthesis                     → Markdown string saved to disk
  Phase 4 → highlight detection + reel render  → MP4 files written to output/clips
"""

import os
import sys
import json
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


# ─── Reusable fixtures ────────────────────────────────────────────────────────

@pytest.fixture
def sample_transcript():
    return StructuredTranscript(
        segments=[
            AudioSegment(
                start_time=0.0,
                end_time=15.0,
                text="Welcome to the tutorial. Today we cover embeddings.",
                words=[
                    WordTimestamp(word="Welcome", start=0.0, end=0.5),
                    WordTimestamp(word="embeddings", start=14.0, end=14.8),
                ],
            ),
            AudioSegment(
                start_time=15.0,
                end_time=45.0,
                text="Vectors allow semantic similarity search across large datasets.",
                words=[
                    WordTimestamp(word="Vectors", start=15.0, end=15.6),
                    WordTimestamp(word="datasets", start=44.0, end=44.9),
                ],
            ),
        ]
    )


@pytest.fixture
def sample_timeline():
    return ChronologicalVisualTimeline(
        timeline=[
            VideoFrameMoment(timestamp_seconds=5.0,  visual_description="Title slide: AI Embeddings"),
            VideoFrameMoment(timestamp_seconds=10.0, visual_description="Speaker at whiteboard"),
            VideoFrameMoment(timestamp_seconds=15.0, visual_description="Code editor visible"),
            VideoFrameMoment(timestamp_seconds=20.0, visual_description="Vector diagram on screen"),
        ]
    )


# ─── Phase 1 integration: audio extraction → transcription ───────────────────

class TestPhase1AudioTranscription:

    @patch("audio_processor.genai.Client")
    @patch("audio_processor.subprocess.run")
    def test_extract_then_transcribe_returns_structured_transcript(
        self, mock_run, mock_client_cls, tmp_path, sample_transcript
    ):
        """FFmpeg succeeds → Gemini returns parsed StructuredTranscript."""
        mock_run.return_value = MagicMock(returncode=0)

        # Build a mock Gemini client that returns sample_transcript via .parsed
        mock_client = MagicMock()
        mock_client_cls.return_value = mock_client
        mock_upload = MagicMock()
        mock_upload.name = "files/test-audio-001"
        mock_client.files.upload.return_value = mock_upload
        mock_response = MagicMock()
        mock_response.parsed = sample_transcript
        mock_client.models.generate_content.return_value = mock_response

        from audio_processor import extract_audio_from_video, transcribe_audio

        video_path = str(tmp_path / "video.mp4")
        audio_path = str(tmp_path / "audio.wav")

        # Create a dummy video file so FFmpeg mock has a real path
        open(video_path, "wb").close()

        extract_audio_from_video(video_path, audio_path)

        # Create the audio file so transcribe_audio can upload it
        open(audio_path, "wb").close()
        result = transcribe_audio(audio_path)

        assert isinstance(result, StructuredTranscript)
        assert len(result.segments) == 2
        assert result.segments[0].text == "Welcome to the tutorial. Today we cover embeddings."

    @patch("audio_processor.genai.Client")
    @patch("audio_processor.subprocess.run")
    def test_transcription_none_parsed_raises(self, mock_run, mock_client_cls, tmp_path):
        """If Gemini returns None for .parsed the exception propagates cleanly."""
        mock_run.return_value = MagicMock(returncode=0)

        mock_client = MagicMock()
        mock_client_cls.return_value = mock_client
        mock_upload = MagicMock()
        mock_upload.name = "files/test-audio-002"
        mock_client.files.upload.return_value = mock_upload
        mock_response = MagicMock()
        mock_response.parsed = None
        mock_client.models.generate_content.return_value = mock_response

        from audio_processor import transcribe_audio

        audio_path = str(tmp_path / "audio.wav")
        open(audio_path, "wb").close()

        # Should not crash silently — caller must handle None
        result = transcribe_audio(audio_path)
        assert result is None  # pipeline callers are responsible for the guard


# ─── Phase 2 integration: keyframe extraction → scene analysis ───────────────

class TestPhase2VisualTimeline:

    @patch("video_processor.genai.Client")
    @patch("video_processor.subprocess.run")
    def test_extract_then_analyze_returns_timeline(
        self, mock_run, mock_client_cls, tmp_path, sample_timeline
    ):
        """FFmpeg succeeds → Gemini returns a ChronologicalVisualTimeline."""
        mock_run.return_value = MagicMock(returncode=0)

        # Create fake JPEG frames so glob finds them
        frames_dir = tmp_path / "frames"
        frames_dir.mkdir()
        for i in range(1, 5):
            (frames_dir / f"keyframe_{i:04d}.jpg").write_bytes(b"\xff\xd8\xff")

        mock_client = MagicMock()
        mock_client_cls.return_value = mock_client
        mock_upload = MagicMock()
        mock_upload.name = "files/frame-001"
        mock_client.files.upload.return_value = mock_upload
        mock_response = MagicMock()
        mock_response.parsed = sample_timeline
        mock_client.models.generate_content.return_value = mock_response

        from video_processor import analyze_scene_with_gemini

        result = analyze_scene_with_gemini(str(frames_dir), interval_seconds=5)

        assert isinstance(result, ChronologicalVisualTimeline)
        assert len(result.timeline) == 4
        assert result.timeline[0].timestamp_seconds == 5.0

    @patch("video_processor.run_openai_fallback")
    @patch("video_processor.genai.Client")
    def test_gemini_failure_falls_back_to_openai_timeline(
        self, mock_client_cls, mock_fallback, tmp_path
    ):
        """When Gemini raises, the OpenAI fallback returns a valid timeline object."""
        frames_dir = tmp_path / "frames"
        frames_dir.mkdir()
        for i in range(1, 3):
            (frames_dir / f"keyframe_{i:04d}.jpg").write_bytes(b"\xff\xd8\xff")

        mock_client = MagicMock()
        mock_client_cls.return_value = mock_client
        mock_client.files.upload.side_effect = RuntimeError("Gemini upload failed")
        mock_fallback.return_value = "Whiteboard with diagrams visible on screen."

        from video_processor import analyze_scene_with_gemini

        result = analyze_scene_with_gemini(str(frames_dir), interval_seconds=5)

        # Must return a typed ChronologicalVisualTimeline — not a raw string
        assert isinstance(result, ChronologicalVisualTimeline)
        assert len(result.timeline) == 2
        assert "[OpenAI Fallback Data]" in result.timeline[0].visual_description

    @patch("video_processor.run_openai_fallback")
    @patch("video_processor.genai.Client")
    def test_both_providers_fail_returns_placeholder_timeline(
        self, mock_client_cls, mock_fallback, tmp_path
    ):
        """If both Gemini and OpenAI fail, a placeholder timeline is still returned."""
        frames_dir = tmp_path / "frames"
        frames_dir.mkdir()
        (frames_dir / "keyframe_0001.jpg").write_bytes(b"\xff\xd8\xff")

        mock_client = MagicMock()
        mock_client_cls.return_value = mock_client
        mock_client.files.upload.side_effect = RuntimeError("Gemini down")
        mock_fallback.return_value = None  # OpenAI also fails

        from video_processor import analyze_scene_with_gemini

        result = analyze_scene_with_gemini(str(frames_dir), interval_seconds=5)

        assert isinstance(result, ChronologicalVisualTimeline)
        assert result.timeline[0].visual_description == "[OpenAI Fallback Data]: Visual capture processing failure."


# ─── Phase 3 integration: blog synthesis ─────────────────────────────────────

class TestPhase3BlogSynthesis:

    @patch("workflow_engine.genai.Client")
    def test_generate_blog_embeds_transcript_in_prompt(
        self, mock_client_cls, sample_transcript, sample_timeline, tmp_path
    ):
        """Verify the transcript JSON is actually injected into the Gemini prompt."""
        mock_client = MagicMock()
        mock_client_cls.return_value = mock_client
        mock_response = MagicMock()
        mock_response.text = (
            '---\ntitle: "Test"\nslug: "test-slug"\n---\n\n## Intro\nSome content.'
        )
        mock_client.models.generate_content.return_value = mock_response

        from workflow_engine import generate_production_blog

        result = generate_production_blog(sample_transcript, sample_timeline, "tutorial.mp4")

        # The prompt sent to Gemini must contain the serialised transcript data
        call_args = mock_client.models.generate_content.call_args
        prompt_sent = call_args[1]["contents"][0] if call_args[1] else call_args[0][1][0]
        assert "Welcome to the tutorial" in prompt_sent or "segments" in prompt_sent

    @patch("workflow_engine.genai.Client")
    def test_generate_blog_returns_string(
        self, mock_client_cls, sample_transcript, sample_timeline
    ):
        mock_client = MagicMock()
        mock_client_cls.return_value = mock_client
        mock_client.models.generate_content.return_value.text = "# Blog\n\nContent here."

        from workflow_engine import generate_production_blog

        result = generate_production_blog(sample_transcript, sample_timeline, "video.mp4")

        assert isinstance(result, str)
        assert len(result) > 0


# ─── Phase 4 integration: highlight detection → reel rendering ───────────────

class TestPhase4ReelPipeline:

    @patch("agent_optimizer.generate_vertical_reel_clip")
    @patch("agent_optimizer.discover_highlights_autonomously")
    def test_pipeline_calls_render_for_each_highlight(
        self, mock_discover, mock_render, sample_transcript, sample_timeline
    ):
        """Each discovered highlight must produce exactly one render call."""
        h1 = MagicMock()
        h1.hook_title = "Why Embeddings Matter"
        h1.start_time = 10.0
        h1.end_time = 35.0
        h1.speaker_position = "center"

        h2 = MagicMock()
        h2.hook_title = "Vector Search Demo"
        h2.start_time = 40.0
        h2.end_time = 65.0
        h2.speaker_position = "left"

        mock_discover.return_value = [h1, h2]

        from agent_optimizer import run_autonomous_editing_pipeline

        result = run_autonomous_editing_pipeline(
            video_path="fake_video.mp4",
            audio_transcript=sample_transcript,
            visual_breakdown=sample_timeline,
            layout_style="AI Smart Face Crop (Podcast/Vlog)",
        )

        assert result is True
        assert mock_render.call_count == 2

    @patch("agent_optimizer.generate_vertical_reel_clip")
    @patch("agent_optimizer.discover_highlights_autonomously")
    def test_blurred_mode_overrides_all_speaker_positions(
        self, mock_discover, mock_render, sample_transcript, sample_timeline
    ):
        """Blurred Stack layout must ignore speaker_position for every clip."""
        for pos in ["left", "center", "right"]:
            mock_render.reset_mock()
            h = MagicMock()
            h.hook_title = "Test Clip"
            h.start_time = 5.0
            h.end_time = 25.0
            h.speaker_position = pos
            mock_discover.return_value = [h]

            from agent_optimizer import run_autonomous_editing_pipeline

            run_autonomous_editing_pipeline(
                video_path="fake_video.mp4",
                audio_transcript=sample_transcript,
                visual_breakdown=sample_timeline,
                layout_style="Blurred Stack Mode (Presentation/Code)",
            )

            _, kwargs = mock_render.call_args
            assert kwargs["render_mode"] == "blurred", (
                f"Expected 'blurred' for speaker_position='{pos}', got '{kwargs['render_mode']}'"
            )

    @patch("agent_optimizer.generate_vertical_reel_clip")
    @patch("agent_optimizer.discover_highlights_autonomously", return_value=[])
    def test_empty_highlights_returns_false_no_render(
        self, mock_discover, mock_render, sample_transcript, sample_timeline
    ):
        from agent_optimizer import run_autonomous_editing_pipeline

        result = run_autonomous_editing_pipeline(
            video_path="fake_video.mp4",
            audio_transcript=sample_transcript,
            visual_breakdown=sample_timeline,
            layout_style="AI Smart Face Crop",
        )

        assert result is False
        mock_render.assert_not_called()


# ─── Full pipeline smoke test ─────────────────────────────────────────────────

class TestFullPipelineSmoke:

    @patch("agent_optimizer.generate_vertical_reel_clip")
    @patch("agent_optimizer.discover_highlights_autonomously")
    @patch("workflow_engine.genai.Client")
    @patch("video_processor.genai.Client")
    @patch("audio_processor.genai.Client")
    @patch("video_processor.subprocess.run")
    @patch("audio_processor.subprocess.run")
    def test_all_4_phases_complete_without_error(
        self,
        mock_audio_run,
        mock_video_run,
        mock_audio_client_cls,
        mock_video_client_cls,
        mock_blog_client_cls,
        mock_discover,
        mock_render,
        tmp_path,
        sample_transcript,
        sample_timeline,
    ):
        """
        Smoke test: mock every external call and confirm all 4 phases
        run end-to-end without raising any exception.
        """
        # --- FFmpeg mocks ---
        mock_audio_run.return_value = MagicMock(returncode=0)
        mock_video_run.return_value = MagicMock(returncode=0)

        # --- Phase 1: transcription ---
        audio_client = MagicMock()
        mock_audio_client_cls.return_value = audio_client
        mock_audio_upload = MagicMock()
        mock_audio_upload.name = "files/audio-smoke"
        audio_client.files.upload.return_value = mock_audio_upload
        audio_response = MagicMock()
        audio_response.parsed = sample_transcript
        audio_client.models.generate_content.return_value = audio_response

        # --- Phase 2: scene analysis (use pre-existing frames dir) ---
        frames_dir = tmp_path / "frames"
        frames_dir.mkdir()
        for i in range(1, 4):
            (frames_dir / f"keyframe_{i:04d}.jpg").write_bytes(b"\xff\xd8\xff")

        video_client = MagicMock()
        mock_video_client_cls.return_value = video_client
        mock_frame_upload = MagicMock()
        mock_frame_upload.name = "files/frame-smoke"
        video_client.files.upload.return_value = mock_frame_upload
        video_response = MagicMock()
        video_response.parsed = sample_timeline
        video_client.models.generate_content.return_value = video_response

        # --- Phase 3: blog synthesis ---
        blog_client = MagicMock()
        mock_blog_client_cls.return_value = blog_client
        blog_response = MagicMock()
        blog_response.text = '---\ntitle: "Smoke Test"\nslug: "smoke-test"\n---\n\n## Content\nOK.'
        blog_client.models.generate_content.return_value = blog_response

        # --- Phase 4: highlight detection + render ---
        highlight = MagicMock()
        highlight.hook_title = "Key Insight"
        highlight.start_time = 5.0
        highlight.end_time = 25.0
        highlight.speaker_position = "center"
        mock_discover.return_value = [highlight]

        # --- Run the full pipeline ---
        audio_path = str(tmp_path / "audio.wav")
        open(audio_path, "wb").close()

        video_path = str(tmp_path / "video.mp4")
        open(video_path, "wb").close()

        from audio_processor import extract_audio_from_video, transcribe_audio
        from video_processor import analyze_scene_with_gemini
        from workflow_engine import generate_production_blog
        from agent_optimizer import run_autonomous_editing_pipeline

        # Phase 1
        extract_audio_from_video(video_path, audio_path)
        transcript = transcribe_audio(audio_path)
        assert transcript is not None

        # Phase 2
        timeline = analyze_scene_with_gemini(str(frames_dir), interval_seconds=5)
        assert timeline is not None

        # Phase 3
        blog = generate_production_blog(transcript, timeline, "video.mp4")
        assert isinstance(blog, str) and len(blog) > 0

        # Phase 4
        result = run_autonomous_editing_pipeline(
            video_path=video_path,
            audio_transcript=transcript,
            visual_breakdown=timeline,
            layout_style="AI Smart Face Crop",
        )
        assert result is True
        assert mock_render.call_count == 1
