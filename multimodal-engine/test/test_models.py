"""
Tests for Pydantic data models (models.py)
These are pure unit tests — no API calls, no FFmpeg, no files needed.
"""
import pytest
import sys
import os

# Allow imports from app/ without installing as a package
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../app")))

from models import (
    WordTimestamp,
    AudioSegment,
    StructuredTranscript,
    VideoFrameMoment,
    ChronologicalVisualTimeline,
)


# ─── WordTimestamp ────────────────────────────────────────────────────────────

class TestWordTimestamp:
    def test_valid_construction(self):
        w = WordTimestamp(word="hello", start=0.0, end=0.45)
        assert w.word == "hello"
        assert w.start == 0.0
        assert w.end == 0.45

    def test_float_types_are_enforced(self):
        # Pydantic coerces int → float automatically
        w = WordTimestamp(word="hi", start=1, end=2)
        assert isinstance(w.start, float)
        assert isinstance(w.end, float)

    def test_missing_field_raises(self):
        with pytest.raises(Exception):
            WordTimestamp(word="oops")  # start and end missing


# ─── AudioSegment ─────────────────────────────────────────────────────────────

class TestAudioSegment:
    def _make_word(self, word="test", start=0.0, end=0.5):
        return WordTimestamp(word=word, start=start, end=end)

    def test_valid_construction(self):
        seg = AudioSegment(
            start_time=0.0,
            end_time=5.0,
            text="Hello world",
            words=[self._make_word()],
        )
        assert seg.text == "Hello world"
        assert len(seg.words) == 1

    def test_empty_words_list_is_valid(self):
        seg = AudioSegment(start_time=0.0, end_time=1.0, text="ok", words=[])
        assert seg.words == []

    def test_multiple_words(self):
        words = [
            self._make_word("Hi", 0.0, 0.3),
            self._make_word("there", 0.3, 0.7),
        ]
        seg = AudioSegment(start_time=0.0, end_time=1.0, text="Hi there", words=words)
        assert len(seg.words) == 2
        assert seg.words[1].word == "there"


# ─── StructuredTranscript ─────────────────────────────────────────────────────

class TestStructuredTranscript:
    def _make_segment(self, start=0.0, end=5.0, text="sample"):
        return AudioSegment(
            start_time=start,
            end_time=end,
            text=text,
            words=[WordTimestamp(word=text, start=start, end=end)],
        )

    def test_single_segment(self):
        t = StructuredTranscript(segments=[self._make_segment()])
        assert len(t.segments) == 1

    def test_multiple_segments(self):
        t = StructuredTranscript(
            segments=[
                self._make_segment(0.0, 5.0, "first"),
                self._make_segment(5.0, 10.0, "second"),
            ]
        )
        assert len(t.segments) == 2
        assert t.segments[0].text == "first"
        assert t.segments[1].text == "second"

    def test_empty_segments_is_valid(self):
        t = StructuredTranscript(segments=[])
        assert t.segments == []

    def test_serialise_to_json_and_back(self):
        original = StructuredTranscript(segments=[self._make_segment()])
        json_str = original.model_dump_json()
        restored = StructuredTranscript.model_validate_json(json_str)
        assert restored.segments[0].text == original.segments[0].text

    def test_float_timestamps_preserved_in_json(self):
        seg = self._make_segment(start=12.5, end=20.75)
        t = StructuredTranscript(segments=[seg])
        data = t.model_dump()
        assert data["segments"][0]["start_time"] == 12.5
        assert data["segments"][0]["end_time"] == 20.75


# ─── VideoFrameMoment ─────────────────────────────────────────────────────────

class TestVideoFrameMoment:
    def test_valid_construction(self):
        frame = VideoFrameMoment(
            timestamp_seconds=10.0,
            visual_description="Speaker gestures at whiteboard.",
        )
        assert frame.timestamp_seconds == 10.0

    def test_zero_timestamp_is_valid(self):
        frame = VideoFrameMoment(timestamp_seconds=0.0, visual_description="Intro slide.")
        assert frame.timestamp_seconds == 0.0

    def test_missing_description_raises(self):
        with pytest.raises(Exception):
            VideoFrameMoment(timestamp_seconds=5.0)


# ─── ChronologicalVisualTimeline ──────────────────────────────────────────────

class TestChronologicalVisualTimeline:
    def _make_frame(self, ts=5.0, desc="Frame description"):
        return VideoFrameMoment(timestamp_seconds=ts, visual_description=desc)

    def test_single_frame(self):
        tl = ChronologicalVisualTimeline(timeline=[self._make_frame()])
        assert len(tl.timeline) == 1

    def test_ordered_timestamps(self):
        tl = ChronologicalVisualTimeline(
            timeline=[
                self._make_frame(5.0, "frame 1"),
                self._make_frame(10.0, "frame 2"),
                self._make_frame(15.0, "frame 3"),
            ]
        )
        timestamps = [f.timestamp_seconds for f in tl.timeline]
        assert timestamps == sorted(timestamps), "Timeline should be in ascending order"

    def test_serialise_roundtrip(self):
        original = ChronologicalVisualTimeline(timeline=[self._make_frame(25.0, "desk shot")])
        restored = ChronologicalVisualTimeline.model_validate_json(original.model_dump_json())
        assert restored.timeline[0].timestamp_seconds == 25.0

    def test_empty_timeline_is_valid(self):
        tl = ChronologicalVisualTimeline(timeline=[])
        assert tl.timeline == []
