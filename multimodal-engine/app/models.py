# app/models.py
from pydantic import BaseModel, Field
from typing import List

class WordTimestamp(BaseModel):
    word: str = Field(description="The word or short fragment spoken.")
    start: float = Field(description="Exact start time in seconds.")
    end: float = Field(description="Exact end time in seconds.")

class AudioSegment(BaseModel):
    start_time: float = Field(description="Segment start time in seconds.")
    end_time: float = Field(description="Segment end time in seconds.")
    text: str = Field(description="The clean text transcription of this segment.")
    words: List[WordTimestamp] = Field(description="Individual word breakdown with timestamps.")

class StructuredTranscript(BaseModel):
    segments: List[AudioSegment]

class VideoFrameMoment(BaseModel):
    timestamp_seconds: float = Field(description="The timestamp of this keyframe based on its position.")
    visual_description: str = Field(description="Detailed summary of visual activity.")

class ChronologicalVisualTimeline(BaseModel):
    timeline: List[VideoFrameMoment]
