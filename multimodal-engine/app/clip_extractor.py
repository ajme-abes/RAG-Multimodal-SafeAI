import os
from typing import List
from pydantic import BaseModel, Field
from google import genai
from google.genai import types
from audio_processor import StructuredTranscript
from video_processor import ChronologicalVisualTimeline

# Define a clean structural format for Stage 1 Candidate Output
class CandidateHighlight(BaseModel):
    title: str = Field(description="A hook-driven viral title candidate.")
    start_time: float = Field(description="The timestamp in seconds from the original video.")
    end_time: float = Field(description="The timestamp in seconds from the original video.")
    semantic_hook_reason: str = Field(description="Why this specific text fragment makes a great standalone clip.")

class CandidateHighlightBatch(BaseModel):
    candidates: List[CandidateHighlight]

def stage_1_semantic_filter(audio_transcript: StructuredTranscript, visual_breakdown: ChronologicalVisualTimeline) -> List[CandidateHighlight]:
    """Stage 1: Filters out filler text and extracts the best 30-60s candidate windows."""
    print(" 🔍 [Stage 1] Running Heavy Text-Based Semantic Filter via Gemini Flash...")
    client = genai.Client()

    prompt = f"""
    You are an expert video editor and viral media strategist.
    Analyze the following transcript and chronological timeline data from a video.
    Identify the top high-impact candidate segments that have a strong hook within the first 3 seconds, a unified topic, and a duration between 30 and 60 seconds.
    
    AUDIO TRANSCRIPT TRANSCRIPTION:
    {audio_transcript.model_dump_json(indent=2)}
    
    CHRONOLOGICAL VISUAL TIMELINE:
    {visual_breakdown.model_dump_json(indent=2)}
    
    Ensure all timestamps align accurately with the source numbers provided. Return only validated structured data.
    """

    try:
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=[prompt],
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=CandidateHighlightBatch,
                temperature=0.2, # Low temperature to ensure time accuracy
            ),
        )
        
        parsed_response: CandidateHighlightBatch = response.parsed
        print(f" Found {len(parsed_response.candidates)} candidate highlights for Stage 2 verification.")
        return parsed_response.candidates

    except Exception as e:
        print(f"🚨 Stage 1 Semantic Extraction Error: {str(e)}")
        raise e
