import os
import subprocess
import time
from typing import List
from pydantic import BaseModel, Field
from google import genai
from google.genai import types
from clip_extractor import CandidateHighlight
from video_processor import generate_vertical_reel_clip

# Define validation checks for our multi-modal confirmation layer
class VisualVerificationReport(BaseModel):
    is_visually_engaging: bool = Field(description="True if the visual pacing, facial cues, or action sequences are high quality.")
    audio_energy_score: int = Field(description="Score from 1 to 100 on speaking tone, excitement, and clear pacing.")
    final_relevance_decision: bool = Field(description="True if this clip passes all standards and is cleared for production rendering.")
    adjusted_start_time: float = Field(description="Optimized start position to avoid cutting off words or speaker context.")
    adjusted_end_time: float = Field(description="Optimized end position for clean pacing.")

def stage_2_visual_verification(video_path: str, candidates: List[CandidateHighlight]) -> List[CandidateHighlight]:
    """Stage 2: Runs multi-modal verification on target clips using video analytics to remove context drift."""
    print(" 🎥 [Stage 2] Running Multi-Modal Verification via Gemini Pro Video Cluster...")
    client = genai.Client()
    verified_production_clips = []

    # Create a temporary directory to store raw clip slices
    temp_dir = "../data/temp_verification_slices"
    os.makedirs(temp_dir, exist_ok=True)

    for idx, candidate in enumerate(candidates):
        print(f" Processing verification step for candidate clip #{idx+1}: {candidate.title}")
        
        temp_clip_path = os.path.join(temp_dir, f"candidate_slice_{idx}.mp4")
        duration = candidate.end_time - candidate.start_time

        # Quick FFmpeg slice to create a small evaluation file
        slice_cmd = [
            "ffmpeg", "-y", "-ss", str(candidate.start_time), "-t", str(duration),
            "-i", video_path, "-c:v", "libx264", "-c:a", "aac", temp_clip_path
        ]
        subprocess.run(slice_cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)

        # Upload the actual video clip slice directly into Gemini Pro
        print(f"   Uploading video slice segment payload to Gemini Cloud assets...")
        uploaded_video = client.files.upload(file=temp_clip_path)
        
        # Audio/Video multi-modal assets require processing time on the server before query execution
        while uploaded_video.state.name == "PROCESSING":
            time.sleep(2)
            uploaded_video = client.files.get(name=uploaded_video.name)

        if uploaded_video.state.name == "FAILED":
            print("   Cloud processing error for video file. Skipping entry.")
            continue

        prompt = """
        You are an expert director and visual quality assurance engine. 
        Analyze the audio energy, speaker presence, and visual changes in this short video file.
        Determine if it is engaging enough to be rendered as a short-form video.
        Provide subtle adjustments to the start and end offsets to ensure words are not cut off mid-sentence.
        """

        try:
            response = client.models.generate_content(
                model="gemini-2.5-pro", # Multi-modal execution requires the Pro engine
                contents=[uploaded_video, prompt],
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    response_schema=VisualVerificationReport,
                    temperature=0.1
                ),
            )
            
            report: VisualVerificationReport = response.parsed
            
            if report.final_relevance_decision and report.audio_energy_score > 60:
                print(f"    Passed Verification! Audio Energy Score: {report.audio_energy_score}/100")
                # Update the target with our verified timestamps
                candidate.start_time = report.adjusted_start_time + candidate.start_time
                candidate.end_time = report.adjusted_end_time + candidate.start_time
                verified_production_clips.append(candidate)
            else:
                print("    Failed Verification: Rejected due to flat visual energy levels.")

        except Exception as e:
            print(f"   Verification layer skip logic invoked: {e}")
        finally:
            # Clean up the cloud file asset right away
            client.files.delete(name=uploaded_video.name)
            if os.path.exists(temp_clip_path):
                os.remove(temp_clip_path)

    return verified_production_clips
