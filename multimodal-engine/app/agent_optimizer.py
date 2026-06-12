import os
from pydantic import BaseModel, Field
from typing import List
from google import genai
from google.genai import types
from dotenv import load_dotenv

# Hook up our high-performance production pipeline assets built over the week
from audio_processor import StructuredTranscript
from video_processor import ChronologicalVisualTimeline, generate_vertical_reel_clip
from utils import retry_with_backoff

load_dotenv()

# =====================================================================
# STAGE 3: PYDANTIC STRUCTURAL DATA VALIDATION SCHEMA
# =====================================================================
class ProductionHighlight(BaseModel):
    """
    Defines the exact numeric and descriptive structure required for 
    programmatic cutting execution without text parsing risks.
    """
    start_time: float = Field(description="The exact timestamp in absolute seconds where the highlight hook begins.")
    end_time: float = Field(description="The exact timestamp in absolute seconds where the highlight hook ends.")
    hook_title: str = Field(description="A short, viral, click-ready title headline for the clip description.")
    justification: str = Field(description="A brief explanation detailing why this segment has high information density or virality.")
    speaker_position: str = Field(description="Where is the main speaker/subject horizontally on screen? Must return exactly 'center', 'left', or 'right'.")


class HighlightAnalysisResult(BaseModel):
    """
    Wraps isolated segments directly into a verified, iterable type-safe collection list.
    """
    highlights: List[ProductionHighlight]


# =====================================================================
# STAGE 1 & 2: THE CASCADE ORCHESTRATION LAYER
# =====================================================================
def discover_highlights_autonomously(
    audio_transcript: StructuredTranscript, 
    visual_breakdown: ChronologicalVisualTimeline
) -> List[ProductionHighlight]:
    """
    Acts as our elite agentic evaluator, matching voice and vision vectors 
    to extract reliable, validated highlight ranges.
    """
    print("🧠 Step 1 & 2: Initializing Semantic Multi-Stage Filter & Agent Optimizer Pipeline...")
    
    client = genai.Client()
    
    macro_prompt = f"""
    You are an elite AI media strategist, virality analyst, and automated clip director.
    
    Analyze the following two multimodal timeline context data records extracted from a video:
    
    AUDIO TRANSCRIPT SCHEMAS:
    {audio_transcript.model_dump_json(indent=2) if isinstance(audio_transcript, StructuredTranscript) else audio_transcript}
    
    VISUAL CHRONOLOGICAL SCHEMAS:
    {visual_breakdown.model_dump_json(indent=2) if isinstance(visual_breakdown, ChronologicalVisualTimeline) else visual_breakdown}
    
    YOUR TASK:
    1. Cross-reference spoken insights against visual screen changes.
    2. Pinpoint the top 2 highest-value standalone 'golden moments' or 'hooks'. 
    3. Select segments with dense educational updates, punchy software demos, or high emotional pacing.
    4. Provide raw absolute numbers in seconds for start_time and end_time.
    
    CRITICAL TIME SANITY CRITERIA:
    - start_time and end_time MUST be raw float numbers tracking absolute seconds (e.g., 22.5). Never write standard clock text strings like '00:01:15'.
    - Keep clips focused and concise. The duration (end_time minus start_time) must fall between 15.0 and 45.0 seconds total.
    """
    
    try:
        print("🤖 Requesting Gemini processing with zero-hallucination Pydantic constraint filters...")

        def excute_call():
            return client.models.generate_content(
                model="gemini-2.5-flash",
            contents=[macro_prompt],
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=HighlightAnalysisResult,
                temperature=0.1, # Keep temperature low to prevent numeric drift
            )

            )
        response = retry_with_backoff(excute_call)

        # Access the typed object directly using response.parsed to avoid raw json parsing errors
        validated_data: HighlightAnalysisResult = response.parsed
        print(f"✅ Structural layout validation checks cleared. Isolated {len(validated_data.highlights)} entries.")
        return validated_data.highlights
        
    except Exception as e:
        print(f"🚨 Agent Optimizer Inference Failure: {str(e)}")
        return []


def run_autonomous_editing_pipeline(
    video_path: str, 
    audio_transcript: StructuredTranscript, 
    visual_breakdown: ChronologicalVisualTimeline,
    layout_style: str
) -> bool:
    """
    Iterates over the validated objects array loop and routes timestamps 
    straight to the video processing cutting filters.
    """
    print("\n🚀 --- STARTING STAGE 3: AUTONOMOUS LOGIC INTEGRATION EXECUTION --- 🚀")
    
    # 1. Pull verified numeric markers via our AI engine
    discovered_highlights = discover_highlights_autonomously(audio_transcript, visual_breakdown)
    if not discovered_highlights:
        return False
        
    os.makedirs("../output/clips", exist_ok=True)
    
    for index, highlight in enumerate(discovered_highlights):
        print(f"🎬 Processing Clip #{index + 1}: [{highlight.hook_title}]")
        safe_title = "".join(c for c in highlight.hook_title if c.isalnum() or c in (' ', '_')).rstrip().replace(' ', '_').lower()
        output_reel = f"../output/clips/auto_reel_{index + 1}_{safe_title}.mp4"
        
        # Determine the string instruction parameter for FFmpeg
        # If user picked Blurred Stack, override position and pass "blurred"
        if "Blurred Stack" in layout_style:
            chosen_mode = "blurred"
        else:
            chosen_mode = highlight.speaker_position # Use 'left', 'center', or 'right' from Gemini
            
        try:
            generate_vertical_reel_clip(
                video_path=video_path,
                start_time=highlight.start_time,
                end_time=highlight.end_time,
                output_path=output_reel,
                render_mode=chosen_mode # 🆕 Pass the calculated layout instruction
            )
            print(f"   Saved asset: {output_reel}\n")
        except Exception as cut_err:
            print(f"   ⚠️ FFmpeg could not compile clip. Reason: {cut_err}")
        
    return True

# =====================================================================
# ISOLATED RUN HARNESS PLACEHOLDER
# =====================================================================
if __name__ == "__main__":
    SOURCE_VIDEO = "../data/sample.mp4"
    print(f"Agent Optimizer initialized. Production schemas locked down.")
