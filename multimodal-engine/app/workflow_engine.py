import os
import sys
from datetime import datetime
from dotenv import load_dotenv
from google import genai
from clip_extractor import stage_1_semantic_filter
from reel_generator import stage_2_visual_verification

APP_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(APP_DIR, ".."))

DATA_DIR = os.path.join(PROJECT_ROOT, "data")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "output")
CLIPS_DIR = os.path.join(OUTPUT_DIR, "clips")
TEMP_DIR = os.path.join(DATA_DIR, "temp_verification_slices")

# Guarantee that folder path hierarchies exist on disk before invoking downstream tools
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(CLIPS_DIR, exist_ok=True)
os.makedirs(TEMP_DIR, exist_ok=True)

# Setup system environment routing parameters
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
load_dotenv()

# High-precision production component imports
from audio_processor import extract_audio_from_video, transcribe_audio, save_transcript_todisk
from video_processor import extract_keyframes, analyze_scene_with_gemini, generate_vertical_reel_clip
from models import StructuredTranscript, ChronologicalVisualTimeline

def generate_production_blog(audio_transcript: StructuredTranscript, visual_breakdown: ChronologicalVisualTimeline, video_name: str) -> str:
    """Synthesizes structured visual mappings and text arrays into a markdown blog post."""
    print(" [3/4] Orchestrating final multimodal content synthesis via Gemini...")
    client = genai.Client()
    current_date = datetime.now().strftime("%Y-%m-%d")

    # Pass the serialized clean string models to preserve structural hierarchy inside the token prompt space
    prompt = f"""
    You are an expert technical content writer and SEO documentation engineer.
    
    I am providing you with two distinct data inputs extracted from the video file '{video_name}':
    1. TIMESTAMPED AUDIO TRANSCRIPT (JSON Mapping):
    ---
    {audio_transcript.model_dump_json(indent=2)}
    ---
    
    2. CHRONOLOGICAL VISUAL TIMELINE (JSON Mapping):
    ---
    {visual_breakdown.model_dump_json(indent=2)}
    ---
    
    CRITICAL STRUCTURE REQUIREMENT:
    You MUST begin your response with exactly this standard YAML Front Matter block (do not add backticks, trailing syntax, or markdown wrappers around it):
    ---
    title: "Generate a catchy, high-impact, SEO-optimized title here"
    date: "{current_date}"
    tags: ["Artificial Intelligence", "Tutorial", "Tech Guide"]
    category: "Technology"
    author: "Multimodal AI Engine"
    description: "Write a short, engaging 150-character meta description summary here for search previews."
    slug: "generate-a-clean-url-friendly-lowercase-slug-here"
    ---
    
    Following the front-matter block, write the blog post document using these strict rules:
    - Write a short introduction explaining the core software/concept demonstrated in the clip.
    - Break the content down into logical sections using clear H2 (##) and H3 (###) headers.
    - Incorporate structural elements like bullet points, summary tables, and bold code blocks cleanly.
    - Blend the visual timeline shifts smoothly with the spoken words so it reads like a comprehensive, standalone web tutorial.
    - Highlight keyboard shortcuts, timestamps, or interface menus mentioned on screen using bold text.
    - Omit call-to-action video catchphrases at the very end (e.g., discard speech elements asking to "like, share, comment, and subscribe"). 
    
    Do not add conversational commentary—return ONLY the markdown content.
    """

    try:
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=[prompt]
        )
        return response.text
    except Exception as e:
        print(f"🚨 Synthesis Pipeline Fatal Exception Error: {str(e)}")
        raise e

def run_integrated_pipeline(video_path: str):
    """The central orchestration engine running Stage 1 filtering and asset rendering."""
    print("🚀 ----- Starting Integrated Multimodal Content Generation Pipeline -----")

    # Hardcoded configurations converted to strict, upgraded extensions
    audio_output_path = os.path.join(DATA_DIR, "extracted_audio.wav") 
    frames_dir = os.path.join(DATA_DIR, "extracted_frames") 
    blogs_output_path = os.path.join(OUTPUT_DIR, "how_multimodals_work_blog.md") 
    transcript_json_path = os.path.join(DATA_DIR, "transcript.json") 

    # -------------------------------------------------------------
    # PHASE 1: Speech-To-Text Timeline Assembly
    # -------------------------------------------------------------
    extract_audio_from_video(video_path, audio_output_path)
    audio_transcript: StructuredTranscript = transcribe_audio(audio_output_path)
    
    if not audio_transcript:
        print(" Pipeline stopped: Audio transcription layer extraction violation.")
        return

    save_transcript_todisk(audio_transcript, transcript_json_path)

    # -------------------------------------------------------------
    # PHASE 2: Chronological Visual Timeline Assembly
    # -------------------------------------------------------------
    extract_keyframes(video_path, frames_dir, interval_seconds=5) 
    visual_breakdown: ChronologicalVisualTimeline = analyze_scene_with_gemini(frames_dir, interval_seconds=5)

    if not visual_breakdown:
        print(" Pipeline stopped: Visual tracking array extraction violation.")
        return

    # -------------------------------------------------------------
    # PHASE 3: Content Marketing Synthesis (Long-Form Blog Asset)
    # -------------------------------------------------------------
    final_blog_content = generate_production_blog(audio_transcript, visual_breakdown, os.path.basename(video_path))

    if final_blog_content:
        with open(blogs_output_path, "w", encoding="utf-8") as f:
            f.write(final_blog_content)
        print(f" System Success: Multimodal blog post generated at: {blogs_output_path}")
    
    # -------------------------------------------------------------
    # PHASE 4: Short-Form Reel Extractor (Two-Stage Verification)
    # -------------------------------------------------------------
    print(" 🎬 [4/4] Activating Two-Stage Filtering Highlight Detection Routine...")

    # 1. Run the Stage 1 text-based check
    candidate_clips = stage_1_semantic_filter(audio_transcript, visual_breakdown)
    
    # 2. Run the Stage 2 multi-modal verification step
    verified_clips = stage_2_visual_verification(video_path, candidate_clips)
    
    # 3. Render the verified shorts using our 9:16 vertical crop filter
    for idx, clip in enumerate(verified_clips):
        reel_path = os.path.join(CLIPS_DIR, f"viral_reel_{idx + 1}.mp4") 
        generate_vertical_reel_clip(video_path, clip.start_time, clip.end_time, reel_path)

    print("🏁 Pipeline run completed successfully.")

if __name__ == "__main__":
    TARGET_VIDEO = "../data/sample.mp4"

    if not os.path.exists(TARGET_VIDEO):
        print(f"⚠️ Verification Guard: Target clip missing at path location: {TARGET_VIDEO}")
    else:
        run_integrated_pipeline(TARGET_VIDEO)
