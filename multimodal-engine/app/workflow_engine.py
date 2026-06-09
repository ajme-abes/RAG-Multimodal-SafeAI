import os, sys
from dotenv import load_dotenv
from google import genai
from openai import OpenAI

from audio_processor import extract_audio_from_video, transcribe_audio, save_transcript_todisk
from video_processor import extract_keyframes, analyze_scene_with_gemini

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
load_dotenv()


def generate_production_blog(audio_transcript, visual_breakdown):

    print(f"[3]: Orchastrating final content Synthesis via gemin")

    client = genai.Client()

    prompt = f"""
    You are an expert technical content writer and software documentation engineer.
    
    I am providing you with two distinct data inputs extracted from a video tutorial:
    1. RAW AUDIO TRANSCRIPT:
    ---
    {audio_transcript}
    ---
    
    2. CHRONOLOGICAL VISUAL BREAKDOWN:
    ---
    {visual_breakdown}
    ---
    
    TASK:
    Synthesize these two inputs into a comprehensive, high-quality, step-by-step Technical Blog Post 
    written in Markdown. 
    
    STRUCTURE RULES:
    - Add a catchy title at the top (#).
    - Write a short introduction explaining what software is being demonstrated.
    - Break the content down into logical step-by-step sections using clear headings (##).
    - Blend the visual timeline actions smoothly with the spoken words so it reads like a cohesive tutorial.
    - Highlight specific keyboard shortcuts, timestamps, or interface menus mentioned on screen using code blocks or bold text.
    - End with a summary conclusion.
    
    Do not add conversational commentary—return ONLY the markdown content.
    """

    try:
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=[prompt]
        )

        return response.text
    except Exception as e:
        print(f"Error during content synthesis: {str(e)}")
        return None

def run_integrated_pipeline(video_path):

    print(f"🚀 -----Starting Integrated Multimodal Content Generation Pipeline-----------")

    audio_output_path = "../data/extracted_audio.mp3"
    frames_dir = "../data/extracted_frames"
    blogs_output_path = "../output/how_multimodals_work_blog.md"

    if not os.path.exists("../output"):
        os.makedirs("../output")

    # phase 1 extracted the audio
    extract_audio_from_video(video_path, audio_output_path)
    audio_transcript = transcribe_audio(audio_output_path)

    if not audio_transcript:
        print("Pipeline stopped: audio transcribe layer Faileed")
        return None
    
    #phase 2: Visual stream trace

    extract_keyframes(video_path, frames_dir, interval_seconds=5) 
    visual_breakdown = analyze_scene_with_gemini(frames_dir)

    if not visual_breakdown:
        print("Pipeline stopped: visual analysis layer Faileed")
        return None
    
    #phase 3 final Blog synthesis

    final_blog_content = generate_production_blog(audio_transcript, visual_breakdown)

    if final_blog_content:
        with open(blogs_output_path, "w", encoding="utf-8") as f:
            f.write(final_blog_content)
        print(f"\n System Success Complete multimodal blog post generated at {blogs_output_path}")
    else:
        print("Pipeline stopped: final content synthesis layer Faileed")

if __name__ == "__main__":
    TARGET_VIDEO = "../data/sample.mp4"

    if not os.path.exists(TARGET_VIDEO):
        print(f"⚠️ Verification Guard: Please confirm your testing file is ready at: {TARGET_VIDEO}")
    else:
        run_integrated_pipeline(TARGET_VIDEO)
    
    
