import subprocess
import os
import glob
import base64
from typing import List, Optional
from pydantic import BaseModel, Field
from google import genai
from google.genai import types
from dotenv import load_dotenv
from openai import OpenAI
from utils import retry_with_backoff
from models import ChronologicalVisualTimeline, VideoFrameMoment

load_dotenv()

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

def extract_keyframes(video_path: str, keyframe_output_dir: str, interval_seconds: int = 5) -> int:
    """Extracts high-quality keyframes at precise uniform chronological markers."""
    print(f"[1] Extracting keyframes from video source: {video_path}")

    if not os.path.exists(keyframe_output_dir):
        os.makedirs(keyframe_output_dir)
    else:
        existing_files = glob.glob(os.path.join(keyframe_output_dir, "*.jpg"))
        for f in existing_files:
            os.remove(f)

    # Output matches exact time mapping calculations (keyframe_0001 = interval * 1)
    output_pattern = os.path.join(keyframe_output_dir, "keyframe_%04d.jpg")
    fps_filter = f"fps=1/{interval_seconds}"

    command = [
        "ffmpeg", "-i", video_path,
        "-vf", fps_filter,
        "-q:v", "2", # High quality JPEG setting
        output_pattern
    ]

    try:
        subprocess.run(command, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        extracted_counts = len(glob.glob(os.path.join(keyframe_output_dir, "*.jpg")))
        print(f" Downsampling completed. Generated {extracted_counts} chronological snapshots.")
        return extracted_counts
    except subprocess.CalledProcessError as e:
        print("FFmpeg keyframe generation pipeline fatal execution error.")
        raise e

def encode_image_to_base64(image_path: str) -> str:
    """Encodes structural image data to base64 for fallback processing models."""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def run_openai_fallback(frame_paths: List[str], prompt: str) -> Optional[str]:
    """Corrected Fallback Engine: Packages frames simultaneously into one context payload."""
    print("[Fallback] Routing Multi-Modal Visual analysis to OpenAI...")
    try:
        openai_client = OpenAI()
        content_payload = [{"type": "text", "text": prompt}]

        # Append all frames into a single payload to analyze the sequence together
        for path in frame_paths:
            base64_image = encode_image_to_base64(path)
            content_payload.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}
            })

        response = openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": content_payload}],
            max_tokens=1500
        )
        print(" OpenAI Vision Processing execution step succeeded.")
        return response.choices[0].message.content
    except Exception as e:
        print(f"🚨 Critical Failure: OpenAI Fallback failed. Reason: {e}")
        return None

def analyze_scene_with_gemini(frame_dir: str, interval_seconds: int = 5) -> Optional[ChronologicalVisualTimeline]:
    """Uploads sequential frames and converts them into structured visual data arrays."""
    print(f"[2] Initiating visual narrative analysis out of directory: {frame_dir}")
    client = genai.Client()

    frame_paths = sorted(glob.glob(os.path.join(frame_dir, "*.jpg")))
    if not frame_paths:
        print(" Empty frame directory provided. Aborting execution step.")
        return None

    frame_uploaded = []
    try:
        print(f"Streaming {len(frame_paths)} temporal frames to GenAI Cloud Asset Manager...")
        for path in frame_paths:
            upload_frame = client.files.upload(file=path)
            frame_uploaded.append(upload_frame)

        prompt = f"""
        You are an expert multi-modal context engine. You are looking at image frames extracted in sequence 
        exactly every {interval_seconds} seconds.
        
        Generate a chronological timeline matching every frame to its real timeline position. Calculate 
        timestamp_seconds accurately. For frame_0001, timestamp_seconds is {interval_seconds * 1}. For 
        frame_0002, it is {interval_seconds * 2}, and so on.
        Describe any visual changes, on-screen slide text, speaker facial actions, or object tracking details.
        """

        def execute_call():
            return client.models.generate_content(
                model="gemini-2.5-flash",
                contents=frame_uploaded + [prompt],
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    response_schema=ChronologicalVisualTimeline,
                    temperature=0.1
                ),
            )

        response = retry_with_backoff(execute_call)
        print(" Gemini visual analysis execution step succeeded.")
        return response.parsed
    except Exception as gemini_error:
        print(f"🚨 Primary Gemini Cluster failed. Reason: {gemini_error}")
        fallback_prompt = f"Analyze these frames in sequential order. Provide a visual summary every {interval_seconds} seconds."
        raw_text_fallback = run_openai_fallback(frame_paths, fallback_prompt)
        
        from models import ChronologicalVisualTimeline, VideoFrameMoment
        
        fallback_desc = raw_text_fallback if raw_text_fallback else "Visual capture processing failure."
        return ChronologicalVisualTimeline(timeline=[
            VideoFrameMoment(
                timestamp_seconds=float(i * interval_seconds),
                visual_description=f"[OpenAI Fallback Data]: {fallback_desc}"
            ) for i in range(1, len(frame_paths) + 1)
        ])
    finally:
        # Guarantee cleanup loops execute under all load behaviors
        if frame_uploaded:
            print("🗑️ Clearing asset instances from Cloud storage buckets...")
            for cloud_file in frame_uploaded:
                try:
                    client.files.delete(name=cloud_file.name)
                except Exception:
                    pass

def generate_vertical_reel_clip(video_path: str, start_time: float, end_time: float, output_path: str, render_mode: str = "center"):
    """Cuts and crops horizontal 16:9 video source files directly into a 9:16 vertical workspace canvas."""
    print(f"[3] Slicing and re-centering vertical layout from {start_time}s to {end_time}s")
    
    if start_time < 0:
        raise ValueError(f"❌ Pipeline Range Violation: start_time ({start_time}s) cannot be negative.")
    if end_time <= start_time:
        raise ValueError(f"❌ Pipeline Range Violation: end_time ({end_time}s) must occur after start_time ({start_time}s).")
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"❌ Pipeline Resource Missing: Source target not located at {video_path}")
        
    output_directory = os.path.dirname(output_path)
    if output_directory:
        os.makedirs(output_directory, exist_ok=True)
        
    if os.path.exists(output_path):
        os.remove(output_path)


    if render_mode == "blurred":
        # Mode A: Dual stacked layers. Keeps full 16:9 tutorial frame completely visible.
        video_filter_chain = (
            "split[original][background];"
            "[background]scale=1080:1920,boxblur=20:10[blurred];"
            "[original]scale=1080:-1[scaled];"
            "[blurred][scaled]overlay=(main_w-overlay_w)/2:(main_h-overlay_h)/2"
        )
    else:
        # Mode B: AI Smart Face Crop. Crops to full-screen vertical, framing based on Gemini host insights.
        if render_mode == "left":
            x_offset = "0"
        elif render_mode == "right":
            x_offset = "iw-ow"
        else:
            x_offset = "(iw-ow)/2"
            
        video_filter_chain = f"crop=ih*(9/16):ih:{x_offset}:0,scale=1080:1920"

    command = [
        "ffmpeg", "-y",
        "-ss", str(start_time),
        "-to", str(end_time),
        "-i", video_path,
        "-vf", video_filter_chain,
        "-c:v", "libx264",
        "-c:a", "aac",
        "-b:v", "2M",
        output_path
    ]

    try:
        subprocess.run(command, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        print(f" Vertical video successfully rendered to file target: {output_path}")
    except subprocess.CalledProcessError as e:
        print(f"FFmpeg render pipeline failed: {e.stderr.decode()}")
        raise e

if __name__ == "__main__":
    VIDEO_PATH = "../data/sample.mp4"
    FRAMES_DIR = "../data/extracted_frames"
    
    if not os.path.exists(VIDEO_PATH):
        print(f"Test file placeholder missing at: {VIDEO_PATH}")
    else:
        extract_keyframes(VIDEO_PATH, FRAMES_DIR, interval_seconds=5)
        timeline_data = analyze_scene_with_gemini(FRAMES_DIR, interval_seconds=5)
        if timeline_data:
            print("\n🤖 --- STRUCTURAL CHRONO TIMELINE DATA VERIFIED ---")
            print(timeline_data)
