import subprocess, os, glob
from google import genai
from dotenv import load_dotenv

load_dotenv()

def extract_keyframe(video_path, keyframe_output_dir, interval_seconds=5):
    print(f"[1] Extracting keyframes from video: {video_path}")

    if not os.path.exists(keyframe_output_dir):
        os.makedirs(keyframe_output_dir)
    else:
        existing_files = glob.glob(os.path.join(keyframe_output_dir, "*.jpg"))
        for f in existing_files:
            os.remove(f)
    output_pattern = os.path.join(keyframe_output_dir, "keyframe_%04d.jpg")

    fps_filter = f"fps= 1/{interval_seconds}"

    command = [
        "ffmpeg",
        "-i", video_path,
        "-vf", fps_filter,
        "-q:v", "2",
        output_pattern
    ]

    try:
        subprocess.run(command, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        print(f"Keyframes successfully extracted to: {keyframe_output_dir}")

        extracted_counts = len(glob.glob(os.path.join(keyframe_output_dir, "*.jpg")))
        print(f"DownSappling total generated: {extracted_counts}")
        return extracted_counts
    except subprocess.CalledProcessError as e:
        print(f"ffmpeg keyfrmae extrction fsiled")
        raise e
    

def analyze_scene_with_gemini(frame_dir):
    print(f"[2] Initializing visaula analyzing")
    client = genai.Client()

    frame_path = sorted(glob.glob(os.path.join(frame_dir, "*.jpg")))

    if not frame_dir:
        print(f"no frame")
        return None
    
    frame_uploaded = []
    try:
        print(f"uploading {len(frame_path)} frame to clod analyze")
        for path in frame_path:
            upload_frame = client.files.upload(file=path)
            frame_uploaded.append(upload_frame)

        print("all visual frame are Sucessfuly in the cloud")

        prompt = """
        You are an expert video analysis system. You are given a sequential list of image frames extracted 
        from a video at an interval of every 5 seconds.
        
        Analyze these frames chronologically and write a highly descriptive breakdown summarizing what is 
        visually happening on screen. Focus on slide changes, screen displays, objects, text captions, or 
        human actions. Format your response into short, bulleted timeline chapters.
        """
        
        print("🤖 Processing Vision-LLM inference (Analyzing images)...")
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=frame_uploaded + [prompt]
        )
        
        print("🗑️ Scrubbing visual file references from cloud memory storage...")
        for cloud_file in frame_uploaded:
            client.files.delete(name=cloud_file.name)
            
        return response.text
    except Exception as e:
        print(f"Vission Infrenec as {e}")
        return None


if __name__ == "__main__":
    VIDEO_PATH = "../data/sample.mp4"
    FRAMES_DIR = "../data/extracted_frames"
    
    if not os.path.exists(VIDEO_PATH):
        print(f"⚠️ Test Guard: Please ensure your file exists at: {VIDEO_PATH}")
    else:
        # 1. Run frame slicer
        extract_keyframe(VIDEO_PATH, FRAMES_DIR, interval_seconds=5)
        
        # 2. Run vision model analysis
        visual_summary = analyze_scene_with_gemini(FRAMES_DIR)
        
        if visual_summary:
            print("\n🤖 --- GEMINI VISUAL BREAKDOWN ---")
            print(visual_summary)
            print("----------------------------------")

