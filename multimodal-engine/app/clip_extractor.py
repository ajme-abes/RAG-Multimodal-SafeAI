import subprocess, os

def extract_video_clip(video_path, start_time, duration_seconds, output_path):

    print("Initializing video clip extraction...")

    output_dir = os.path.dirname(output_path)

    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    if os.path.exists(output_path):
        os.remove(output_path)


    command = [
        "ffmpeg",
        "-ss", str(start_time),
        "-i", video_path,
        "-t", str(duration_seconds),
        "-c:v", "libx264",
        "-c:a", "aac",
        output_path
    ]

    try:
        subprocess.run(command, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        print(f"Video clip successfully extracted to: {output_path}")

        return True
    except subprocess.CalledProcessError as e:
        print(f"ffmpeg video clip extraction failed: {e.stderr.decode()}")
        raise e
    
if __name__ == "__main__":
    SOURCE_VIDEO = "../data/sample.mp4"
    TARGET_CLIP = "../data/extracted_clip/highlight_clip_1.mp4"

    if os.path.exists(SOURCE_VIDEO):
        extract_video_clip(SOURCE_VIDEO, start_time="00:00:20", duration_seconds=15, output_path=TARGET_CLIP)
    else:
        print("Source video not found.")