import subprocess, os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


def generate_vertical_reel(video_input_path, start_time, duration_seconds, output_real_path):

    print("Initializing Reel Generator")
    output_dir = os.path.dirname(output_real_path)

    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    if os.path.exists(output_real_path):
        try:
           os.remove(output_real_path)
        except PermissionError:
            print("File is locked, forcing overwrite with ffmpeg")


    crop_filter = "crop=ih*(9/16):ih:(iw-ow)/2:0"
    command = [
        "ffmpeg",
        "-y",
        "-ss", str(start_time),
        "-i", video_input_path,
        "-t", str(duration_seconds),
        "-vf", crop_filter,
        "-c:a", "aac",
        output_real_path

    ]

    try:
        print("Re-muxing and cropping landscape frame to 9:16 Vertical stream")
        subprocess.run(command, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        print("Reel compiled Sucessfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"Reel Generator Error: Processing Failed. Reason {e}")

if __name__ == "__main__":
    SOURCE = "../data/sample.mp4"
    target_real = "../data/extracted_real/vertical_short_real.mp4"

    if os.path.exists(SOURCE):
        generate_vertical_reel(SOURCE, start_time="00:00:15", duration_seconds=10, output_real_path=target_real)

