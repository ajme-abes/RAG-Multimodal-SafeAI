import subprocess, os
from dotenv import load_dotenv
from google import genai
load_dotenv()
def extract_audio_from_video(video_path, audio_output_path):

    print(f"[1] Demultplexing audio from video: {video_path}")

    if os.path.exists(audio_output_path):
        os.remove(audio_output_path)

    command = [
        "ffmpeg",
        "-i", video_path,
        "-vn", 
        "-acodec", "libmp3lame",
        "-q:a", "4",
        audio_output_path
    ]

    try:
        subprocess.run(command, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        print(f"Audio successfully extracted to: {audio_output_path}")
    except subprocess.CalledProcessError as e:
        print(f"Error during audio extraction: {e.stderr.decode()}")
        raise e

def transcribe_audio(audio_file_path):

    print(f"[2] Transcribing audio file: {audio_file_path}")

    client = genai.Client()
    print(f"Uploading Audio file payloade to GenAI")
    upload_audio = client.files.upload(file=audio_file_path)
    print(f"Audio file Uploaded Sucessfully")

    prompt = """
    You are an expert audio transcription system. 
    Transcribe the provided audio file with absolute accuracy. 
    Separate the transcript into readable paragraphs based on natural conversational pauses.
    Do not add introductory remarks or conversational commentary—return ONLY the final transcript text.
    """
    
    print("🤖 Processing Speech-to-Text inference (Waiting for model response)...")

    try:
        response = client.models.generate_content(
            model = "gemini-2.5-flash",
            contents = [upload_audio, prompt]
        )

        client.files.delete(name=upload_audio.name)
        print("Audio file deleted from GenAI storage.")

        return response.text.strip()
    except Exception as e:
        print(f"Error during transcription: {str(e)}")
        raise e
def save_transcript_todisk(transcript_text, output_text_path):

    print(f"[3] Saving transcript to disk: {output_text_path}")
    try:
          
        with open(output_text_path, "w", encoding="utf-8") as f:
            f.write(transcript_text)
        print(f"Transcript successfully saved to: {output_text_path}")
    except Exception as e:
        print(f"File I/O error while saving transcript: {str(e)}")

if __name__ == "__main__":
    video_file_path = "../data/sample.mp4"
    audio_file_path = "../data/extracted_audio.mp3"
    transcript_output_path = "../data/transcript.txt"

    if not os.path.exists(video_file_path):
        print(f"Error: Video file not found at path: {video_file_path}")
    else:
        extract_audio_from_video(video_file_path, audio_file_path)
        transcript_text = transcribe_audio(audio_file_path)

        if transcript_text:
            save_transcript_todisk(transcript_text, transcript_output_path)




