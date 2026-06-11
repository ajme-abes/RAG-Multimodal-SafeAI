import subprocess
import os
from dotenv import load_dotenv
from google import genai
from google.genai import types

# Import the shared structural schema to fix your Streamlit ImportError
from models import StructuredTranscript

load_dotenv()

def extract_audio_from_video(video_path, audio_output_path):
    print(f"[1] Demultiplexing audio from video: {video_path}")

    if os.path.exists(audio_output_path):
        os.remove(audio_output_path)

    # Force 16,000Hz, single-channel Mono, uncompressed PCM 16-bit WAV format 
    # This is required for Whisper pipelines and ultra-precise Gemini time tracking
    command = [
        "ffmpeg",
        "-i", video_path,
        "-vn", 
        "-acodec", "pcm_s16le",
        "-ac", "1",
        "-ar", "16000",
        audio_output_path
    ]

    try:
        subprocess.run(command, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        print(f"Audio successfully optimized and saved to: {audio_output_path}")
    except subprocess.CalledProcessError as e:
        print(f"Error during audio extraction: {e.stderr.decode()}")
        raise e

def transcribe_audio(audio_file_path) -> StructuredTranscript:
    print(f"[2] Transcribing audio file: {audio_file_path}")

    client = genai.Client()
    print(f"Uploading Audio file payload to GenAI...")
    upload_audio = client.files.upload(file=audio_file_path)
    print(f"Audio file Uploaded Successfully. Cloud Target: {upload_audio.name}")

    prompt = """
    You are an advanced, high-precision speech-to-text validation system. 
    Transcribe this audio file completely with absolute word-for-word precision.
    For every phrase or sentence, calculate the absolute starting and ending offsets in seconds.
    Provide an accurate chronological breakdown of the words spoken within that exact timeframe.
    Do not add introductory greetings, conversational commentary, or formatting wrappers.
    """
    
    print("🤖 Processing Speech-to-Text structured inference (Waiting for engine response)...")

    try:
        # Enforcing response_schema locks Gemini into returning exact JSON that satisfies our model
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=[upload_audio, prompt],
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=StructuredTranscript,
                temperature=0.0  # Zero out creativity to force strict transcription accuracy
            ),
        )
        # Directly returns the parsed Pydantic object structure
        return response.parsed
        
    except Exception as e:  
        print(f"Critical error during automated transcription: {str(e)}")
        raise e
    finally:
        # Guarantee cloud file deletion to prevent ongoing storage costs or leaks
        print("Cleaning up GenAI cloud storage bucket items...")
        try:
            client.files.delete(name=upload_audio.name)
            print("Audio file completely removed from cloud storage lifetime.")
        except Exception as cleanup_err:
            print(f"Warning: Cloud file cleanup skipped: {cleanup_err}")

def save_transcript_todisk(transcript_data: StructuredTranscript, output_text_path):
    print(f"[3] Saving structured transcript JSON to disk: {output_text_path}")
    try:
        with open(output_text_path, "w", encoding="utf-8") as f:
            f.write(transcript_data.model_dump_json(indent=4))
        print(f"Transcript successfully saved to: {output_text_path}")
    except Exception as e:
        print(f"File I/O error while saving transcript: {str(e)}")

if __name__ == "__main__":
    video_file_path = "../data/sample.mp4"
    audio_file_path = "../data/extracted_audio.wav"  # Switched to production WAV extension
    transcript_output_path = "../data/transcript.json"  # Switched to structured JSON database storage

    if not os.path.exists(video_file_path):
        print(f"Error: Video file not found at path: {video_file_path}")
    else:
        extract_audio_from_video(video_file_path, audio_file_path)
        transcript_obj = transcribe_audio(audio_file_path)

        if transcript_obj:
            save_transcript_todisk(transcript_obj, transcript_output_path)
