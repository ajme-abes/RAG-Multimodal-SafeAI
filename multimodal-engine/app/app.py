import os
import sys
import glob
import streamlit as st

# Setup system environment alignment routes
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from workflow_engine import generate_production_blog
from audio_processor import extract_audio_from_video, transcribe_audio
from video_processor import extract_keyframes, analyze_scene_with_gemini
from agent_optimizer import run_autonomous_editing_pipeline

from models import StructuredTranscript, ChronologicalVisualTimeline

# Configure clean, production-grade page layout parameters
st.set_page_config(page_title="Multimodal Content Engine", page_icon="🎬", layout="wide")

# Sync runtime directory configurations with core backends
DATA_DIR = "../data"
FRAMES_DIR = "../data/extracted_frames"
OUTPUT_CLIPS_DIR = "../output/clips"
BLOG_OUTPUT_PATH = "../output/how_multimodals_work_blog.md"

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(FRAMES_DIR, exist_ok=True)
os.makedirs(OUTPUT_CLIPS_DIR, exist_ok=True)

st.title("🎬 Multimodal AI Content Engine")
st.caption("Convert long horizontal streams into clear technical blog posts and vertical mobile shorts using a type-safe pipeline.")

# --- PERSISTENT STATE MEMORY LIFECYCLE ---
if "transcript_obj" not in st.session_state:
    st.session_state.transcript_obj = None
if "visual_breakdown_obj" not in st.session_state:
    st.session_state.visual_breakdown_obj = None
if "final_blog" not in st.session_state:
    st.session_state.final_blog = None
if "pipeline_executed" not in st.session_state:
    st.session_state.pipeline_executed = False

# --- SIDEBAR INTERFACE CONTROL PANEL ---
with st.sidebar:
    st.header("⚡ Processing Controls")
    uploaded_video = st.file_uploader("Upload Tutorial Video (MP4)", type=["mp4"])
    
    sampling_interval = st.slider(
        "Downsampling Interval (Seconds)", 
        min_value=1, max_value=30, value=5,
        help="Capture 1 image frame every X seconds of timeline playback."
    )
    
    st.markdown("---")
    if st.button("🗑️ Clear Engine Cache"):
        st.session_state.transcript_obj = None
        st.session_state.visual_breakdown_obj = None
        st.session_state.final_blog = None
        st.session_state.pipeline_executed = False
        st.success("State memory scrubbed cleanly.")
        st.rerun()

# --- INTEGRATED ORCHESTRATION PIPELINE ---
if uploaded_video is not None:
    video_input_path = os.path.join(DATA_DIR, uploaded_video.name)
    
    # Cache video inputs directly onto internal workspace directories
    if not os.path.exists(video_input_path):
        with open(video_input_path, "wb") as f:
            f.write(uploaded_video.getbuffer())
        st.toast(f"Staged asset source: {uploaded_video.name}", icon="📁")

    st.info(f"📁 Source file staged and locked: **{uploaded_video.name}**")
    
    # Execute full workflow sequence if user initializes primary action call
    if st.button("🚀 Process Complete AI Workflow", type="primary"):
        # Switched to production .wav parameters to match high-precision processing models
        audio_output_path = os.path.join(DATA_DIR, "extracted_audio.wav")
        
        # Phase 1 Step: Hearing
        with st.status("🎙️ Phase 1: Separating and Transcribing Audio...", expanded=True) as status:
            st.write("Demuxing 16kHz uncompressed mono track via FFmpeg...")
            extract_audio_from_video(video_input_path, audio_output_path)
            st.write("Transcribing audio tracks into structured Pydantic time models...")
            st.session_state.transcript_obj = transcribe_audio(audio_output_path)
            status.update(label="Phase 1 Complete: Audio Transcript Secured!", state="complete")
            
        # Phase 2 Step: Seeing
        with st.status("🎬 Phase 2: Extracting Timelines & Scene Layouts...", expanded=True) as status:
            st.write("Slicing uniform visual image arrays from video timeline...")
            extract_keyframes(video_input_path, FRAMES_DIR, interval_seconds=sampling_interval)
            st.write("Analyzing chronological frame sequences via Vision-VLM array...")
            st.session_state.visual_breakdown_obj = analyze_scene_with_gemini(FRAMES_DIR, interval_seconds=sampling_interval)
            status.update(label="Phase 2 Complete: Visual Timeline Extracted!", state="complete")
            
        # Phase 3 Step: Fusing and Clipping
        with st.status("🧠 Phase 3: Synthesizing Content & Cutting Highlights...", expanded=True) as status:
            st.write("Fusing text arrays and structural visual timelines together into markdown entries...")
            st.session_state.final_blog = generate_production_blog(
                st.session_state.transcript_obj, 
                st.session_state.visual_breakdown_obj
            )
            
            # Save the final blog post straight to the output directory
            with open(BLOG_OUTPUT_PATH, "w", encoding="utf-8") as b_file:
                b_file.write(st.session_state.final_blog)

            st.write("Running Two-Stage Filtering Highlight Detection Engine...")
            # Automatically scans, validates, and runs 9:16 vertical center crops completely hands-free
            run_autonomous_editing_pipeline(
                video_path=video_input_path,
                audio_transcript=st.session_state.transcript_obj,
                visual_breakdown=st.session_state.visual_breakdown_obj
            )
            
            status.update(label="Phase 3 Complete: Content Generated & Reels Rendered!", state="complete")
            
        st.session_state.pipeline_executed = True
        st.success("🎉 Multimodal Content Engine Processed All Layers Successfully with Zero Time Hallucinations!")

# --- AUTOMATED RESULTS CONTAINER PANEL DISPLAY ---
if st.session_state.pipeline_executed:
    st.markdown("### 📊 Engine Output Results Workspace")
    
    tab1, tab2, tab3 = st.tabs(["📄 Generated Blog Post", "📱 Automated Mobile Reels", "🎙️ System Data Logs"])
    
    with tab1:
        st.subheader("📝 Compiled Technical Blog Article")
        st.markdown(st.session_state.final_blog)
        st.download_button(
            label="💾 Download Article Markdown File",
            data=st.session_state.final_blog,
            file_name="generated_tutorial_blog.md",
            mime="text/markdown"
        )
        
    with tab2:
        st.subheader("🎞️ Extracted 9:16 Vertical Video Reels")
        # Automatically pull whatever verified files were generated by the autonomous cutting engine loop
        generated_reels = sorted(glob.glob(os.path.join(OUTPUT_CLIPS_DIR, "*.mp4")))
        
        if generated_reels:
            # Layout elements dynamically side-by-side using Streamlit grid models
            cols = st.columns(len(generated_reels))
            for i, reel_path in enumerate(generated_reels):
                with cols[i]:
                    clip_name = os.path.basename(reel_path).replace(".mp4", "").replace("auto_reel_", "Reel #")
                    st.markdown(f"**{clip_name.title()}**")
                    st.video(reel_path)
        else:
            st.info("No clips passed the strict Stage 2 Visual Verification energy standards layer.")
                
    with tab3:
        st.subheader("System Data Diagnostics Logs")
        col_left, col_right = st.columns(2)
        with col_left:
            st.markdown("**Text Transcription JSON Output Track:**")
            # Display serialized models directly into interactive text area components
            st.text_area(
                "Whisper Transcript Objects Data", 
                st.session_state.transcript_obj.model_dump_json(indent=2) if st.session_state.transcript_obj else "", 
                height=400
            )
        with col_right:
            st.markdown("**Vision Scene Analysis JSON Output Track:**")
            st.text_area(
                "VLM Chrono Objects Data", 
                st.session_state.visual_breakdown_obj.model_dump_json(indent=2) if st.session_state.visual_breakdown_obj else "", 
                height=400
            )
else:
    if uploaded_video is None:
        st.warning("👈 Please drag and drop an MP4 video file in the left panel to kick off the pipeline.")
