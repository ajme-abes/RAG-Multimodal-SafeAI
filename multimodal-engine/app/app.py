# app/app.py
import os
import streamlit as st
from workflow_engine import run_integrated_pipeline, generate_production_blog
from audio_processor import extract_audio_from_video, transcribe_audio
from video_processor import extract_keyframes, analyze_scene_with_gemini
from clip_extractor import extract_video_clip
from reel_generator import generate_vertical_reel

# Configure page layout and style
st.set_page_config(page_title="Multimodal Content Engine", page_icon="🎬", layout="wide")

# Ensure all system tracking folders exist locally
os.makedirs("../data", exist_ok=True)
os.makedirs("../data/extracted_frames", exist_ok=True)
os.makedirs("../data/extracted_clips", exist_ok=True)
os.makedirs("../outputs", exist_ok=True)

st.title("🎬 Multimodal AI Content Engine")
st.caption("Extract insights, generate technical articles, cut clips, and compile vertical mobile reels completely automatically.")

# --- STATE MANAGEMENT MEMORY CACHE ---
# Keeps data safely persistent across UI element interactions
if "transcript" not in st.session_state:
    st.session_state.transcript = None
if "visual_breakdown" not in st.session_state:
    st.session_state.visual_breakdown = None
if "final_blog" not in st.session_state:
    st.session_state.final_blog = None

# --- SIDEBAR COMPONENT PANEL ---
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
        st.session_state.transcript = None
        st.session_state.visual_breakdown = None
        st.session_state.final_blog = None
        st.rerun()

# --- PIPELINE ENGINE EXECUTION ORCHESTRATION ---
if uploaded_video is not None:
    # Save uploaded file chunk straight into our data workspace directory
    video_input_path = os.path.join("../data", uploaded_video.name)
    with open(video_input_path, "wb") as f:
        f.write(uploaded_video.getbuffer())
        
    st.info(f"📁 Source file staged and locked: **{uploaded_video.name}**")
    
    # Run the entire pipeline if no previous state exists in memory
    if st.button("🚀 Process Complete AI Workflow", type="primary"):
        audio_output_path = "../data/extracted_audio.mp3"
        frames_dir = "../data/extracted_frames"
        
        # Phase 1 Tracker: Hearing
        with st.status("🎙️ Phase 1: Separating and Transcribing Audio...", expanded=True) as status:
            st.write("Demuxing target stream via FFmpeg...")
            extract_audio_from_video(video_input_path, audio_output_path)
            st.write("Transcribing audio tracks into textual tokens...")
            st.session_state.transcript = transcribe_audio(audio_output_path)
            status.update(label="Phase 1 Complete: Audio Transcript Secured!", state="complete")
            
        # Phase 2 Tracker: Seeing
        with st.status("🎬 Phase 2: Extracting Timelines & Scene Layouts...", expanded=True) as status:
            st.write("Slicing downsampled image arrays from video stream...")
            extract_keyframes(video_input_path, frames_dir, interval_seconds=sampling_interval)
            st.write("Analyzing chronological image batches via Vision-VLM array...")
            st.session_state.visual_breakdown = analyze_scene_with_gemini(frames_dir)
            status.update(label="Phase 2 Complete: Visual Timeline Extracted!", state="complete")
            
        # Phase 3 Tracker: Orchestration & Media Slicing
        with st.status("🧠 Phase 3: Synthesizing Final Content Asset Layouts...", expanded=True) as status:
            st.write("Fusing text strings and scene images together...")
            st.session_state.final_blog = generate_production_blog(
                st.session_state.transcript, 
                st.session_state.visual_breakdown
            )
            
            st.markdown("---")
            st.write("🎬 **Configure Your Video Slices:**")
            
            # 1. Dynamic Inputs for the Widescreen Highlight Clip
            clip_start = st.text_input("Highlight Clip Start Time (HH:MM:SS)", value="00:00:10")
            clip_duration = st.number_input("Highlight Duration (Seconds)", min_value=1, max_value=60, value=15)
            
            # 2. Dynamic Inputs for the Vertical Reel
            reel_start = st.text_input("Vertical Reel Start Time (HH:MM:SS)", value="00:00:20")
            reel_duration = st.number_input("Reel Duration (Seconds)", min_value=1, max_value=60, value=10)
            
            st.write("Automating custom horizontal highlight clipping...")
            extract_video_clip(video_input_path, clip_start, clip_duration, "../data/extracted_clips/horizontal_highlight.mp4")
            
            st.write("Compiling vertical 9:16 mobile portrait shorts...")
            generate_vertical_reel(video_input_path, reel_start, reel_duration, "../data/extracted_clips/vertical_reel.mp4")
            
            status.update(label="Phase 3 Complete: Workspace Assets Compiled!", state="complete")
            
        st.success("🎉 Multimodal Content Engine Processed All Layers Successfully!")

# --- MULTI-TAB RESULTS WORKSPACE DISPLAY ---
if st.session_state.final_blog:
    st.markdown("### 📊 Engine Output Results Workspace")
    
    tab1, tab2, tab3 = st.tabs(["📄 Generated Blog Post", "🎬 Automated Video Assets", "🎙️ Raw Data Tracks"])
    
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
        st.subheader("🎞️ Generated Video Highlights & Shorts")
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### 📐 16:9 Landscape Video Highlight Clip")
            clip_path = "../data/extracted_clips/horizontal_highlight.mp4"
            if os.path.exists(clip_path):
                # Streamlit lets us display video players directly inside the web browser!
                st.video(clip_path)
                
        with col2:
            st.markdown("#### 📱 9:16 Vertical Smartphone Shorts Reel")
            reel_path = "../data/extracted_clips/vertical_reel.mp4"
            if os.path.exists(reel_path):
                st.video(reel_path)
                
    with tab3:
        st.subheader("System Data Diagnostics Logs")
        col_left, col_right = st.columns(2)
        with col_left:
            st.markdown("**Text Transcription Track:**")
            st.text_area("Whisper Outputs", st.session_state.transcript, height=350)
        with col_right:
            st.markdown("**Vision Scene Analysis Track:**")
            st.text_area("VLM Outputs", st.session_state.visual_breakdown, height=350)
else:
    if uploaded_video is None:
        st.warning("👈 Please drag and drop an MP4 video file in the left panel to kick off the pipeline.")