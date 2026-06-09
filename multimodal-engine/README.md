# 🎬 SynapseMedia: Multimodal Content Engine

An AI-powered pipeline that takes a raw video file and automatically produces a full content package — audio transcript, visual scene breakdown, a generated technical blog post, a 16:9 highlight clip, and a 9:16 vertical short reel.

---

## 🧠 What It Does

Upload any tutorial or walkthrough video. The engine runs three sequential AI phases:

- **Phase 1 — Audio:** Demuxes the audio stream via FFmpeg and transcribes it using Gemini 2.5 Flash.
- **Phase 2 — Vision:** Extracts keyframes at a configurable interval and runs chronological scene analysis via Gemini Vision (with GPT-4o-mini as fallback).
- **Phase 3 — Synthesis:** Fuses the transcript and visual breakdown into a structured Markdown blog post, then cuts a horizontal highlight clip and a vertical mobile short reel.

---

## 📂 Project Structure

```
multimodal-engine/
├── app/
│   ├── app.py               # Streamlit UI — main entry point
│   ├── workflow_engine.py   # Orchestrates the full pipeline & blog synthesis
│   ├── audio_processor.py   # FFmpeg audio extraction + Gemini transcription
│   ├── video_processor.py   # FFmpeg keyframe extraction + Gemini/OpenAI scene analysis
│   ├── clip_extractor.py    # Cuts a 16:9 horizontal highlight clip
│   └── reel_generator.py    # Crops and compiles a 9:16 vertical short reel
├── data/
│   ├── sample.mp4                        # Test video input
│   ├── extracted_audio.mp3               # Audio demux output
│   ├── extracted_frames/                 # Keyframe JPGs
│   ├── extracted_clips/                  # Output clips
│   └── extracted_real/                   # Output vertical reels
├── output/                               # Generated blog posts (.md)
├── requirements.txt
└── README.md
```

---

## ⚙️ Setup

### 1. Prerequisites

- Python 3.10+
- **FFmpeg** must be installed and available in your system PATH
  - Windows: [ffmpeg.org/download](https://ffmpeg.org/download.html) or `winget install ffmpeg`
  - macOS: `brew install ffmpeg`
  - Linux: `sudo apt install ffmpeg`

### 2. Install dependencies

```bash
cd multimodal-engine
pip install -r requirements.txt
```

### 3. Environment variables

Create a `.env` file in the root of the repository (or in `multimodal-engine/`):

```env
GOOGLE_API_KEY=your_google_gemini_key
OPENAI_API_KEY=your_openai_key
```

Both keys are used — Gemini is the primary model, OpenAI GPT-4o-mini is the visual analysis fallback.

---

## 🚀 Running the App

```bash
cd multimodal-engine/app
streamlit run app.py
```

Then open [http://localhost:8501](http://localhost:8501) in your browser.

### UI Walkthrough

1. Drag and drop an `.mp4` file into the sidebar uploader
2. Set the **keyframe sampling interval** (seconds between captured frames)
3. Click **🚀 Process Complete AI Workflow**
4. View results across three tabs:
   - **📄 Generated Blog Post** — download the Markdown article
   - **🎬 Automated Video Assets** — preview the 16:9 clip and 9:16 reel side by side
   - **🎙️ Raw Data Tracks** — inspect the raw transcript and scene analysis text

---

## 🔧 Running Modules Standalone

Each module has its own `__main__` block for isolated testing:

```bash
# Test audio extraction + transcription only
python app/audio_processor.py

# Test keyframe extraction + scene analysis only
python app/video_processor.py

# Test clip cutting only
python app/clip_extractor.py

# Test vertical reel generation only
python app/reel_generator.py

# Run the full pipeline headlessly (no UI)
python app/workflow_engine.py
```

All standalone runners expect `../data/sample.mp4` as the input file.

---

## 🏗️ Architecture

```
Video Input (.mp4)
      │
      ├──[FFmpeg]──► extracted_audio.mp3
      │                    │
      │              [Gemini 2.5 Flash]
      │                    │
      │              Audio Transcript
      │
      ├──[FFmpeg]──► keyframe_XXXX.jpg (every N seconds)
      │                    │
      │         [Gemini Vision / GPT-4o-mini fallback]
      │                    │
      │              Visual Scene Breakdown
      │
      └──[Gemini 2.5 Flash]──► Technical Blog Post (.md)
      │
      ├──[FFmpeg]──► 16:9 Highlight Clip (.mp4)
      └──[FFmpeg]──► 9:16 Vertical Short Reel (.mp4)
```

---

## 🛠️ Tech Stack

| Component         | Technology                          |
|-------------------|-------------------------------------|
| UI                | Streamlit                           |
| Audio extraction  | FFmpeg (`libmp3lame`)               |
| Transcription     | Google Gemini 2.5 Flash             |
| Keyframe slicing  | FFmpeg (`fps` filter)               |
| Scene analysis    | Google Gemini 2.5 Flash Vision      |
| Vision fallback   | OpenAI GPT-4o-mini                  |
| Blog synthesis    | Google Gemini 2.5 Flash             |
| Clip/reel cutting | FFmpeg (`libx264`, crop filter)     |
| Environment       | python-dotenv                       |

---

## 📈 Roadmap

- [x] FFmpeg audio demux
- [x] Gemini audio transcription
- [x] FFmpeg keyframe extraction
- [x] Gemini visual scene analysis
- [x] OpenAI vision fallback
- [x] Blog post synthesis (transcript + visuals)
- [x] 16:9 highlight clip cutter
- [x] 9:16 vertical reel generator
- [x] Streamlit UI with 3-phase progress tracking
- [ ] Smart clip detection (auto-find highlight moments from transcript)
- [ ] Subtitle/caption burn-in on reels
- [ ] Cross-modal search (natural language → video timestamp)
- [ ] Batch processing (queue multiple videos)
- [ ] FastAPI backend

---

## 🖼️ Screenshots

> Screenshots will be added once the UI is deployed.

---

## 👤 Author

Built as Phase 2 of a structured AI engineering roadmap.  
See the [root README](../README.md) for the full portfolio overview.
