<div align="center">

<img src="assets/dashboard.png" alt="SynapseMedia Dashboard" width="100%"/>

# 🎬 SynapseMedia — Multimodal Content Engine

**Turn any long-form video into a vertical reel and a CMS-ready blog post — fully automated.**

[![Python](https://img.shields.io/badge/Python-3.12-blue?logo=python&logoColor=white)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.57-FF4B4B?logo=streamlit&logoColor=white)](https://streamlit.io/)
[![Gemini](https://img.shields.io/badge/Gemini_2.5-Flash_%26_Pro-4285F4?logo=google&logoColor=white)](https://ai.google.dev/)
[![Tests](https://img.shields.io/badge/Tests-72_passing-brightgreen?logo=pytest)](multimodal-engine/test/)
[![Docker](https://img.shields.io/badge/Docker-Ready-2496ED?logo=docker&logoColor=white)](Dockerfile)
[![License](https://img.shields.io/badge/License-MIT-yellow)](LICENSE)

[**Live Demo**](#-demo) · [**Quick Start**](#-quick-start) · [**Architecture**](#-architecture) · [**Docker**](#-docker-deployment)

</div>

---

## ✨ What It Does

Upload an MP4 — a tutorial, podcast, or product walkthrough. SynapseMedia runs it through a 4-phase AI pipeline and delivers two production-ready assets:

| Output | Description |
|---|---|
| 📝 **Blog Post** | Full Markdown article with YAML front matter, headers, and tables — ready for WordPress, Dev.to, or Hugo |
| 📱 **Vertical Reels** | 1080×1920 MP4 clips at 9:16 — verified for energy score, auto-cropped, ready for TikTok, Reels, or Shorts |

Everything runs with **zero manual editing**. No timeline scrubbing, no copy-pasting transcripts, no cropping by hand.

---

## 🎥 Demo

<div align="center">

https://github.com/ajme-abes/RAG-Multimodal-SafeAI/tree/main/multimodal-engine/assets/video.mp4

</div>

> Upload a video → click one button → get a blog post and vertical reels in minutes.

---

## 📸 Screenshots

<table>
  <tr>
    <td align="center" width="33%">
      <img src="assets/dashboard.png" alt="Main Dashboard" width="100%"/>
      <br/><b>Main Dashboard</b>
      <br/><sub>Upload panel, interval slider, layout mode selector</sub>
    </td>
    <td align="center" width="33%">
      <img src="assets/mdpage.png" alt="Blog Post Output" width="100%"/>
      <br/><b>Generated Blog Post</b>
      <br/><sub>YAML front matter + full Markdown rendered in-app</sub>
    </td>
    <td align="center" width="33%">
      <img src="assets/reelpage.png" alt="Reels Output" width="100%"/>
      <br/><b>Vertical Reels</b>
      <br/><sub>9:16 clips displayed side-by-side with inline playback</sub>
    </td>
  </tr>
</table>

---

## 🏗️ Architecture

The pipeline runs two tracks in parallel — one for audio, one for video — then fuses them.

```
┌─────────────────────────────────────┐
│           Uploaded MP4 Video        │
└──────────────┬──────────────────────┘
               │
     ┌─────────┴──────────┐
     ▼                    ▼
 TRACK A               TRACK B
 audio_processor       video_processor
 ─────────────         ───────────────
 FFmpeg demux          FFmpeg keyframe
 16kHz PCM WAV         extraction (JPEGs)
     │                      │
     ▼                      ▼
 Gemini 2.5 Flash      Gemini 2.5 Flash
 StructuredTranscript  ChronologicalVisual
 (float timestamps)    Timeline (float ts)
     │                      │
     └──────────┬───────────┘
                │
                ▼
        workflow_engine.py
        ───────────────────
        Injects both tracks
        into Gemini prompt
        → Markdown Blog Post
        (YAML front matter)
                │
                ▼
        agent_optimizer.py
        ──────────────────
        Stage 1: Text filter
        (Gemini 2.5 Flash)
        Top clip candidates
                │
                ▼
        reel_generator.py
        ─────────────────
        Stage 2: Video verify
        (Gemini 2.5 Pro)
        Energy score > 60
        Timestamp fine-tune
                │
                ▼
        video_processor.py
        ──────────────────
        FFmpeg 9:16 render
        Blurred Stack or
        AI Smart Face Crop
                │
       ┌────────┴────────┐
       ▼                 ▼
  📝 Blog Post      📱 Vertical Reels
  (.md + YAML)      (1080×1920 MP4)
```

### The 4 Phases

| Phase | Module | What Happens |
|---|---|---|
| **1 — Hear** | `audio_processor.py` | FFmpeg extracts 16kHz mono WAV → Gemini 2.5 Flash transcribes into `StructuredTranscript` with per-word float timestamps |
| **2 — See** | `video_processor.py` | FFmpeg samples keyframes at configurable intervals → Gemini 2.5 Flash maps each frame to a float timestamp in `ChronologicalVisualTimeline` |
| **3 — Write** | `workflow_engine.py` | Both data streams are serialised and injected into a Gemini prompt → structured Markdown blog post with YAML front matter |
| **4 — Cut** | `agent_optimizer.py` → `reel_generator.py` | Stage 1 text filter finds hook candidates → Stage 2 uploads raw clips to Gemini 2.5 Pro for energy scoring → FFmpeg renders verified clips |

---

## ⚡ Key Features

**🎯 Zero-Hallucination Timestamps**
Pydantic models enforce `float` types at every pipeline boundary. No `HH:MM:SS` strings, no off-by-one second errors, no silent type mismatches.

**🔄 Dual Render Modes**
- **Blurred Stack** — Scales the full 16:9 frame to fit 9:16, fills margins with a blurred duplicate. Keeps code and UI text fully readable.
- **AI Smart Face Crop** — Gemini detects speaker position (`left` / `center` / `right`) and centers the crop frame around them.

**🛡️ Two-Stage Clip Verification**
Every candidate clip is scored by Gemini 2.5 Pro before rendering. Only clips with `audio_energy_score > 60` and `final_relevance_decision = true` make it to FFmpeg.

**♻️ Automatic Fallback Chain**
If Gemini scene analysis fails → OpenAI GPT-4o-mini takes over → if that fails too → a placeholder timeline keeps the pipeline alive.

**⏳ Exponential Backoff with Jitter**
Every Gemini call is wrapped in `retry_with_backoff` — handles 429 / 503 errors automatically with randomised delay to prevent thundering herd.

**☁️ Cloud Asset Cleanup**
All files uploaded to the Gemini Files API are deleted inside `finally` blocks — no leaked cloud storage, even on exception.

**📝 CMS-Ready Output**
Blog posts include complete YAML front matter: `title`, `slug`, `date`, `tags`, `category`, `description`. Date-prefixed filenames prevent overwrites.

---

## 🛠️ Tech Stack

| Layer | Tool | Version |
|---|---|---|
| **UI** | Streamlit | 1.57.0 |
| **Audio extraction** | FFmpeg (`pcm_s16le`) | system |
| **Transcription** | Google Gemini 2.5 Flash | via `google-genai` 1.68.0 |
| **Scene analysis** | Google Gemini 2.5 Flash | via `google-genai` 1.68.0 |
| **Vision fallback** | OpenAI GPT-4o-mini | via `openai` 2.37.0 |
| **Clip verification** | Google Gemini 2.5 Pro | via `google-genai` 1.68.0 |
| **Blog synthesis** | Google Gemini 2.5 Flash | via `google-genai` 1.68.0 |
| **Video rendering** | FFmpeg (`libx264` + `aac`) | system |
| **Schema validation** | Pydantic v2 | 2.12.5 |
| **Testing** | pytest | 72 tests, 0 failures |

---

## 🚀 Quick Start

### 1. Prerequisites

Install **FFmpeg** and make sure it's on your system PATH:

```bash
# macOS
brew install ffmpeg

# Ubuntu / Debian
sudo apt install ffmpeg

# Windows
winget install ffmpeg

# Verify
ffmpeg -version
```

### 2. Clone & Install

```bash
git clone https://github.com/ajme-abes/RAG-Multimodal-SafeAI
cd synapsemedia/multimodal-engine

# Create a virtual environment
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate

# Install pinned dependencies
pip install -r requirements.txt
```

### 3. Configure API Keys

Copy the example env file and fill in your keys:

```bash
cp .env.example .env
```

Edit `.env`:

```env
GEMINI_API_KEY="your_google_gemini_api_key"
OPENAI_API_KEY="your_openai_api_key"
```

> Get your Gemini key at [aistudio.google.com](https://aistudio.google.com/app/apikey)
> The OpenAI key is optional — used only as a fallback if Gemini scene analysis fails.

### 4. Run

```bash
streamlit run app/app.py
```

Open **http://localhost:8501** in your browser.

---

## 🎛️ Usage Guide

<table>
  <tr>
    <th>Step</th>
    <th>Action</th>
    <th>Tip</th>
  </tr>
  <tr>
    <td>1</td>
    <td>Upload an MP4 in the left sidebar</td>
    <td>Works best with videos 3–60 minutes long</td>
  </tr>
  <tr>
    <td>2</td>
    <td>Set the <b>Downsampling Interval</b> slider</td>
    <td>Use <b>5s</b> for code tutorials, <b>10–15s</b> for interviews</td>
  </tr>
  <tr>
    <td>3</td>
    <td>Choose a <b>Reel Layout</b></td>
    <td><b>Blurred Stack</b> for screen recordings, <b>AI Smart Crop</b> for talking-head</td>
  </tr>
  <tr>
    <td>4</td>
    <td>Click <b>🚀 Process Complete AI Workflow</b></td>
    <td>A 10-min video takes roughly 2–4 minutes to process</td>
  </tr>
  <tr>
    <td>5</td>
    <td>Download from the output tabs</td>
    <td><b>Blog Post</b> tab → Markdown file · <b>Reels</b> tab → MP4 files</td>
  </tr>
</table>

### Output Tabs

| Tab | Content |
|---|---|
| 📄 **Blog Post** | Rendered Markdown with download button |
| 📱 **Mobile Reels** | Inline video player for each 9:16 clip |
| 🎙️ **System Logs** | Raw JSON for the transcript and visual timeline |

---

## 🐳 Docker Deployment

### Build & Run Locally

```bash
docker build -t synapsemedia .
docker run -p 8501:8501 \
  -e GEMINI_API_KEY="your_key" \
  -e OPENAI_API_KEY="your_key" \
  synapsemedia
```

Open **http://localhost:8501**.

### Docker Compose (Recommended)

```yaml
version: "3.9"
services:
  synapsemedia:
    build: .
    ports:
      - "8501:8501"
    environment:
      - GEMINI_API_KEY=${GEMINI_API_KEY}
      - OPENAI_API_KEY=${OPENAI_API_KEY}
    volumes:
      - ./data:/workspace/data
      - ./output:/workspace/output
```

```bash
docker compose up --build
```

---

## 🧪 Tests

The test suite runs without any API keys or video files.

```bash
# From the multimodal-engine/ directory
pytest
```

**72 tests · 0 failures · ~30s runtime**

| File | What It Covers |
|---|---|
| `test_models.py` | Pydantic schema construction, type coercion, JSON roundtrip |
| `test_utils.py` | Retry on 429/503, immediate raise on 400/401, backoff delay doubling |
| `test_video_processor.py` | All 4 FFmpeg render modes, directory cleanup, base64 encoding, input validation |
| `test_agent_optimizer.py` | Slug sanitisation, layout routing, per-clip render calls, error isolation |
| `test_pipeline_integration.py` | All 4 phases mocked end-to-end, fallback chains, full pipeline smoke test |

---

## 📁 Project Structure

```
multimodal-engine/
├── app/
│   ├── app.py                    # Streamlit dashboard — pipeline orchestration & UI
│   ├── models.py                 # Shared Pydantic schemas (single source of truth)
│   ├── utils.py                  # Exponential backoff with jitter
│   ├── audio_processor.py        # Phase 1 — FFmpeg WAV demux + Gemini transcription
│   ├── video_processor.py        # Phase 2 — Keyframe extraction + VLM scene analysis + FFmpeg render
│   ├── clip_extractor.py         # Phase 4a — Stage 1 text-based semantic filter
│   ├── reel_generator.py         # Phase 4b — Stage 2 multi-modal video verification
│   └── workflow_engine.py        # Phase 3 — Blog synthesis + pipeline orchestration
├── assets/
│   ├── dashboard.png             # UI screenshot — main dashboard
│   ├── mdpage.png                # UI screenshot — blog output tab
│   ├── reelpage.png              # UI screenshot — reels output tab
│   └── video.mp4                 # Demo walkthrough video
├── data/                         # Runtime: uploaded videos, audio, keyframes (gitignored)
├── output/                       # Runtime: blog posts and rendered reels (gitignored)
├── test/
│   ├── __init__.py
│   ├── test_models.py
│   ├── test_utils.py
│   ├── test_video_processor.py
│   ├── test_agent_optimizer.py
│   └── test_pipeline_integration.py
├── .env.example                  # API key template
├── Dockerfile                    # Production container
├── pytest.ini                    # Test runner config
└── requirements.txt              # Pinned Python dependencies
```

---

## 📈 Roadmap

- [ ] **Word-level animated subtitles** — Burn styled captions into frames using ASS/SSA overlay filters
- [ ] **Frame-by-frame face tracking** — Replace static `left/center/right` with a landmark model for smooth pan-and-scan
- [ ] **Social publishing webhooks** — Direct push to YouTube Shorts, Instagram Reels, and TikTok APIs
- [ ] **Natural language video search** — Query timestamps using the `StructuredTranscript` as a semantic index
- [ ] **Configurable resolutions** — 1080×1920 (TikTok/Reels), 1080×1080 (square), custom aspect ratios
- [ ] **Progress persistence** — Save pipeline state to disk so a page refresh doesn't lose results

---

## ⚠️ Known Limitations

- **Gemini file processing delay** — Uploaded video files enter a `PROCESSING` state before they can be queried. Large clips can take 30–60 seconds to process server-side.
- **No persistent storage** — Streamlit session state resets on page refresh. Results are saved to `output/` on disk but not re-loaded automatically.
- **Rate limits** — Gemini 2.5 Pro (used in Stage 2) has lower rate limits than Flash. If you process many clips back-to-back, the backoff utility will kick in.

---

## 🤝 Contributing

1. Fork the repo
2. Create a feature branch: `git checkout -b feature/my-feature`
3. Make your changes and add tests
4. Run `pytest` — all tests must pass
5. Open a pull request

---

<div align="center">

Built with ❤️ using **Gemini 2.5**, **FFmpeg**, and **Streamlit**

⭐ Star this repo if it saved you hours of manual editing

</div>
