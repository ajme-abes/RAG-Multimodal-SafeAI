---
title: multi-content-engine
emoji: 🎬
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 7860
app_dir: multimodal-engine
---

<div align="center">

# 🤖 AI Engineering Portfolio

**Three production-grade AI systems — built end-to-end, deployed, and tested.**

[![Phase 1 — Live](https://img.shields.io/badge/Phase_1-Live_on_Streamlit-FF4B4B?logo=streamlit&logoColor=white)](https://ragsystem-chatpadf.streamlit.app/)
[![Phase 2 — Live](https://img.shields.io/badge/Phase_2-Live_on_HuggingFace-FFD21E?logo=huggingface&logoColor=black)](https://huggingface.co/spaces)
[![Python](https://img.shields.io/badge/Python-3.12-3776AB?logo=python&logoColor=white)](https://python.org)
[![Tests](https://img.shields.io/badge/Tests-72_passing-brightgreen?logo=pytest)](multimodal-engine/test/)

</div>

---

## 📂 Projects

| # | Project | What It Does | Stack | Status |
|---|---|---|---|---|
| 1 | [**RAG System**](./rag-system/) | Chat with any PDF — grounded answers with page citations | Gemini · ChromaDB · LangChain · Streamlit | ✅ [Live](https://ragsystem-chatpadf.streamlit.app/) |
| 2 | [**Multimodal Engine**](./multimodal-engine/) | Convert long videos into vertical reels + blog posts | Gemini 2.5 · FFmpeg · Pydantic · Streamlit | ✅ [Live](https://huggingface.co/spaces) |
| 3 | [**AI Safety Audit**](./ai-saftey-audit/) | LLM red-teaming, bias scanning, prompt injection defense | Custom framework · ragas | 🔧 In progress |

---

## 🗂️ Structure

```
├── rag-system/           # Phase 1 — RAG Pipeline
├── multimodal-engine/    # Phase 2 — Multimodal Content Engine
├── ai-saftey-audit/      # Phase 3 — AI Safety Toolkit
├── .env.example
├── .gitignore
└── README.md
```

> Each project has its own detailed README with architecture, setup, and usage guides.

---

## Phase 1 — RAG System

**[→ Full README](./rag-system/README.md) · [→ Live Demo](https://ragsystem-chatpadf.streamlit.app/)**

Upload any PDF and have a grounded, citation-backed conversation with it. Multi-LLM fallback chain (Ollama → HuggingFace → OpenAI → Gemini), persistent ChromaDB vector store, sentence-aware chunking, and streaming responses.

| | |  | |
|---|---|---|---|
| ![Dashboard](rag-system/assets/image.png) | ![Upload](rag-system/assets/image_pdfup.png) | ![Chat](rag-system/assets/image_chat.png) | ![Score](rag-system/assets/image_scor.png) |

---

## Phase 2 — Multimodal Content Engine

**[→ Full README](./multimodal-engine/README.md) · [→ Live Demo](https://huggingface.co/spaces)**

Upload an MP4 → get a 9:16 vertical reel and a CMS-ready blog post. Parallel audio/visual processing tracks, two-stage AI clip verification, and dual FFmpeg render modes.

| | | |
|---|---|---|
| ![Dashboard](multimodal-engine/assets/dashboard.png) | ![Blog](multimodal-engine/assets/mdpage.png) | ![Reels](multimodal-engine/assets/reelpage.png) |

---

## Phase 3 — AI Safety Audit

**[→ Full README](./ai-saftey-audit/README.md)**

An alignment and auditing framework for stress-testing LLMs — prompt injection defense, bias and fairness auditing, toxicity guardrails, and jailbreak red-teaming.

**Status:** Architecture planned. Implementation in progress.

---

## 🛠️ Tech Across All Phases

| Category | Tools |
|---|---|
| **Language** | Python 3.12 |
| **UI** | Streamlit |
| **LLMs** | Google Gemini 2.5 Flash/Pro · OpenAI GPT-4o-mini · Ollama llama3 |
| **Embeddings** | HuggingFace `all-MiniLM-L6-v2` |
| **Vector DB** | ChromaDB |
| **Audio/Video** | FFmpeg · Google Gemini 2.5 Flash/Pro |
| **Validation** | Pydantic v2 |
| **Testing** | pytest — 72 tests, 0 failures |
| **Deployment** | Streamlit Cloud · Hugging Face Spaces · Docker |

---

## 👤 Author

Built as a structured AI engineering roadmap — progressing from RAG fundamentals through multimodal systems to AI safety and evaluation.
