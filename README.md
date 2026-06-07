# 🤖 AI Engineering Portfolio

A structured, end-to-end AI engineering roadmap covering three production-focused projects — from document intelligence and multimodal processing to AI safety and red-teaming. Each project is built with real architecture, not tutorial code.

---

## 📂 Repository Structure

```
├── rag-system/          # Phase 1 — Chat with PDF (RAG Pipeline)
├── multimodal-engine/   # Phase 2 — AI Reel Generator (Multimodal)
├── ai-saftey-audit/     # Phase 3 — Safety & Bias Evaluation Toolkit
├── requirements.txt     # Shared top-level dependencies
├── .env
├── .gitignore
└── README.md

```

---

## 🗺️ Project Overview

### Phase 1 — RAG System ✅ Complete

**`/rag-system`**

A production-grade Retrieval-Augmented Generation pipeline. Upload any PDF and have a grounded, citation-backed conversation with its contents. Features a multi-LLM fallback chain (Ollama → HuggingFace → OpenAI → Google Gemini), persistent ChromaDB vector storage, sentence-aware chunking, and a full Streamlit chat UI.

🔗 **Live Demo:** [ragsystem-chatpadf.streamlit.app](https://ragsystem-chatpadf.streamlit.app/)

**Status:** Fully functional — live deployed on Streamlit Cloud with UI, CLI runner, and diagnostic test suite.

---

### Phase 2 — Multimodal Engine 🔧 In Progress

**`/multimodal-engine`**
An AI engine designed to process, analyze, and generate content across multiple data types — text, images, and audio. The core objective is building automated pipelines that link transcription, summarization, and video processing into a single workflow.

**Planned Deliverable:** AI Reel Generator — takes long-form video/audio, transcribes it, identifies key moments, and outputs a short-form highlight reel.

**Status:** Architecture planned. Implementation in progress.

---

### Phase 3 — AI Safety Audit 🔧 In Progress

**`/ai-saftey-audit`**

An alignment and auditing framework for stress-testing LLMs against adversarial inputs. Covers prompt injection defense, bias and fairness auditing, toxicity guardrails, and jailbreak red-teaming.

**Planned Deliverable:** AI Safety Audit Report + automated testing framework.

**Status:** Architecture planned. Implementation in progress.

---

## 🏗️ Architecture Overview

```text
┌─────────────────────────────────────────────────────────────────┐
│                     AI ENGINEERING PORTFOLIO                    │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Phase 1: RAG System                                            │
│  ┌──────────┐   ┌──────────┐   ┌──────────┐   ┌────────────┐  │
│  │  loader  │──►│ chunker  │──►│embedding │──►│ ChromaDB   │  │
│  └──────────┘   └──────────┘   └──────────┘   └─────┬──────┘  │
│                                                      │         │
│  ┌──────────┐   ┌──────────┐   ┌──────────────────┐ │         │
│  │Streamlit │◄──│qa_pipeln │◄──│    retriever     │◄┘         │
│  │   UI     │   │(LLM chain│   │ (distance filter)│           │
│  └──────────┘   └──────────┘   └──────────────────┘           │
│                                                                 │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Phase 2: Multimodal Engine (Planned)                           │
│  ┌──────────┐   ┌──────────┐   ┌──────────┐   ┌────────────┐  │
│  │  Video   │──►│ Whisper  │──►│Summarize │──►│Reel Output │  │
│  │  Input   │   │(Transcr.)│   │  (LLM)   │   │ Generator  │  │
│  └──────────┘   └──────────┘   └──────────┘   └────────────┘  │
│                                                                 │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Phase 3: AI Safety Audit (Planned)                             │
│  ┌──────────┐   ┌──────────┐   ┌──────────┐   ┌────────────┐  │
│  │ Prompt   │──►│ Injection│──►│  Bias    │──►│   Audit    │  │
│  │ Red-Team │   │ Defense  │   │ Scanner  │   │   Report   │  │
│  └──────────┘   └──────────┘   └──────────┘   └────────────┘  │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## ✅ Features by Project

### RAG System

- PDF text extraction with regex-based cleaning (pypdf)
- NLTK sentence-aware chunking (no mid-sentence cuts)
- Semantic embeddings via `all-MiniLM-L6-v2` (HuggingFace)
- Persistent ChromaDB vector store
- Distance-filtered retrieval (cosine similarity threshold)
- Multi-LLM fallback: Ollama → HuggingFace → OpenAI → Google Gemini
- Strictly grounded prompt — LLM cannot answer outside retrieved context
- Page-level citations with text snippets and distance scores
- Full Streamlit chat UI with session state management
- Chunk quality diagnostic tooling

### Multimodal Engine _(planned)_

- Audio transcription via OpenAI Whisper
- LLM-powered content summarization
- Automated video timeline slicing
- Cross-modal search (text query → image/video results)
- Unified API for vision and language models

### AI Safety Audit _(planned)_

- Prompt injection detection and filtering
- Automated bias and fairness test suites
- Toxicity evaluation layer
- Jailbreak red-teaming framework
- Structured audit report generation

---

## 🛠️ Tech Stack

| Category       | Technologies                                             |
| -------------- | -------------------------------------------------------- |
| Language       | Python 3.10+                                             |
| UI             | Streamlit                                                |
| LLM — Local    | Ollama (llama3)                                          |
| LLM — Cloud    | OpenAI GPT-4o-mini, Google Gemini 2.5 Flash              |
| Embeddings     | HuggingFace `sentence-transformers` (`all-MiniLM-L6-v2`) |
| Vector DB      | ChromaDB                                                 |
| PDF Parsing    | pypdf                                                    |
| Text Splitting | LangChain, NLTK                                          |
| Audio/Video    | OpenAI Whisper _(Phase 2)_                               |
| Safety/Eval    | Custom framework + `ragas` _(Phase 3)_                   |
| Environment    | python-dotenv                                            |
| Security       | cryptography, pyjwt                                      |

---

## 📦 Installation

**1. Clone the repository**

```bash
git clone https://github.com/ajme-abes/RAG-Multimodal-SafeAI.git
cd your-repo
```

**2. Create and activate a virtual environment**

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS / Linux
source venv/bin/activate
```

**3. Install shared dependencies**

```bash
pip install -r requirements.txt
```

**4. Set up environment variables**

Create a `.env` file in the root directory:

```env
OPENAI_API_KEY=your_openai_key_here
GOOGLE_API_KEY=your_google_key_here
HF_API_KEY=your_huggingface_key_here
```

---

## 🚀 Usage

### Run the RAG System

```bash
cd rag-system/app
streamlit run app.py
```

See [`rag-system/README.md`](./rag-system/README.md) for full setup and usage details.

### Run Diagnostic Tests

```bash
# Chunk quality audit
python rag-system/test/inspect_chunk.py

# Semantic search test
python rag-system/test/test_search.py

# Vector DB connection test
python rag-system/test/testdb_load.py
```

---

## 🖼️ Screenshots

**RAG System** — Live at [ragsystem-chatpadf.streamlit.app](https://ragsystem-chatpadf.streamlit.app/)

| Dashboard | Document Ingestion | Chat + Citations |
|---|---|---|
| ![Dashboard](rag-system/assets/image.png) | ![Ingestion](rag-system/assets/image_pdfup.png) | ![Chat](rag-system/assets/image_chat.png) |

> **Multimodal Engine** and **AI Safety Audit** screenshots will be added as each phase ships.

---

## 📈 Roadmap

### Phase 1 — RAG System

- [x] PDF loader with text cleaning
- [x] NLTK sentence-aware chunker
- [x] HuggingFace embedding model
- [x] ChromaDB persistent vector store
- [x] Distance-filtered retriever
- [x] Multi-LLM fallback chain (Ollama / HF / OpenAI / Gemini)
- [x] Streamlit chat UI with citations
- [x] Conversation memory (chat history in prompt)
- [x] Streaming LLM responses
- [x] Multi-document support
- [x] Confidence gate (block hallucination on off-topic queries)
- [ ] Cross-encoder reranking
- [ ] FastAPI backend
- [ ] Docker deployment

### Phase 2 — Multimodal Engine

- [ ] Whisper audio transcription
- [ ] LLM summarization pipeline
- [ ] Video clip extraction
- [ ] Reel generator output
- [ ] Cross-modal search API

### Phase 3 — AI Safety Audit

- [ ] Prompt injection test suite
- [ ] Bias and fairness scanner
- [ ] Toxicity guardrail layer
- [ ] Jailbreak red-team framework
- [ ] Automated audit report generator

---

## 🔮 Future Improvements

- **RAG Evaluation** — Integrate `ragas` to score faithfulness, answer relevancy, and context recall across all three phases
- **Unified API Gateway** — Single FastAPI service exposing all three systems under one interface
- **Docker Compose** — One-command deployment for the entire portfolio stack
- **CI/CD Pipeline** — GitHub Actions for automated testing and linting on every push
- **Async Processing** — Parallelize embedding and inference for production-scale throughput
- **Web Scraping Ingestion** — Extend the RAG loader to ingest URLs and web pages, not just PDFs

---

## 👤 Author

Built as part of a structured AI engineering internship roadmap — progressing from RAG fundamentals through multimodal systems to AI safety and evaluation.
