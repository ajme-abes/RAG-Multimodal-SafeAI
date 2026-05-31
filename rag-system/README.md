# ⚙️ Enterprise RAG System — Chat with PDF

A production-grade **Retrieval-Augmented Generation (RAG)** pipeline that allows users to upload multiple PDFs and hold grounded, conversational, citation-backed interactions with their contents. Built with a highly modular architecture, a real-time stream validation engine, a multi-LLM resilient fallback chain, and an interactive Streamlit UI dashboard.

![Main App Dashboard](../assets/image.png)

---

## 📌 Project Overview

Most LLMs hallucinate or return generic answers when questioned about private or domain-specific documentation. This system solves that problem through a structured sequence: it compresses conversational context window queries into search terms, retrieves the most semantically relevant chunks from a persistent vector index, filters them mathematically based on geometric distance thresholds, and grounds the active model's response strictly within that context.

Upload multiple PDFs ➔ ask questions ➔ get real-time streaming answers complete with page-level citations and mathematical validation metrics.

---

## 🏗️ Architecture & Operational Workflow

```text
┌─────────────────────────────────────────────────────────────┐
│                       USER (Streamlit UI)                   │
└──────────────────────────┬──────────────────────────────────┘
                           │  Upload Multiple PDFs + Chat
                           ▼
┌─────────────────────────────────────────────────────────────┐
│  INGESTION PIPELINE                                         │
│                                                             │
│  loader.py          chunker.py          embeding.py         │
│  ┌──────────┐      ┌──────────┐        ┌──────────────┐     │
│  │ Extract  │ ───► │  NLTK    │ ──────►│ all-MiniLM   │     │
│  │ & Clean  │      │ Sentence │        │ -L6-v2       │     │
│  │ PDF Text │      │ Splitter │        │ (HuggingFace)│     │
│  └──────────┘      └──────────┘        └──────┬───────┘     │
│                                               │             │
│                                               ▼             │
│                                    ┌──────────────────┐     │
│                                    │ vectore_store.py │     │
│                                    │ ChromaDB Index   │     │
│                                    │ (Continuous/Disk)│     │
│                                    └──────────────────┘     │
└─────────────────────────────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────┐
│  CONVERSATIONAL QUERY CONDENSER & TRANSFORMER LAYER         │
│                                                             │
│  qa_pipeline.py ➔ condense_user_query()                      │
│  Transforms user prompt + chat history window into optimal  │
│  database search keyword vectors (Filters conversational noise)│
└──────────────────────────┬──────────────────────────────────┘
                           │ Optimized Keywords Target
                           ▼
┌─────────────────────────────────────────────────────────────┐
│  RETRIEVAL & GENERATION PIPELINE                            │
│                                                             │
│  retriever.py                      qa_pipeline.py           │
│  ┌─────────────────────┐          ┌───────────────────────┐ │
│  │ Similarity Search   │          │ Grounded Prompt Core  │ │
│  │ + Score Metric Calc │ ───────► │ + Token-Probed        │ │
│  │ (threshold: 0.85)   │          │   Fallback Stream Loop│ │
│  └─────────────────────┘          │                       │ │
│                                   │ 1. Ollama (local)     │ │
│                                   │ 2. HuggingFace Hub    │ │
│                                   │ 3. OpenAI GPT-4o-mini │ │
│                                   │ 4. Google Gemini      │ │
│                                   └───────────────────────┘ │
└─────────────────────────────────────────────────────────────┘
                           │
                           ▼
     Live Word Token Stream + Source Citations + Confidence Display

---

## ✅ Features

- **Multi-Document Appending Matrix** — Upload and read several documents consecutively without wiping older context out of the vector database index
- **Persistent State Tracker** — Detects duplicate data files via the Streamlit UI to avoid redundant vector embedding transformations
- **Conversational Memory Query Rewriting** — Compiles recent context windows to extract semantic search keywords, filtering out conversational filler like "what is" or "tell me about"
- **Sentence-Aware Token Chunking** — Employs NLTK tokenizers to process documents along natural sentence structures (chunk size: 800 tokens, overlap: 150 tokens)
- **Distance-Filtered Retrieval Guard** — Rejects irrelevant background context chunks that fall outside the configured mathematical distance threshold
- **Grounded System Prompt Architecture** — Instructs models to restrict responses to the source material and return an explicit fallback text if the query is unanswerable from the context
- **Token-Probed Multi-LLM Cascading Fallback Chain** — Routes traffic down a secure pipeline (Ollama ➔ Hugging Face ➔ OpenAI ➔ Google Gemini). The pipeline probes the stream's first token to capture errors (like 429 rate limits or quota issues) instantly, falling back automatically without crashing the user interface
- **Real-Time Stream Rendering** — Displays text generation token-by-token directly inside the chat workspace
- **Source Citation Analytics** — Expands detailed source dropdown cards detailing the document origin name, target page location, raw vector distance score, and precise text snippet matching
- **Dynamic Retrieval Confidence Scoring** — Translates database Euclidean distance calculations into a user-friendly percentage rating on the dashboard interface


---

## 🖼️ User Interface Captures

### 1. Ingestion Control Centre
Manages document loading buffers and tracking indexes inside the active sidebar database monitor.

### 2. Live Conversational Streaming Engine
Displays text generation token-by-token using the optimized multi-vendor routing stack.

### 3. Source Citations & Retrieval Confidence Scoring Metrics
Translates raw distance metrics into analytical confidence scores and includes page-level citations.

---

## 🛠️ Tech Stack

| Layer                     | Technology                                                                                     |
|----------------------------|-----------------------------------------------------------------------------------------------|
| **Frontend Framework**     | Streamlit                                                                                     |
| **PDF Extraction Engine**  | pypdf                                                                                         |
| **Segmentation Layer**     | LangChain Text Splitters + NLTK Tokenizers                                                    |
| **Embedding Matrix**       | HuggingFace `all-MiniLM-L6-v2` via sentence-transformers                                      |
| **Vector Index Database**  | ChromaDB (Persistent Local Storage Architecture)                                              |
| **Local LLM Client**       | Ollama Engine Instance (llama3 / llama3.2)                                                    |
| **Cloud LLM Providers**    | OpenAI API (`gpt-4o-mini`), Google GenAI SDK (`gemini-2.5-flash`), Hugging Face Inference API |
| **Environment Control**    | python-dotenv                                                                                 |
| **Language Profile**       | Python 3.10+ / 3.12                                                                           |

---

## 📦 Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/ajme-abes/RAG-Multimodal-SafeAI.git
   cd RAG-Multimodal-SafeAI/rag-system



**2. Create and activate a virtual environment**
```bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS / Linux
source venv/bin/activate
```

**3. Install dependencies**
```bash
pip install -r requirements.txt
```

**4. Set up environment variables**

Create a `.env` file in the `rag-system/` directory:
```env
OPENAI_API_KEY=your_openai_key_here
GOOGLE_API_KEY=your_google_key_here
HF_API_KEY=your_huggingface_key_here
```

> Note: At least one cloud vendor API credential key or a running local instance of Ollama is required for inference processing.

**5. Set up local inference models (Optional)**
Download from [ollama.com](https://ollama.com) then pull the model:
```bash
ollama pull llama3
```

---

## 🚀 Usage Guide

**Launching the Dashboard Web UI Application**
```bash
cd rag-system/app
streamlit run app.py
```

Then:
1. Open `http://localhost:8501` in your browser
2. Upload a PDF using the sidebar
3. Wait for the ingestion pipeline to complete
4. Ask questions in the chat input

**Run the CLI pipeline**
```bash
cd rag-system/app
python main.py
```
**Running the Terminal Command Line Interface Core** 
```bash
python rag-system/test/inspect_chunk.py
cd rag-system/app
python main.py
```

**Running Pipeline Integrity Diagnostics**
```bash
# Inspect chunk quality
python rag-system/test/inspect_chunk.py

# Test semantic search
python rag-system/test/test_search.py
```

---

## 📁 Project Structure

```
rag-system/
├── app/
│   ├── app.py             # Streamlit Application Workspace Entrypoint
│   ├── main.py            # Local command line CLI system orchestration loop
│   ├── loader.py          # PDF text extraction and character normalizing
│   ├── chunker.py         # NLTK sentence boundary segmentation engine
│   ├── embeding.py        # HuggingFace dense vector conversion manager
│   ├── vectore_store.py   # ChromaDB transaction and persistence handlers
│   ├── retriever.py       # Geometric similarity matching and score filtering
│   └── qa_pipeline.py     # Prompt grounding construction + cascading stream fallback routing
├── assets/
│   ├── image.png          # Dashboard layout overview capture
│   ├── image_pdfup.png    # Ingestion tracking system window screenshot
│   ├── image_scor.png     # Evaluation score components screenshot
│   └── image_chat.png     # Active conversational response capture
├── data/
│   └── ArtificiaL_.pdf    # Sample testing asset document
├── test/
│   ├── inspect_chunk.py   # Segmentation output formatting auditor
│   ├── test_search.py     # Database index semantic matching integrity tool
│   └── testdb_load.py     # Persistent disk read/write operational verification test
├── notebooks/
│   └── chunk_inspect.ipynb# Visual parsing inspection workspace
├── requirements.txt       # Production dependency pinning profile
└── README.md              # Project documentation profile
```

---

## 🖼️ Interface Walkthrough

### 1. Main Dashboard Overview
The main entry point showing the full layout of the system interface.

![Main App Dashboard](../assets/image.png)

---

### 2. Document Ingestion Panel
Tracks uploaded files and ensures duplicates are not processed twice.

![Document Ingestion Panel](../assets/image_pdfup.png)

---

### 3. Active Conversational Response
Streaming engine routes text token‑by‑token using local or cloud fallback models.

![Conversational Chat Interface](../assets/image_chat.png)

---

### 4. Metrics and Citation Expanders
Custom calculation block translates raw Euclidean vector distances into a clean confidence percentage, complete with document page numbers.

![Metrics and Citation Expanders](../assets/image_scor.png)


## 🔮 Future Improvements

- [ ] **Cross-Encoder Reranking** — Use a `cross-encoder/ms-marco-MiniLM` model to rerank retrieved chunks before LLM generation
- [ ] **OCR Fallback** — Handle scanned PDFs using `pytesseract` or `pymupdf` when `pypdf` returns empty text
- [ ] **FastAPI Backend** — Expose the pipeline as a REST API for integration with other services
- [ ] **Docker Deployment** — Containerize the full stack with a multi-stage Dockerfile for cloud scaling
- [ ] **RAG Evaluation** — Integrate `ragas` toolkit to benchmark faithfulness, semantic relevancy, and context recall accuracy



