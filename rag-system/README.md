# ⚙️ Enterprise RAG System — Chat with PDF

A production-grade **Retrieval-Augmented Generation (RAG)** pipeline that lets you upload any PDF and have a grounded, citation-backed conversation with its contents. Built with a modular architecture, multi-LLM fallback chain, and a Streamlit UI.

---

## 📌 Project Overview

Most LLMs hallucinate when asked about private or domain-specific documents. This system solves that by retrieving the most semantically relevant chunks from your document first, then grounding the LLM's answer strictly within that retrieved context — no guessing, no fabrication.

Upload a PDF → ask questions → get precise answers with page-level citations.

---

## 🏗️ Architecture

```text
┌─────────────────────────────────────────────────────────────┐
│                        USER (Streamlit UI)                  │
└──────────────────────────┬──────────────────────────────────┘
                           │  Upload PDF + Ask Question
                           ▼
┌─────────────────────────────────────────────────────────────┐
│  INGESTION PIPELINE                                         │
│                                                             │
│  loader.py          chunker.py          embedding.py        │
│  ┌──────────┐      ┌──────────┐        ┌──────────────┐    │
│  │ Extract  │ ───► │  NLTK    │ ──────► │ all-MiniLM   │    │
│  │ & Clean  │      │ Sentence │        │ -L6-v2       │    │
│  │ PDF Text │      │ Splitter │        │ (HuggingFace)│    │
│  └──────────┘      └──────────┘        └──────┬───────┘    │
│                                               │             │
│                                               ▼             │
│                                    ┌──────────────────┐    │
│                                    │  vector_store.py │    │
│                                    │  ChromaDB        │    │
│                                    │  (Persistent)    │    │
│                                    └──────────────────┘    │
└─────────────────────────────────────────────────────────────┘
                           │  Query Time
                           ▼
┌─────────────────────────────────────────────────────────────┐
│  QUERY PIPELINE                                             │
│                                                             │
│  retriever.py                    qa_pipeline.py             │
│  ┌─────────────────────┐        ┌──────────────────────┐   │
│  │ Similarity Search   │        │ Grounded Prompt      │   │
│  │ + Distance Filter   │ ──────►│ + Fallback LLM Chain │   │
│  │ (threshold: 0.80)   │        │                      │   │
│  └─────────────────────┘        │ 1. Ollama (local)    │   │
│                                 │ 2. HuggingFace API   │   │
│                                 │ 3. OpenAI GPT-4o     │   │
│                                 │ 4. Google Gemini     │   │
│                                 └──────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
                           │
                           ▼
              Answer + Page Citations + Snippets
```

---

## ✅ Features

- **PDF Ingestion** — Extracts and cleans text from any digital PDF, page by page
- **Sentence-Aware Chunking** — Uses NLTK sentence tokenizer to avoid cutting mid-sentence (chunk size: 800, overlap: 150)
- **Semantic Embeddings** — `all-MiniLM-L6-v2` via HuggingFace for fast, high-quality vector representations
- **Persistent Vector Store** — ChromaDB stores embeddings locally across sessions
- **Distance-Filtered Retrieval** — Only chunks with a relevance score ≤ 0.80 are passed to the LLM
- **Multi-LLM Fallback Chain** — Tries Ollama → HuggingFace → OpenAI → Google Gemini in sequence
- **Grounded Prompt Design** — LLM is strictly instructed to answer only from retrieved context
- **Page-Level Citations** — Every answer includes source file, page number, distance score, and a text snippet
- **Streamlit Chat UI** — Full conversational interface with chat history and sidebar document control
- **Chunk Quality Diagnostics** — `inspect_chunk.py` audits broken sentences and short chunks

---

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| UI | Streamlit |
| PDF Parsing | pypdf |
| Text Chunking | LangChain + NLTK |
| Embeddings | HuggingFace `all-MiniLM-L6-v2` via `sentence-transformers` |
| Vector Database | ChromaDB (persistent local) |
| LLM — Local | Ollama (llama3) |
| LLM — Cloud | OpenAI GPT-4o-mini, Google Gemini 2.5 Flash, HuggingFace Inference API |
| Environment | python-dotenv |
| Language | Python 3.10+ |

---

## 📦 Installation

**1. Clone the repository**
```bash
git clone https://github.com/ajme-abes/RAG-Multimodal-SafeAI.git
cd your-repo/rag-system
```

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

> At least one API key is required. If you have Ollama installed locally, no key is needed for basic usage.

**5. (Optional) Install Ollama for local inference**

Download from [ollama.com](https://ollama.com) then pull the model:
```bash
ollama pull llama3
```

---

## 🚀 Usage

**Run the Streamlit app**
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

**Run diagnostic tools**
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
│   ├── app.py              # Streamlit UI — main entry point
│   ├── main.py             # CLI pipeline runner
│   ├── loader.py           # PDF extraction & text cleaning
│   ├── chunker.py          # NLTK sentence-aware text splitting
│   ├── embeding.py         # HuggingFace embedding model loader
│   ├── vectore_store.py    # ChromaDB vector store management
│   ├── retriever.py        # Similarity search with distance filtering
│   └── qa_pipeline.py      # Prompt construction + multi-LLM fallback
├── data/
│   └── ArtificiaL_.pdf     # Sample test document
├── test/
│   ├── inspect_chunk.py    # Chunk quality auditor
│   ├── test_search.py      # Semantic search integration test
│   └── testdb_load.py      # ChromaDB connection test
├── notebooks/
│   └── chunk_inspect.ipynb # Jupyter chunk inspection notebook
├── requirements.txt
└── README.md
```

---

## 🖼️ Screenshots

> _Add screenshots here after running the app._
>
> Suggested captures:
> - Sidebar with uploaded PDF and success message
> - Chat interface with a question and answer
> - Citation expander showing page snippet and distance score

---

## 🔮 Future Improvements

- [ ] **Conversation Memory** — Pass recent chat history into the prompt so the LLM understands follow-up questions
- [ ] **Streaming Responses** — Stream LLM tokens to the UI in real-time instead of waiting for the full response
- [ ] **Multi-Document Support** — Query across multiple uploaded PDFs simultaneously without wiping the vector store
- [ ] **Confidence Gate** — Block LLM response entirely when best retrieval score exceeds threshold (prevent hallucination on off-topic queries)
- [ ] **Cross-Encoder Reranking** — Use a `cross-encoder/ms-marco-MiniLM` model to rerank retrieved chunks before LLM generation
- [ ] **OCR Fallback** — Handle scanned PDFs using `pytesseract` or `pymupdf` when `pypdf` returns empty text
- [ ] **FastAPI Backend** — Expose the pipeline as a REST API for integration with other services
- [ ] **Docker Deployment** — Containerize the full stack for one-command deployment anywhere
- [ ] **RAG Evaluation** — Integrate `ragas` library to score faithfulness, answer relevancy, and context recall
- [ ] **Async Embedding** — Parallelize chunk embedding for faster ingestion of large documents
