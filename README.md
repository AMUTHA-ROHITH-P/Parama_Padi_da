# 📚 Fusion Learning – AI-Powered Study Assistant

> Upload your study PDFs, ask questions in plain English, and get intelligent, source-backed answers — powered by local Llama/Ollama, OpenAI GPT, Anthropic Claude, or zero-install extractive fallback.

---

## Table of Contents

1. [Project Overview](#project-overview)
2. [Tech Stack](#tech-stack)
3. [Overall Architecture](#overall-architecture)
4. [How PDF Analysis Works](#how-pdf-analysis-works)
5. [RAG Pipeline (Step-by-Step)](#rag-pipeline-step-by-step)
6. [LLM Backends](#llm-backends)
7. [Frontend Features (Streamlit UI)](#frontend-features-streamlit-ui)
8. [File Structure](#file-structure)
9. [Installation & Setup](#installation--setup)
10. [Running the App](#running-the-app)
11. [Troubleshooting](#troubleshooting)

---

## Project Overview

EduRAG is a **Retrieval-Augmented Generation (RAG)** application built for students. It lets you upload any PDF (textbook, notes, research paper) and ask questions about it in natural language. Instead of hallucinating answers, EduRAG retrieves the most relevant passages from your documents first, then feeds them to an LLM to generate accurate, grounded answers.

**Key Design Goals:**
- Works 100% offline/free using Ollama (local LLMs)
- Gracefully degrades — even works without any LLM via extractive fallback
- Clean educational UI with source attribution
- Multi-document support with persistent vector store

---

## Tech Stack

| Layer | Technology | Purpose |
|---|---|---|
| **Frontend** | Streamlit | Web UI, file uploads, chat interface |
| **PDF Parsing** | PyMuPDF (`fitz`) / pdfplumber | Text + heading extraction from PDFs |
| **Embeddings** | sentence-transformers (`all-MiniLM-L6-v2`) | Convert text to semantic vectors |
| **Embedding Fallback** | scikit-learn TF-IDF | If sentence-transformers not installed |
| **Vector Store** | FAISS (`IndexFlatIP`) | Fast semantic similarity search |
| **Vector Store Fallback** | NumPy dot-product | If FAISS not installed |
| **Local LLM** | Ollama (Qwen2, Llama3, Mistral, etc.) | Free, runs on your machine |
| **Cloud LLM (optional)** | OpenAI GPT-4o / GPT-4o-mini | Requires `OPENAI_API_KEY` |
| **Cloud LLM (optional)** | Anthropic Claude Haiku / Sonnet | Requires `ANTHROPIC_API_KEY` |
| **Language** | Python 3.10+ | Core backend logic |

---

## Overall Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        STREAMLIT UI (app.py)                    │
│  ┌──────────────┐  ┌───────────────────────┐  ┌─────────────┐   │
│  │  Sidebar     │  │   Chat Interface      │  │  Stats Bar  │   │
│  │  - Backend   │  │   - Q&A History       │  │  - Docs     │   │
│  │    selector  │  │   - Source badges     │  │  - Chunks   │   │
│  │  - PDF upload│  │   - Suggestions       │  │  - Q&As     │   │
│  └──────┬───────┘  └──────────┬────────────┘  └─────────────┘   │
└─────────┼─────────────────────┼─────────────────────────────────┘
          │ ingest_pdf()        │ answer()
          ▼                    ▼
┌─────────────────────────────────────────────────────────────────┐
│                     RAGBackend (rag_backend.py)                 │
│                                                                 │
│  ┌─────────────┐   ┌──────────────┐   ┌───────────────────────┐ │
│  │ PDFExtractor│ → │ TextChunker  │ → │   EmbeddingModel      │ │
│  │             │   │              │   │ (sentence-transformers│ │
│  │ PyMuPDF     │   │ 400-word     │   │  or TF-IDF fallback)  │ │
│  │ pdfplumber  │   │ chunks,      │   └──────────┬────────────┘ │
│  │ (fallback)  │   │ 80-word      │              │              │
│  └─────────────┘   │ overlap      │              ▼              │
│                    └──────────────┘   ┌──────────────────────┐  │
│                                       │    VectorStore       │  │
│                                       │ (FAISS or NumPy)     │  │
│                                       │  - add()             │  │
│                                       │  - search() + MMR    │  │
│                                       └──────────┬───────────┘  │
│                                                  │              │
│                                       ┌──────────▼───────────┐  │
│                                       │    LLMGenerator      │  │
│                                       │  1. Ollama (local)   │  │
│                                       │  2. OpenAI           │  │
│                                       │  3. Anthropic        │  │
│                                       │  4. Extractive       │  │
│                                       └──────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
```

---

## How PDF Analysis Works

When you upload a PDF, it goes through a 4-stage pipeline before anything is stored.

### Stage 1 — Text Extraction (`PDFExtractor`)

**Primary method: PyMuPDF (`fitz`)**

PyMuPDF reads PDFs at the block/span level, giving access to text, font size, and font flags (bold, italic) for every piece of text on every page. This allows EduRAG to:

- Extract all page text in reading order
- Detect **headings** by checking if text is bold (`flags & 2**4`) or large (`size > 14`) and under 120 characters
- Store headings separately from body text for section-aware chunking

```
Page → Blocks → Lines → Spans → {text, font_size, is_bold, is_large}
```

**Fallback: pdfplumber**

If PyMuPDF isn't installed, pdfplumber is used. It extracts plain text per page without heading detection (headings list is empty), but the rest of the pipeline works identically.

Each page produces a dictionary:
```python
{
  "page": 3,
  "text": "Photosynthesis is the process by which...",
  "headings": ["Chapter 2: Photosynthesis"]
}
```

---

### Stage 2 — Intelligent Chunking (`TextChunker`)

Raw pages are too long to embed meaningfully and too large for LLM context windows. The chunker breaks pages into overlapping chunks of **~400 words** with **80-word overlap**.

**How it works:**
1. Text is first split on double newlines (paragraph boundaries)
2. Paragraphs are accumulated into a buffer
3. When the buffer exceeds 400 words, it is flushed as a Chunk — but the last 80 words are carried over into the next chunk (overlap) to preserve context across boundaries
4. The last active heading from `PDFExtractor` is stamped onto each chunk as its `section`

Each `Chunk` stores:
```python
Chunk(
  text="...",          # chunk body
  source="chapter1.pdf",
  page=3,
  section="Photosynthesis",  # nearest heading
  chunk_id=42
)
```

**Why overlap?** If a key sentence sits at the boundary between two chunks, overlap ensures it appears in at least one chunk's context when retrieved.

---

### Stage 3 — Embedding (`EmbeddingModel`)

Every chunk's text is converted into a high-dimensional vector using `sentence-transformers/all-MiniLM-L6-v2`, a lightweight 80MB model that runs locally with no internet needed.

- Vectors are **384 dimensions**
- Embeddings are **L2-normalized** (unit length), so cosine similarity = dot product
- The same model is used to embed user questions at query time, keeping the vector space consistent

**TF-IDF fallback:** If sentence-transformers isn't installed, scikit-learn's `TfidfVectorizer` is used with 768 features and L2 normalization — no ML model required.

---

### Stage 4 — Storing in the Vector Store (`VectorStore`)

Embeddings are added to a **FAISS `IndexFlatIP`** (inner product index). Since vectors are normalized, inner product = cosine similarity.

- `VectorStore.add()` indexes embeddings and stores chunk metadata in a parallel list
- `VectorStore.search()` takes a query vector, finds top-k nearest chunks by cosine similarity, then applies MMR reranking

**NumPy fallback:** Without FAISS, a plain NumPy matrix holds all vectors and search is done via `matrix @ query_vector`.

---

## RAG Pipeline (Step-by-Step)

### At Upload Time (Ingestion)
```
PDF File
  → PDFExtractor.extract()     → list of {page, text, headings}
  → TextChunker.chunk_pages()  → list of Chunk objects
  → EmbeddingModel.encode()    → numpy array of shape (n_chunks, 384)
  → VectorStore.add()          → indexed in FAISS
```

### At Query Time (Retrieval + Generation)
```
User Question
  → EmbeddingModel.encode([question])   → query vector (384-dim)
  → VectorStore.search(query_vec, k=4)  → top-4 RetrievedChunks (with MMR)
  → LLMGenerator.generate(question, chunks)
       → builds context string from chunks
       → calls LLM (Ollama / OpenAI / Anthropic / extractive)
  → Returns {answer, sources, retrieved}
```

### MMR Reranking (Maximal Marginal Relevance)

After retrieving top-k chunks by cosine similarity, MMR reranking ensures diversity. It iteratively selects chunks that are both **relevant to the query** and **different from already-selected chunks**:

```
MMR score = λ × relevance − (1−λ) × similarity_to_selected
```

With `λ = 0.5`, relevance and diversity are weighted equally. This prevents the top-4 results from being near-duplicate passages about the same sentence.

---

## LLM Backends

EduRAG auto-detects the best available backend in this priority order:

### 1. 🦙 Ollama (Local, 100% Free)
- Runs LLMs on your own machine — no API key, no cost, no data sent to cloud
- Detected by pinging `http://localhost:11500`
- Preferred model list (in order): `qwen2:1.5b`, `phi3`, `mistral`, `llama3.2`, `llama3.1`, `llama3`, `llama2`, `gemma2`
- Uses the `ollama` Python library if installed, otherwise falls back to raw HTTP calls

### 2. 🟢 OpenAI GPT
- Used if `OPENAI_API_KEY` env variable is set
- Default model: `gpt-4o-mini` (fast and cheap)
- Also supports: `gpt-4o`

### 3. 🟣 Anthropic Claude
- Used if `ANTHROPIC_API_KEY` env variable is set
- Default model: `claude-haiku-4-5-20251001` (fastest)
- Also supports: `claude-sonnet-4-6`

### 4. 📄 Extractive Fallback (Always Available)
- No LLM required at all — zero installation
- Returns the raw top retrieved chunk plus summaries of the next 2
- Includes a tip suggesting users install Ollama for AI-generated answers

The system prompt instructs the LLM to act as **EduRAG**, an educational tutor, answering only from the provided document excerpts and being honest when information isn't present.

---

## Frontend Features (Streamlit UI)

### Sidebar

**AI Model Panel**
- Live Ollama status indicator (green if running, number of models available)
- Backend selector dropdown — only shows backends that are actually available
- Model selector (dynamically populated from Ollama's available models or hardcoded lists for OpenAI/Anthropic)
- "Apply Model" button to switch backend at runtime without restarting
- Optional API key input fields for OpenAI and Anthropic (stored in environment for the session)

**Document Upload**
- Multi-PDF file uploader
- Each PDF is written to a temp file, ingested through the full RAG pipeline, and confirmed with chunk count
- Indexed document list with filenames

**Settings**
- `top_k` slider (2–8): controls how many chunks are retrieved per question
- Toggle to show/hide source excerpt badges under each answer
- "Clear All & Start Over" button that resets the vector store and chat history

### Main Area

**Hero Banner**
- Styled gradient header with app name and tagline

**Stats Bar**
- Three live-updating metric boxes: Documents indexed, Knowledge Chunks, Q&As answered

**Chat Interface**
- Full scrollable conversation history with styled bubbles (user right-aligned in navy, AI left-aligned in white)
- Each answer shows which backend generated it
- Source badges below each AI answer showing `filename (page number)`
- Chat form with text input and submit button (clears on submit)

**Suggestion Buttons**
- When documents are loaded but no questions asked yet, 4 starter question buttons appear:
  - "Summarise the main topics in this document."
  - "What are the key definitions introduced?"
  - "Explain the most important concept simply."
  - "Give me 5 quiz questions based on this material."

---

## File Structure

```
Parama_Padi_da/
├── app.py              # Streamlit frontend — all UI logic
├── rag_backend.py      # Complete RAG pipeline:
│                       #   PDFExtractor, TextChunker,
│                       #   EmbeddingModel, VectorStore,
│                       #   LLMGenerator, RAGBackend
├── requirements.txt    # Python dependencies (see below)
└── README.md           # This file
```

---

## Installation & Setup

### Prerequisites
- Python 3.10 or higher
- (Recommended) [Ollama](https://ollama.com) for free local LLM

### 1. Clone / download the project

```bash
cd Parama_Padi_da
```

### 2. Create and activate a virtual environment

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Mac/Linux
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install streamlit pymupdf pdfplumber sentence-transformers faiss-cpu numpy scikit-learn ollama
```

Optional (for cloud LLMs):
```bash
pip install openai anthropic
```

### 4. Set up Ollama (recommended — free local AI)

1. Download Ollama from https://ollama.com
2. Configure it to run on port 11500 (set `OLLAMA_HOST=http://127.0.0.1:11500`)
3. Pull a model:

```bash
ollama pull qwen2:1.5b    # Recommended — fast, 934 MB
# or
ollama pull llama3.2      # Larger but more capable
```

### 5. (Optional) Set API keys for cloud LLMs

```bash
# Windows PowerShell
$env:OPENAI_API_KEY = "sk-..."
$env:ANTHROPIC_API_KEY = "sk-ant-..."

# Mac/Linux
export OPENAI_API_KEY="sk-..."
export ANTHROPIC_API_KEY="sk-ant-..."
```

---

## Running the App

```bash
# Make sure Ollama is running first (in a separate terminal):
ollama serve

# Then start the app:
streamlit run app.py
```

Open your browser to `http://localhost:8501`.

---

## Troubleshooting

### `model 'qwen2' not found (status code: 404)`
The model name needs the full tag. Run `ollama list` to see what's installed (e.g. `qwen2:1.5b`). The app now uses full names like `qwen2:1.5b` in its model list. If you see this error with an older version of `rag_backend.py`, ensure `ollama_list_models()` does **not** strip the tag with `.split(":")[0]`.

### Ollama not detected / `ollama_is_running()` returns False
Check that Ollama is configured on port 11500 (`OLLAMA_HOST=http://127.0.0.1:11500`) and that `ollama serve` is running in a separate terminal before launching Streamlit.

### `No text extracted from PDF`
Some PDFs are scanned images rather than real text. PyMuPDF and pdfplumber both need embedded text. For scanned PDFs, you would need an OCR tool like `pytesseract` (not currently included).

### Slow first question
The embedding model (`all-MiniLM-L6-v2`) is downloaded on first use (~80 MB). Subsequent runs use the cached model.

### FAISS not installing
FAISS can be tricky on Windows. Try:
```bash
pip install faiss-cpu --no-cache-dir
```
If it still fails, the app will automatically fall back to NumPy-based search — no action needed.