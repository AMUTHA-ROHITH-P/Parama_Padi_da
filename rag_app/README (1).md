# 📚 EduRAG – AI Study Assistant
### A production-ready Retrieval-Augmented Generation (RAG) system for students

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                        STUDENT BROWSER                          │
│                    (Streamlit Frontend)                         │
└────────────────────────────┬────────────────────────────────────┘
                             │ PDF upload / Question
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│                    RAG BACKEND  (rag_backend.py)                │
│                                                                 │
│  ┌──────────────┐   ┌───────────────┐   ┌──────────────────┐   │
│  │ PDF          │   │  Text         │   │  Embedding       │   │
│  │ Extractor    │──▶│  Chunker      │──▶│  Model           │   │
│  │ (PyMuPDF)    │   │  (overlap)    │   │  (MiniLM-L6)     │   │
│  └──────────────┘   └───────────────┘   └────────┬─────────┘   │
│                                                   │ vectors     │
│  ┌────────────────────────────────────────────────▼─────────┐  │
│  │                  FAISS Vector Store                       │  │
│  │   (inner-product search + MMR diversity re-ranking)       │  │
│  └────────────────────────────────────────────────┬─────────┘  │
│                                                   │ top-k chunks│
│  ┌────────────────────────────────────────────────▼─────────┐  │
│  │               LLM Generator                              │  │
│  │   GPT-4o-mini  OR  Claude Sonnet  OR  Extractive fallback│  │
│  └────────────────────────────────────────────────┬─────────┘  │
│                                                   │ answer      │
└───────────────────────────────────────────────────┼────────────┘
                                                    ▼
                                        Student sees the answer
```

---

## Quick Start

### 1. Install dependencies
```bash
cd rag_app
pip install -r requirements.txt
```

### 2. Set your LLM API key (pick one)
```bash
# Option A: OpenAI
export OPENAI_API_KEY="sk-..."

# Option B: Anthropic Claude
export ANTHROPIC_API_KEY="sk-ant-..."

# Or create a .env file:
echo "OPENAI_API_KEY=sk-..." > .env
```

### 3. Run the app
```bash
streamlit run app.py
```

Open http://localhost:8501 in your browser.

---

## How It Works — Step by Step

### Step 1 · PDF Ingestion Pipeline

```
PDF file
  └─▶ PyMuPDF extracts text block-by-block, preserving font sizes
        └─▶ Large / bold text → detected as section headings
              └─▶ TextChunker splits into ~400-word chunks with 80-word overlap
                    └─▶ SentenceTransformer encodes each chunk → 384-dim vector
                          └─▶ FAISS IndexFlatIP stores all vectors
```

**Why overlap?** The last 80 words of each chunk are repeated at the start of the
next. This prevents answers being split across chunk boundaries.

**Why section detection?** Headings are stored as metadata so the LLM context
includes `[Chapter 3: Photosynthesis | Page 12]` — making answers more precise.

---

### Step 2 · Retrieval

```
Student question
  └─▶ Embedded to same 384-dim space as chunks
        └─▶ FAISS dot-product search → top-k candidates (e.g. k=4)
              └─▶ MMR re-ranking → removes near-duplicate chunks
                    └─▶ 4 diverse, relevant chunks returned
```

**MMR (Maximal Marginal Relevance)** balances relevance vs. diversity:
`score = λ × relevance − (1−λ) × similarity_to_already_selected`

This means if two chunks say the same thing, only the better one is kept.

---

### Step 3 · Generation

The LLM receives a carefully engineered prompt:

```
SYSTEM: You are EduRAG, a helpful AI tutor for school students…

DOCUMENT EXCERPTS:
[Biology_Textbook.pdf | Page 12 | Photosynthesis]
Photosynthesis is the process by which plants…

[Biology_Textbook.pdf | Page 14 | Light Reactions]
The light-dependent reactions occur in the thylakoid…

STUDENT QUESTION:
What happens during the light reactions of photosynthesis?
```

The LLM is instructed to answer **only from the excerpts**, preventing hallucination.

---

### Step 4 · Multi-Document & Multi-Student Scalability

- **Multiple PDFs**: All chunks go into a single FAISS index. Each chunk carries
  its source filename, so answers can cite multiple documents.
- **Multiple students**: Each Streamlit session gets its own Python process
  (Streamlit's default behaviour). For true multi-tenant isolation, deploy
  with Kubernetes and one pod per student session (see Docker section below).

---

## File Structure

```
rag_app/
├── app.py              # Streamlit frontend
├── rag_backend.py      # Full RAG pipeline (extract → chunk → embed → retrieve → generate)
├── requirements.txt    # Python dependencies
├── Dockerfile          # Container build
├── docker-compose.yml  # Multi-service deployment
└── README.md           # This file
```

---

## Configuration

| Parameter      | Default | Where to change            | Effect                                  |
|---------------|---------|----------------------------|-----------------------------------------|
| `chunk_size`   | 400 words | `TextChunker(chunk_size=)` | Larger = more context, slower embedding |
| `overlap`      | 80 words  | `TextChunker(overlap=)`    | More overlap = fewer boundary cuts      |
| `top_k`        | 4         | Streamlit sidebar slider   | More chunks = richer context            |
| `MODEL_NAME`   | `all-MiniLM-L6-v2` | `EmbeddingModel` | Swap for a larger model for better quality |
| LLM model      | `gpt-4o-mini` | `LLMGenerator._call_openai` | Use `gpt-4o` for best quality          |

---

## Docker Deployment (Multi-Student)

```bash
# Build and run
docker-compose up --build

# Scale to 4 instances behind a load balancer
docker-compose up --scale edurag=4
```

For production, add **nginx** as a reverse proxy and **Redis** to share the
FAISS index across instances.

---

## Improving the Model Over Time

1. **Log interactions**: Save (question, retrieved_chunks, answer) to a database.
2. **Identify failures**: Mark answers where students clicked "Not helpful".
3. **Fine-tune embeddings**: Use those logs to fine-tune the sentence-transformer
   with contrastive learning (positive = good retrieval, negative = bad).
4. **Prompt iteration**: Refine the system prompt based on answer quality.
5. **Hybrid search**: Add BM25 keyword search alongside dense vectors and
   combine scores with RRF (Reciprocal Rank Fusion) for harder factual queries.

---

## Embedding Model Alternatives

| Model                          | Size   | Quality  | Speed  |
|-------------------------------|--------|----------|--------|
| `all-MiniLM-L6-v2` (default)  | 80 MB  | Good     | Fast   |
| `all-mpnet-base-v2`           | 420 MB | Better   | Medium |
| `text-embedding-3-small` (OpenAI API) | Cloud | Best  | API call |
| `nomic-embed-text` (Ollama)   | Local  | Very good | Medium |

---

## FAQ

**Q: The app works without an API key?**  
A: Yes — it uses an extractive fallback that shows the most relevant text chunks directly. Set an API key for full AI-generated, synthesised answers.

**Q: How many pages can it handle?**  
A: Tested up to 500-page textbooks. For larger PDFs, increase `chunk_size` to reduce the number of chunks, or use a persistent vector DB like Chroma or Pinecone.

**Q: Can I use a local LLM (no API cost)?**  
A: Yes — replace `_call_openai` in `LLMGenerator` with an Ollama call:
```python
import ollama
resp = ollama.chat(model="llama3", messages=[...])
```
