"""
rag_backend.py
==============
Complete RAG pipeline:
  1. PDF text extraction  (PyMuPDF / pdfplumber fallback)
  2. Intelligent chunking (section-aware, overlap)
  3. Embedding           (sentence-transformers, local & free)
  4. Vector store        (FAISS, in-memory)
  5. Retrieval           (semantic top-k with MMR de-duplication)
  6. Generation          (Ollama/Llama3 LOCAL  ← default, 100% free
                          OpenAI GPT-4o-mini   ← optional, needs key
                          Anthropic Claude     ← optional, needs key
                          Extractive fallback  ← always works, no install needed)

LLM PRIORITY ORDER:
  1. Ollama (local Llama / Mistral / Gemma — runs on your machine, FREE)
  2. OpenAI  (if OPENAI_API_KEY is set)
  3. Anthropic (if ANTHROPIC_API_KEY is set)
  4. Extractive fallback (shows raw retrieved text — no AI needed)
"""

from __future__ import annotations

import os
import re
import textwrap
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

# ── Optional heavy imports (graceful degradation) ────────────────────────────
try:
    import fitz  # PyMuPDF
    HAS_FITZ = True
except ImportError:
    HAS_FITZ = False

try:
    import pdfplumber
    HAS_PDFPLUMBER = True
except ImportError:
    HAS_PDFPLUMBER = False

try:
    from sentence_transformers import SentenceTransformer
    HAS_ST = True
except ImportError:
    HAS_ST = False

try:
    import faiss
    HAS_FAISS = True
except ImportError:
    HAS_FAISS = False

try:
    import openai
    HAS_OPENAI = True
except ImportError:
    HAS_OPENAI = False

try:
    import anthropic
    HAS_ANTHROPIC = True
except ImportError:
    HAS_ANTHROPIC = False

try:
    import ollama as _ollama_lib          # pip install ollama
    HAS_OLLAMA_LIB = True
except ImportError:
    HAS_OLLAMA_LIB = False

import urllib.request, json as _json    # stdlib — used to ping Ollama REST API


# ── Data structures ───────────────────────────────────────────────────────────

@dataclass
class Chunk:
    text: str
    source: str          # filename
    page: int
    section: str = ""
    chunk_id: int = 0


@dataclass
class RetrievedChunk:
    chunk: Chunk
    score: float


# ── PDF Extractor ─────────────────────────────────────────────────────────────

class PDFExtractor:
    """Extract structured text from PDFs, preserving headings / page numbers."""

    def extract(self, pdf_path: str) -> list[dict]:
        """Return list of {page, text, headings}."""
        if HAS_FITZ:
            return self._extract_fitz(pdf_path)
        elif HAS_PDFPLUMBER:
            return self._extract_pdfplumber(pdf_path)
        else:
            raise RuntimeError("Install pymupdf or pdfplumber: pip install pymupdf pdfplumber")

    def _extract_fitz(self, path: str) -> list[dict]:
        pages = []
        doc = fitz.open(path)
        for page_num, page in enumerate(doc, 1):
            blocks = page.get_text("dict")["blocks"]
            page_text = []
            headings  = []
            for block in blocks:
                if block.get("type") != 0:
                    continue
                for line in block.get("lines", []):
                    for span in line.get("spans", []):
                        text = span["text"].strip()
                        if not text:
                            continue
                        size = span.get("size", 12)
                        flags = span.get("flags", 0)
                        is_bold   = bool(flags & 2**4)
                        is_large  = size > 14
                        if (is_bold or is_large) and len(text) < 120:
                            headings.append(text)
                        page_text.append(text)
            pages.append({
                "page": page_num,
                "text": " ".join(page_text),
                "headings": headings,
            })
        doc.close()
        return pages

    def _extract_pdfplumber(self, path: str) -> list[dict]:
        pages = []
        with pdfplumber.open(path) as pdf:
            for page_num, page in enumerate(pdf.pages, 1):
                text = page.extract_text() or ""
                pages.append({
                    "page": page_num,
                    "text": text,
                    "headings": [],
                })
        return pages


# ── Text Chunker ──────────────────────────────────────────────────────────────

class TextChunker:
    """
    Splits page text into overlapping chunks, honouring:
     - paragraph boundaries (double newlines / blank lines)
     - max token budget (~words × 1.3)
     - overlap for context continuity
    """

    def __init__(self, chunk_size: int = 400, overlap: int = 80):
        self.chunk_size = chunk_size   # words
        self.overlap    = overlap      # words

    def chunk_pages(self, pages: list[dict], source: str) -> list[Chunk]:
        chunks: list[Chunk] = []
        cid = 0
        current_section = ""

        for page_info in pages:
            # Track the latest heading as the running section
            if page_info["headings"]:
                current_section = page_info["headings"][-1]

            paragraphs = self._split_paragraphs(page_info["text"])
            buffer: list[str] = []
            buf_words = 0

            for para in paragraphs:
                words = para.split()
                if not words:
                    continue

                # If adding this paragraph overflows, flush
                if buf_words + len(words) > self.chunk_size and buffer:
                    chunk_text = " ".join(buffer)
                    chunks.append(Chunk(
                        text=chunk_text,
                        source=source,
                        page=page_info["page"],
                        section=current_section,
                        chunk_id=cid,
                    ))
                    cid += 1
                    # Overlap: keep last N words
                    overlap_text = " ".join(chunk_text.split()[-self.overlap:])
                    buffer = [overlap_text]
                    buf_words = len(overlap_text.split())

                buffer.append(para)
                buf_words += len(words)

            # Flush remainder
            if buffer:
                chunks.append(Chunk(
                    text=" ".join(buffer),
                    source=source,
                    page=page_info["page"],
                    section=current_section,
                    chunk_id=cid,
                ))
                cid += 1

        return chunks

    def _split_paragraphs(self, text: str) -> list[str]:
        # Split on double newlines, then on sentence-ending punctuation
        raw = re.split(r"\n{2,}", text)
        paras = []
        for p in raw:
            p = p.replace("\n", " ").strip()
            if p:
                paras.append(p)
        return paras


# ── Embedding Model ───────────────────────────────────────────────────────────

class EmbeddingModel:
    """
    Wraps sentence-transformers (local, free).
    Falls back to a simple TF-IDF bag-of-words if not installed.
    """
    MODEL_NAME = "all-MiniLM-L6-v2"   # 80 MB, fast, good quality

    def __init__(self):
        self._model = None
        self._tfidf = None
        self._use_st = HAS_ST

    def _load(self):
        if self._model is None:
            if self._use_st:
                self._model = SentenceTransformer(self.MODEL_NAME)
            else:
                from sklearn.feature_extraction.text import TfidfVectorizer
                self._tfidf = TfidfVectorizer(max_features=768, stop_words="english")

    def encode(self, texts: list[str]) -> np.ndarray:
        self._load()
        if self._use_st:
            return self._model.encode(texts, show_progress_bar=False, normalize_embeddings=True)
        else:
            # TF-IDF path
            if not hasattr(self._tfidf, "vocabulary_"):
                vecs = self._tfidf.fit_transform(texts).toarray().astype("float32")
            else:
                vecs = self._tfidf.transform(texts).toarray().astype("float32")
            # L2-normalise
            norms = np.linalg.norm(vecs, axis=1, keepdims=True) + 1e-10
            return (vecs / norms).astype("float32")

    @property
    def dim(self) -> int:
        self._load()
        if self._use_st:
            return self._model.get_sentence_embedding_dimension()
        return 768


# ── Vector Store ──────────────────────────────────────────────────────────────

class VectorStore:
    """FAISS-backed store with fallback to numpy dot-product search."""

    def __init__(self, dim: int):
        self.dim    = dim
        self.chunks: list[Chunk] = []
        self._index = None
        self._vecs: np.ndarray | None = None
        self._init_index()

    def _init_index(self):
        if HAS_FAISS:
            self._index = faiss.IndexFlatIP(self.dim)   # inner-product (cosine after norm)
        else:
            self._vecs = np.zeros((0, self.dim), dtype="float32")

    def add(self, chunks: list[Chunk], embeddings: np.ndarray):
        self.chunks.extend(chunks)
        emb = embeddings.astype("float32")
        if HAS_FAISS:
            self._index.add(emb)
        else:
            self._vecs = np.vstack([self._vecs, emb]) if self._vecs.shape[0] else emb

    def search(self, query_vec: np.ndarray, k: int) -> list[RetrievedChunk]:
        if not self.chunks:
            return []
        k = min(k, len(self.chunks))
        qv = query_vec.astype("float32").reshape(1, -1)

        if HAS_FAISS:
            scores, indices = self._index.search(qv, k)
            results = [
                RetrievedChunk(chunk=self.chunks[i], score=float(s))
                for s, i in zip(scores[0], indices[0]) if i >= 0
            ]
        else:
            dots = (self._vecs @ qv.T).flatten()
            top_ids = np.argsort(dots)[::-1][:k]
            results = [
                RetrievedChunk(chunk=self.chunks[i], score=float(dots[i]))
                for i in top_ids
            ]

        # MMR-style de-duplication: penalise near-duplicate chunks
        return self._mmr_rerank(results, qv.flatten(), lam=0.5)

    def _mmr_rerank(
        self, results: list[RetrievedChunk], qv: np.ndarray, lam: float
    ) -> list[RetrievedChunk]:
        """Maximal Marginal Relevance to increase diversity."""
        if len(results) <= 1:
            return results
        selected: list[RetrievedChunk] = [results[0]]
        remaining = results[1:]
        while remaining:
            best, best_score = None, -1e9
            for cand in remaining:
                rel = cand.score
                # similarity to already-selected
                sim_to_sel = max(
                    float(np.dot(
                        self._chunk_vec(s.chunk),
                        self._chunk_vec(cand.chunk),
                    ))
                    for s in selected
                )
                mmr = lam * rel - (1 - lam) * sim_to_sel
                if mmr > best_score:
                    best_score = mmr
                    best = cand
            selected.append(best)
            remaining.remove(best)
        return selected

    def _chunk_vec(self, chunk: Chunk) -> np.ndarray:
        """Retrieve stored vector for a chunk by ID."""
        if HAS_FAISS:
            vec = np.zeros(self.dim, dtype="float32")
            self._index.reconstruct(chunk.chunk_id, vec)
            return vec
        else:
            return self._vecs[chunk.chunk_id]

    def clear(self):
        self.chunks = []
        self._init_index()

    def __len__(self):
        return len(self.chunks)


# ── LLM Generator ─────────────────────────────────────────────────────────────

# Models available per backend
OLLAMA_MODELS  = ["qwen2:1.5b","phi3", "mistral","llama3.2", "llama3.1", "llama3", "llama2", "gemma2"]
OPENAI_MODELS  = ["gpt-4o-mini", "gpt-4o"]
CLAUDE_MODELS  = ["claude-haiku-4-5-20251001", "claude-sonnet-4-6"]


def ollama_is_running() -> bool:
    """Return True if the Ollama server is reachable at localhost:11500."""
    try:
        urllib.request.urlopen("http://localhost:11500", timeout=1)
        return True
    except Exception:
        return False


def ollama_list_models() -> list[str]:
    """Return models currently pulled in Ollama (empty list on failure)."""
    try:
        with urllib.request.urlopen("http://localhost:11500/api/tags", timeout=2) as r:
            data = _json.loads(r.read())
            return [m["name"] for m in data.get("models", [])]
    except Exception:
        return []


class LLMGenerator:
    """
    Multi-backend LLM generator.

    Priority when backend='auto':
      1. Ollama  (local — completely free, runs on your machine)
      2. OpenAI  (cloud — needs OPENAI_API_KEY)
      3. Anthropic (cloud — needs ANTHROPIC_API_KEY)
      4. Extractive fallback (no LLM needed at all)

    You can also force a backend: 'ollama', 'openai', 'anthropic', 'extractive'
    """

    SYSTEM_PROMPT = textwrap.dedent("""\
        You are EduRAG, a helpful and encouraging AI tutor for school students.
        You answer questions ONLY using the provided document excerpts.
        Your answers should be:
        - Clear, structured, and educational
        - Written in simple language appropriate for students
        - Supported by specific information from the excerpts
        - Honest when the excerpts don't contain enough info (say so clearly)
        Format your answer with headings and bullet points where appropriate.
    """)

    def __init__(self, backend: str = "auto", model: str = ""):
        """
        backend : 'auto' | 'ollama' | 'openai' | 'anthropic' | 'extractive'
        model   : override the default model name for the chosen backend
        """
        self.backend = backend
        self.model   = model

    # ── Public ───────────────────────────────────────────────────────────────

    def generate(self, question: str, context_chunks: list[RetrievedChunk]) -> str:
        context = self._build_context(context_chunks)
        prompt  = f"DOCUMENT EXCERPTS:\n{context}\n\nSTUDENT QUESTION:\n{question}"
        backend = self._resolve_backend()

        if backend == "ollama":
            return self._call_ollama(prompt)
        elif backend == "openai":
            return self._call_openai(prompt)
        elif backend == "anthropic":
            return self._call_anthropic(prompt)
        else:
            return self._extractive_answer(context_chunks)

    def active_backend(self) -> str:
        """Return which backend will actually be used right now."""
        return self._resolve_backend()

    # ── Internal ──────────────────────────────────────────────────────────────

    def _resolve_backend(self) -> str:
        if self.backend != "auto":
            return self.backend
        # Auto-detect best available
        if ollama_is_running():
            return "ollama"
        if HAS_OPENAI and os.getenv("OPENAI_API_KEY"):
            return "openai"
        if HAS_ANTHROPIC and os.getenv("ANTHROPIC_API_KEY"):
            return "anthropic"
        return "extractive"

    def _build_context(self, chunks: list[RetrievedChunk]) -> str:
        parts = []
        for rc in chunks:
            c = rc.chunk
            header = (
                f"[{c.source} | Page {c.page}"
                + (f" | {c.section}" if c.section else "")
                + "]"
            )
            parts.append(f"{header}\n{c.text}")
        return "\n\n---\n\n".join(parts)

    # ── Ollama / Llama ────────────────────────────────────────────────────────

    def _call_ollama(self, prompt: str) -> str:
        model = self._best_ollama_model()

        # Prefer the ollama Python library if installed
        if HAS_OLLAMA_LIB:
            resp = _ollama_lib.chat(
                model=model,
                messages=[
                    {"role": "system",  "content": self.SYSTEM_PROMPT},
                    {"role": "user",    "content": prompt},
                ],
            )
            return resp["message"]["content"].strip()

        # Fallback: raw HTTP call (no extra library needed)
        payload = _json.dumps({
            "model": model,
            "messages": [
                {"role": "system", "content": self.SYSTEM_PROMPT},
                {"role": "user",   "content": prompt},
            ],
            "stream": False,
        }).encode()
        req = urllib.request.Request(
            "http://localhost:11500/api/chat",
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=120) as r:
            data = _json.loads(r.read())
        return data["message"]["content"].strip()

    def _best_ollama_model(self) -> str:
        """Pick the best available Ollama model from the preferred list."""
        available = ollama_list_models()
        for preferred in OLLAMA_MODELS:
            if preferred in available:
                return preferred
        # Return first available, or fallback
        return available[0] if available else "qwen2:1.5b"

    # ── OpenAI ────────────────────────────────────────────────────────────────

    def _call_openai(self, prompt: str) -> str:
        model = self.model or "gpt-4o-mini"
        client = openai.OpenAI()
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": self.SYSTEM_PROMPT},
                {"role": "user",   "content": prompt},
            ],
            temperature=0.3,
            max_tokens=1024,
        )
        return resp.choices[0].message.content.strip()

    # ── Anthropic ─────────────────────────────────────────────────────────────

    def _call_anthropic(self, prompt: str) -> str:
        model = self.model or "claude-haiku-4-5-20251001"
        client = anthropic.Anthropic()
        resp = client.messages.create(
            model=model,
            max_tokens=1024,
            system=self.SYSTEM_PROMPT,
            messages=[{"role": "user", "content": prompt}],
        )
        return resp.content[0].text.strip()

    # ── Extractive fallback ───────────────────────────────────────────────────

    def _extractive_answer(self, chunks: list[RetrievedChunk]) -> str:
        """Shows the most relevant raw text when no LLM is available."""
        if not chunks:
            return "No relevant content found."
        c = chunks[0].chunk
        answer = f"**📄 From:** {c.source} (Page {c.page})\n\n{c.text[:1200]}"
        if len(chunks) > 1:
            answer += "\n\n**Also relevant:**\n"
            for rc in chunks[1:3]:
                answer += f"\n> *{rc.chunk.source}, p.{rc.chunk.page}* — {rc.chunk.text[:300]}…\n"
        answer += (
            "\n\n---\n*💡 Tip: This is the raw extracted text. "
            "For AI-generated explanations, install [Ollama](https://ollama.com) "
            "and run `ollama pull llama3.2` — it's completely free!*"
        )
        return answer


# ── RAG Backend (main class) ──────────────────────────────────────────────────

class RAGBackend:
    """Orchestrates the full RAG pipeline."""

    def __init__(self, backend: str = "auto", model: str = ""):
        self._extractor  = PDFExtractor()
        self._chunker    = TextChunker(chunk_size=400, overlap=80)
        self._embedder   = EmbeddingModel()
        self._store: VectorStore | None = None
        self._generator  = LLMGenerator(backend=backend, model=model)
        self._doc_names: list[str] = []

    def set_backend(self, backend: str, model: str = ""):
        """Switch LLM backend at runtime (e.g. from the UI)."""
        self._generator.backend = backend
        self._generator.model   = model

    def active_backend(self) -> str:
        return self._generator.active_backend()

    def ollama_status(self) -> dict:
        running = ollama_is_running()
        models  = ollama_list_models() if running else []
        return {"running": running, "models": models}

    def _get_store(self) -> VectorStore:
        if self._store is None:
            self._store = VectorStore(dim=self._embedder.dim)
        return self._store

    # ── Public API ────────────────────────────────────────────────────────────

    def ingest_pdf(self, pdf_path: str, display_name: str) -> dict:
        try:
            # 1. Extract
            pages  = self._extractor.extract(pdf_path)
            # 2. Chunk
            chunks = self._chunker.chunk_pages(pages, display_name)
            if not chunks:
                return {"success": False, "error": "No text extracted from PDF."}
            # 3. Embed
            texts  = [c.text for c in chunks]
            embs   = self._embedder.encode(texts)
            # 4. Store
            store  = self._get_store()
            store.add(chunks, embs)
            self._doc_names.append(display_name)
            return {"success": True, "chunks": len(chunks), "pages": len(pages)}
        except Exception as e:
            return {"success": False, "error": str(e)}

    def answer(self, question: str, top_k: int = 4) -> dict:
        store = self._get_store()
        if len(store) == 0:
            return {"answer": "No documents indexed yet. Please upload a PDF first.", "sources": []}

        # 1. Embed question
        q_vec = self._embedder.encode([question])[0]
        # 2. Retrieve
        results = store.search(q_vec, k=top_k)
        # 3. Generate
        answer  = self._generator.generate(question, results)
        # 4. Collect sources
        sources = list({f"{rc.chunk.source} (p.{rc.chunk.page})" for rc in results})

        return {"answer": answer, "sources": sources, "retrieved": len(results)}

    def total_chunks(self) -> int:
        return len(self._get_store()) if self._store else 0

    def clear(self):
        if self._store:
            self._store.clear()
        self._doc_names = []