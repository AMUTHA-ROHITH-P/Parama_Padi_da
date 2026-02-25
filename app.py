import tempfile

import streamlit as st
import os
import tempfile
from pathlib import Path
from rag_backend import (
    RAGBackend, OLLAMA_MODELS, OPENAI_MODELS, CLAUDE_MODELS,
    ollama_is_running, ollama_list_models,
)

st.set_page_config(page_title="EduRAG – AI Study Assistant", page_icon="📚",
                   layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<style>
.stApp { background-color: #f0f4f8; }
[data-testid="stSidebar"] { background-color: #1e3a5f; }
[data-testid="stSidebar"] * { color: white !important; }
[data-testid="stSidebar"] .stButton > button {
    background-color: #2ecc71; color: white !important; border: none;
    border-radius: 8px; width: 100%; font-weight: bold; }
.chat-user {
    background: #1e3a5f; color: white; border-radius: 18px 18px 4px 18px;
    padding: 0.8rem 1.2rem; margin: 0.5rem 0 0.5rem 20%; }
.chat-ai {
    background: white; color: #1a1a2e; border-radius: 18px 18px 18px 4px;
    padding: 0.8rem 1.2rem; margin: 0.5rem 20% 0.5rem 0;
    box-shadow: 0 2px 6px rgba(0,0,0,0.08); }
.chat-label { font-size: 0.75rem; color: #888; margin-bottom: 0.2rem; }
.source-badge {
    display: inline-block; background: #e8f4fd; color: #1e3a5f;
    border: 1px solid #bee3f8; border-radius: 20px;
    padding: 0.2rem 0.8rem; font-size: 0.8rem; margin: 0.2rem; }
.stat-box {
    background: linear-gradient(135deg, #1e3a5f, #2980b9);
    color: white; border-radius: 10px; padding: 1rem; text-align: center; }
.stat-number { font-size: 2rem; font-weight: bold; }
.stat-label  { font-size: 0.85rem; opacity: 0.85; }
.hero {
    background: linear-gradient(135deg, #1e3a5f 0%, #2980b9 100%);
    color: white; border-radius: 12px; padding: 2rem; margin-bottom: 1.5rem;
    text-align: center; }
.hero h1 { margin: 0; font-size: 2.2rem; }
.hero p  { margin: 0.5rem 0 0; opacity: 0.9; font-size: 1.05rem; }
</style>
""", unsafe_allow_html=True)

if "rag"           not in st.session_state: st.session_state.rag = RAGBackend()
if "chat_history"  not in st.session_state: st.session_state.chat_history = []
if "uploaded_docs" not in st.session_state: st.session_state.uploaded_docs = []

rag: RAGBackend = st.session_state.rag

with st.sidebar:
    st.markdown("## 📚 EduRAG")
    st.markdown("*Your AI-powered study companion*")
    st.divider()

    st.markdown("### 🤖 AI Model")

    ollama_ok     = ollama_is_running()
    ollama_models = ollama_list_models() if ollama_ok else []
    has_openai    = bool(os.getenv("OPENAI_API_KEY"))
    has_anthropic = bool(os.getenv("ANTHROPIC_API_KEY"))

    if ollama_ok:
        st.success(f"● Ollama running ({len(ollama_models)} models)")
    else:
        st.warning("○ Ollama not running")
        st.caption("Get free local AI → https://ollama.com")

    backend_options = []
    if ollama_ok:     backend_options.append("🦙 Llama / Ollama (local, FREE)")
    if has_openai:    backend_options.append("🟢 OpenAI GPT")
    if has_anthropic: backend_options.append("🟣 Anthropic Claude")
    backend_options.append("📄 Extractive (no AI)")

    backend_choice = st.selectbox("Choose AI backend", backend_options, index=0)
    selected_model = ""

    if "Llama" in backend_choice or "Ollama" in backend_choice:
        backend_key    = "ollama"
        model_list     = ollama_models if ollama_models else ["llama3.2"]
        selected_model = st.selectbox("Ollama model", model_list)
    elif "OpenAI" in backend_choice:
        backend_key    = "openai"
        selected_model = st.selectbox("OpenAI model", OPENAI_MODELS)
    elif "Anthropic" in backend_choice:
        backend_key    = "anthropic"
        selected_model = st.selectbox("Claude model", CLAUDE_MODELS)
    else:
        backend_key = "extractive"

    if st.button("Apply Model"):
        rag.set_backend(backend_key, selected_model)
        st.success("Model updated!")

    st.caption(f"Active backend: **{rag.active_backend()}**")

    if not has_openai:
        with st.expander("➕ Add OpenAI key (optional)"):
            oai = st.text_input("OpenAI API key", type="password", placeholder="sk-...")
            if oai:
                os.environ["OPENAI_API_KEY"] = oai
                st.success("Saved!")
                st.rerun()

    if not has_anthropic:
        with st.expander("➕ Add Anthropic key (optional)"):
            ant = st.text_input("Anthropic API key", type="password", placeholder="sk-ant-...")
            if ant:
                os.environ["ANTHROPIC_API_KEY"] = ant
                st.success("Saved!")
                st.rerun()

    st.divider()
    st.markdown("### 📂 Upload Study Materials")
    uploaded_files = st.file_uploader("Upload PDF(s)", type=["pdf"], accept_multiple_files=True)


if uploaded_files:
    for uf in uploaded_files:
        if uf.name not in st.session_state.uploaded_docs:
            with st.spinner(f"Processing {uf.name}…"):
                # Create a safe temporary file
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                    tmp_path = Path(tmp.name)
                    tmp.write(uf.read())  # write uploaded file bytes
                
                # Ingest PDF into RAG
                result = rag.ingest_pdf(str(tmp_path), uf.name)

            if result["success"]:
                st.session_state.uploaded_docs.append(uf.name)
                st.success(f"✅ {uf.name} — {result['chunks']} chunks")
            else:
                st.error(f"❌ {result['error']}")

    st.divider()
    st.markdown("### 📑 Indexed Documents")
    for doc in st.session_state.uploaded_docs:
        st.markdown(f"📄 {doc}")
    if not st.session_state.uploaded_docs:
        st.info("No documents yet.")

    st.divider()
    top_k    = st.slider("Retrieved chunks (top-k)", 2, 8, 4)
    show_src = st.toggle("Show source excerpts", value=True)

    if st.button("🗑️ Clear All & Start Over"):
        rag.clear()
        st.session_state.chat_history  = []
        st.session_state.uploaded_docs = []
        st.rerun()

st.markdown("""
<div class="hero">
  <h1>📚 EduRAG – AI Study Assistant</h1>
  <p>Upload your study materials and ask anything. Powered by Llama, GPT, Claude — or no API at all.</p>
</div>
""", unsafe_allow_html=True)

c1, c2, c3 = st.columns(3)
for col, num, label in [
    (c1, len(st.session_state.uploaded_docs), "Documents"),
    (c2, rag.total_chunks(),                  "Knowledge Chunks"),
    (c3, len(st.session_state.chat_history),  "Q&As"),
]:
    col.markdown(f'<div class="stat-box"><div class="stat-number">{num}</div>'
                 f'<div class="stat-label">{label}</div></div>', unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)
st.markdown("### 💬 Ask a Question")

for turn in st.session_state.chat_history:
    st.markdown(f'<div class="chat-label">You</div>'
                f'<div class="chat-user">{turn["question"]}</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="chat-label">EduRAG ({turn.get("backend","?")})</div>'
                f'<div class="chat-ai">{turn["answer"]}</div>', unsafe_allow_html=True)
    if show_src and turn.get("sources"):
        badges = "".join(f'<span class="source-badge">📄 {s}</span>' for s in turn["sources"])
        st.markdown(badges, unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)

with st.form("q_form", clear_on_submit=True):
    question = st.text_input("Your question",
        placeholder="e.g. What is photosynthesis? Explain Newton's laws…",
        label_visibility="collapsed")
    submit = st.form_submit_button("Ask ➤", use_container_width=True)

if submit and question.strip():
    if rag.total_chunks() == 0:
        st.warning("Please upload at least one PDF first.")
    else:
        with st.spinner(f"Thinking with {rag.active_backend()}…"):
            result = rag.answer(question.strip(), top_k=top_k)
        st.session_state.chat_history.append({
            "question": question.strip(),
            "answer":   result["answer"],
            "sources":  result.get("sources", []),
            "backend":  rag.active_backend(),
        })
        st.rerun()

if rag.total_chunks() > 0 and not st.session_state.chat_history:
    st.markdown("#### 💡 Try asking…")
    suggestions = [
        "Summarise the main topics in this document.",
        "What are the key definitions introduced?",
        "Explain the most important concept simply.",
        "Give me 5 quiz questions based on this material.",
    ]
    cols = st.columns(2)
    for i, s in enumerate(suggestions):
        if cols[i % 2].button(s, key=f"sug_{i}"):
            with st.spinner("Thinking…"):
                result = rag.answer(s, top_k=top_k)
            st.session_state.chat_history.append({
                "question": s, "answer": result["answer"],
                "sources": result.get("sources", []), "backend": rag.active_backend(),
            })
            st.rerun()