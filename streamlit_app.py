"""
streamlit_app.py — Streamlit frontend for the Petroleum Engineering RAG chatbot.

Deploy on https://share.streamlit.io

Usage:
    streamlit run streamlit_app.py
"""

import sys
import os
import json
import logging
from pathlib import Path

import streamlit as st
import numpy as np
import pandas as pd
import faiss

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT / "implementation"))

from pipeline_utils import (
    ensure_nltk_resources,
    clean_page_text,
    chunk_text,
    extract_chapter_titles,
    find_chapter_start_pages,
    assign_chapter_labels,
    build_context_package,
    generate_answer_hf,
)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

EMBEDDING_MODEL_NAME = "BAAI/bge-base-en-v1.5"
HF_MODEL = os.environ.get("GROQ_MODEL", "qwen/qwen3-32b")
HF_TOKEN = os.environ.get("GROQ_API_KEY")

PDF_PATH = PROJECT_ROOT / "geokniga-drillingengineeringprasslwl.pdf"
CACHE_DIR = PROJECT_ROOT / "implementation" / ".cache"
CHUNKS_CACHE = CACHE_DIR / "chunks.pkl"
FAISS_CACHE = CACHE_DIR / "faiss_index.bin"
METADATA_CACHE = CACHE_DIR / "metadata.json"

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("petro_rag_streamlit")


# ---------------------------------------------------------------------------
# Pipeline functions (adapted from app.py for Streamlit)
# ---------------------------------------------------------------------------

def run_pipeline():
    """Build chunks from the PDF."""
    import fitz

    if not PDF_PATH.exists():
        raise FileNotFoundError(f"PDF not found: {PDF_PATH}")

    pdf_document = fitz.open(str(PDF_PATH))

    raw_pages = []
    for page in pdf_document:
        raw_pages.append(page.get_text())

    chapter_titles = extract_chapter_titles(pdf_document)
    chapter_start_pages = find_chapter_start_pages(pdf_document)
    chapter_labels = assign_chapter_labels(pdf_document.page_count, chapter_start_pages, chapter_titles)

    pages_df = pd.DataFrame({
        "page_number": range(pdf_document.page_count),
        "raw_text": raw_pages,
        "chapter": chapter_labels,
    })
    pages_df["word_count"] = pages_df["raw_text"].apply(lambda t: len(t.split()))
    pages_df["char_count"] = pages_df["raw_text"].apply(len)
    pages_df["clean_text"] = pages_df["raw_text"].apply(clean_page_text)
    pages_df["clean_word_count"] = pages_df["clean_text"].apply(lambda t: len(t.split()))

    pages_df = pages_df[pages_df["chapter"] != "Front Matter"].reset_index(drop=True)
    pages_df = pages_df[pages_df["clean_word_count"] >= 40].reset_index(drop=True)

    chunk_rows = []
    for _, row in pages_df.iterrows():
        page_chunks = chunk_text(row["clean_text"], chunk_size=100, overlap=25)
        for i, chunk in enumerate(page_chunks):
            chunk_rows.append({
                "chunk_id": f"page{row['page_number']}_chunk{i}",
                "page_number": row["page_number"],
                "chapter": row["chapter"],
                "chunk_index": i,
                "chunk_text": chunk,
                "search_text": f"{row['chapter']} : {chunk}",
            })
    chunks_df = pd.DataFrame(chunk_rows)

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    chunks_df.to_pickle(str(CHUNKS_CACHE))

    metadata = {
        "pdf_mtime": PDF_PATH.stat().st_mtime,
        "num_pages": len(pages_df),
        "num_chunks": len(chunks_df),
    }
    with open(METADATA_CACHE, "w") as f:
        json.dump(metadata, f)

    pdf_document.close()
    return chunks_df


def load_or_build_chunks():
    """Load cached chunks or build them from PDF."""
    if CHUNKS_CACHE.exists() and METADATA_CACHE.exists():
        with open(METADATA_CACHE) as f:
            metadata = json.load(f)
        if metadata.get("pdf_mtime") == PDF_PATH.stat().st_mtime:
            return pd.read_pickle(str(CHUNKS_CACHE))
    return run_pipeline()


def load_or_build_faiss(embedding_model, chunks_df):
    """Load cached FAISS index or build it."""
    if FAISS_CACHE.exists() and METADATA_CACHE.exists():
        with open(METADATA_CACHE) as f:
            metadata = json.load(f)
        if metadata.get("pdf_mtime") == PDF_PATH.stat().st_mtime:
            return faiss.read_index(str(FAISS_CACHE))

    chunk_embeddings = embedding_model.encode(
        chunks_df["search_text"].tolist(),
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=True,
    )
    embedding_dimension = chunk_embeddings.shape[1]
    faiss_index = faiss.IndexFlatIP(embedding_dimension)
    faiss_index.add(chunk_embeddings.astype("float32"))

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    faiss.write_index(faiss_index, str(FAISS_CACHE))
    return faiss_index


# ---------------------------------------------------------------------------
# Cached resource loading
# ---------------------------------------------------------------------------

@st.cache_resource
def load_embedding_model():
    """Load the sentence-transformers model (cached across reruns)."""
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(EMBEDDING_MODEL_NAME)


@st.cache_resource
def load_chunks_and_index():
    """Load chunks and FAISS index (cached across reruns)."""
    ensure_nltk_resources()
    chunks_df = load_or_build_chunks()
    embedding_model = load_embedding_model()
    faiss_index = load_or_build_faiss(embedding_model, chunks_df)
    return chunks_df, faiss_index


# ---------------------------------------------------------------------------
# Custom CSS — Petroleum Engineering RAG Theme
# ---------------------------------------------------------------------------

def inject_custom_css():
    """Inject custom CSS for a polished petroleum-themed UI."""
    st.markdown("""
    <style>
    /* ── Global ─────────────────────────────────────────────────── */
    .stApp {
        background: linear-gradient(180deg, #0C1220 0%, #111927 100%);
    }

    /* ── Hero Banner ────────────────────────────────────────────── */
    .hero-banner {
        background: linear-gradient(135deg, #0F1A2E 0%, #1A2744 40%, #243352 100%);
        border: 1px solid rgba(212, 168, 67, 0.25);
        border-radius: 16px;
        padding: 2rem 2.5rem;
        margin-bottom: 1.5rem;
        position: relative;
        overflow: hidden;
    }
    .hero-banner::before {
        content: '';
        position: absolute;
        top: -50%;
        right: -20%;
        width: 400px;
        height: 400px;
        background: radial-gradient(circle, rgba(212, 168, 67, 0.08) 0%, transparent 70%);
        pointer-events: none;
    }
    .hero-title {
        font-size: 2rem;
        font-weight: 700;
        color: #D4A843;
        margin: 0 0 0.3rem 0;
        letter-spacing: -0.02em;
    }
    .hero-subtitle {
        font-size: 1rem;
        color: #9CA3AF;
        margin: 0;
        font-weight: 400;
    }
    .hero-badge {
        display: inline-block;
        background: rgba(212, 168, 67, 0.15);
        border: 1px solid rgba(212, 168, 67, 0.3);
        color: #D4A843;
        font-size: 0.7rem;
        font-weight: 600;
        padding: 0.2rem 0.6rem;
        border-radius: 20px;
        margin-bottom: 0.8rem;
        letter-spacing: 0.08em;
        text-transform: uppercase;
    }

    /* ── Sidebar ────────────────────────────────────────────────── */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0F1A2E 0%, #131B2E 100%);
        border-right: 1px solid rgba(212, 168, 67, 0.12);
    }
    [data-testid="stSidebar"] [data-testid="stMarkdown"] h1,
    [data-testid="stSidebar"] [data-testid="stMarkdown"] h2,
    [data-testid="stSidebar"] [data-testid="stMarkdown"] h3 {
        color: #D4A843 !important;
    }

    /* ── Chat Messages ──────────────────────────────────────────── */
    [data-testid="stChatMessage"] {
        border-radius: 12px;
        padding: 1rem 1.2rem;
        margin-bottom: 0.5rem;
        border: 1px solid rgba(255, 255, 255, 0.06);
    }
    [data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-user"]) {
        background: rgba(212, 168, 67, 0.06);
        border-color: rgba(212, 168, 67, 0.12);
    }
    [data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-assistant"]) {
        background: rgba(255, 255, 255, 0.03);
        border-color: rgba(255, 255, 255, 0.08);
    }

    /* ── Chat Input ─────────────────────────────────────────────── */
    [data-testid="stChatInput"] {
        border-radius: 12px;
        border: 1px solid rgba(212, 168, 67, 0.2);
        background: rgba(255, 255, 255, 0.04);
    }
    [data-testid="stChatInput"]:focus-within {
        border-color: rgba(212, 168, 67, 0.5);
        box-shadow: 0 0 0 2px rgba(212, 168, 67, 0.1);
    }

    /* ── Expanders ──────────────────────────────────────────────── */
    [data-testid="stExpander"] {
        background: rgba(255, 255, 255, 0.02);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 10px;
    }

    /* ── Status Cards (sidebar) ─────────────────────────────────── */
    .status-card {
        background: rgba(255, 255, 255, 0.03);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 10px;
        padding: 0.8rem 1rem;
        margin-bottom: 0.6rem;
    }
    .status-card .label {
        font-size: 0.7rem;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        color: #6B7280;
        margin-bottom: 0.2rem;
    }
    .status-card .value {
        font-size: 0.9rem;
        color: #D4A843;
        font-weight: 600;
        font-family: 'Courier New', monospace;
    }

    /* ── Divider ────────────────────────────────────────────────── */
    .gold-divider {
        border: none;
        height: 1px;
        background: linear-gradient(90deg, transparent, rgba(212, 168, 67, 0.3), transparent);
        margin: 1rem 0;
    }

    /* ── Footer ─────────────────────────────────────────────────── */
    .footer {
        text-align: center;
        color: #4B5563;
        font-size: 0.75rem;
        padding: 1.5rem 0 0.5rem 0;
        border-top: 1px solid rgba(255, 255, 255, 0.04);
        margin-top: 2rem;
    }

    /* ── How It Works Steps ─────────────────────────────────────── */
    .step-item {
        display: flex;
        align-items: flex-start;
        gap: 0.7rem;
        margin-bottom: 0.6rem;
    }
    .step-num {
        background: rgba(212, 168, 67, 0.15);
        color: #D4A843;
        font-size: 0.7rem;
        font-weight: 700;
        width: 22px;
        height: 22px;
        border-radius: 50%;
        display: flex;
        align-items: center;
        justify-content: center;
        flex-shrink: 0;
        margin-top: 2px;
    }
    .step-text {
        color: #9CA3AF;
        font-size: 0.82rem;
        line-height: 1.4;
    }

    /* ── Example Questions ──────────────────────────────────────── */
    .example-q {
        background: rgba(212, 168, 67, 0.06);
        border: 1px solid rgba(212, 168, 67, 0.12);
        border-radius: 8px;
        padding: 0.5rem 0.8rem;
        margin-bottom: 0.4rem;
        color: #D1D5DB;
        font-size: 0.82rem;
        cursor: default;
    }
    </style>
    """, unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Streamlit UI
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="Petroleum RAG — Drilling Engineering AI",
    page_icon="🛢️",
    layout="wide",
    initial_sidebar_state="expanded",
)

inject_custom_css()

# ── Hero Banner ─────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero-banner">
    <div class="hero-badge">🛢️ Retrieval-Augmented Generation</div>
    <div class="hero-title">Petroleum Engineering RAG</div>
    <p class="hero-subtitle">
        AI-powered drilling engineering assistant — answers grounded in textbook evidence
    </p>
</div>
""", unsafe_allow_html=True)

# ── Sidebar ─────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### ⚙️ System Status")
    st.markdown("")

    if not HF_TOKEN:
        st.warning("**GROQ_API_KEY** not set. Add it as a secret for the LLM to work.")
    else:
        st.success("**GROQ_API_KEY** configured")

    st.markdown('<hr class="gold-divider">', unsafe_allow_html=True)

    st.markdown("### 🧠 Models")
    st.markdown(f"""
    <div class="status-card">
        <div class="label">Embedding Model</div>
        <div class="value">{EMBEDDING_MODEL_NAME}</div>
    </div>
    <div class="status-card">
        <div class="label">LLM Model</div>
        <div class="value">{HF_MODEL}</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<hr class="gold-divider">', unsafe_allow_html=True)

    st.markdown("### 📖 How It Works")
    st.markdown("""
    <div class="step-item"><div class="step-num">1</div><div class="step-text">You ask a question about drilling engineering</div></div>
    <div class="step-item"><div class="step-num">2</div><div class="step-text">FAISS retrieves the most relevant textbook chunks</div></div>
    <div class="step-item"><div class="step-num">3</div><div class="step-text">LLM generates an answer grounded in the evidence</div></div>
    <div class="step-item"><div class="step-num">4</div><div class="step-text">Sources and chunks are shown for transparency</div></div>
    """, unsafe_allow_html=True)

    st.markdown('<hr class="gold-divider">', unsafe_allow_html=True)

    st.markdown("### 💡 Try Asking")
    st.markdown("""
    <div class="example-q">"What is drill string design?"</div>
    <div class="example-q">"How does rotary drilling work?"</div>
    <div class="example-q">"Explain well control methods"</div>
    <div class="example-q">"What are the types of drill bits?"</div>
    """, unsafe_allow_html=True)

    st.markdown('<hr class="gold-divider">', unsafe_allow_html=True)

    st.markdown("""
    <div style="color: #4B5563; font-size: 0.75rem; line-height: 1.5;">
        <strong style="color: #6B7280;">Tech Stack</strong><br>
        Streamlit · FAISS · Sentence-Transformers · Groq Inference API
    </div>
    """, unsafe_allow_html=True)

# ── Load resources ──────────────────────────────────────────────────────────
try:
    with st.spinner("🔄 Loading embedding model and building index... (first run may take a few minutes)"):
        chunks_df, faiss_index = load_chunks_and_index()
    st.sidebar.markdown(f"""
    <div class="status-card">
        <div class="label">Index Status</div>
        <div class="value">✅ {len(chunks_df):,} chunks · {faiss_index.ntotal:,} vectors</div>
    </div>
    """, unsafe_allow_html=True)
except Exception as e:
    st.error(f"❌ Failed to load resources: {e}")
    st.stop()

# ── Chat History ────────────────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []

# Welcome message on first load
if not st.session_state.messages:
    st.markdown("""
    <div style="text-align: center; padding: 2rem 1rem; color: #6B7280;">
        <div style="font-size: 2.5rem; margin-bottom: 0.5rem;">🛢️</div>
        <div style="font-size: 1.1rem; color: #D4A843; font-weight: 600; margin-bottom: 0.3rem;">
            Welcome to Petroleum RAG
        </div>
        <div style="font-size: 0.9rem; max-width: 480px; margin: 0 auto; line-height: 1.6;">
            Ask any question about drilling engineering. Answers are generated from a petroleum
            engineering textbook using retrieval-augmented generation for accuracy.
        </div>
    </div>
    """, unsafe_allow_html=True)

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message.get("sources"):
            with st.expander("📚 Sources"):
                for src in message["sources"]:
                    st.markdown(f"- **Page {src['page']}** — {src['chapter']}")
        if message.get("chunks"):
            with st.expander("🔍 Retrieved Chunks"):
                for chunk in message["chunks"]:
                    st.markdown(
                        f"**{chunk['chunk_id']}** (Page {chunk['page']}, Score: {chunk['score']:.4f})\n"
                        f"> {chunk['text']}"
                    )

# ── Chat Input ──────────────────────────────────────────────────────────────
if prompt := st.chat_input("Ask a petroleum engineering question..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        if not HF_TOKEN:
            st.error(
                "❌ **GROQ_API_KEY not set.** Please add your Groq API key as a secret in "
                "Streamlit Cloud settings → Secrets.\n\n"
                "Get a free key at: https://console.groq.com/keys"
            )
            st.session_state.messages.append({
                "role": "assistant",
                "content": "❌ GROQ_API_KEY not configured. Please add it in Streamlit Cloud secrets.",
            })
        else:
            with st.spinner("🔍 Searching textbook and generating answer..."):
                try:
                    embedding_model = load_embedding_model()

                    package = build_context_package(
                        prompt,
                        embedding_model,
                        chunks_df,
                        faiss_index,
                    )

                    answer = generate_answer_hf(
                        prompt,
                        package["context_text"],
                        model_name=HF_MODEL,
                        hf_token=HF_TOKEN,
                    )

                    st.markdown(answer)

                    sources = []
                    if len(package["selected"]) > 0:
                        for _, row in package["selected"].iterrows():
                            sources.append({
                                "page": int(row["page_number"]),
                                "chapter": row["chapter"],
                                "chunk_id": row["chunk_id"],
                            })

                    if sources:
                        with st.expander("📚 Sources"):
                            for src in sources:
                                st.markdown(f"- **Page {src['page']}** — {src['chapter']}")

                    retrieved_chunks = []
                    if len(package["candidates"]) > 0:
                        for _, row in package["candidates"].head(8).iterrows():
                            retrieved_chunks.append({
                                "chunk_id": row["chunk_id"],
                                "page": int(row["page_number"]),
                                "chapter": row["chapter"],
                                "score": round(float(row["score"]), 4),
                                "text": row["chunk_text"][:300] + ("..." if len(row["chunk_text"]) > 300 else ""),
                            })

                    if retrieved_chunks:
                        with st.expander("🔍 Retrieved Chunks"):
                            for chunk in retrieved_chunks:
                                st.markdown(
                                    f"**{chunk['chunk_id']}** (Page {chunk['page']}, Score: {chunk['score']:.4f})\n"
                                    f"> {chunk['text']}"
                                )

                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": answer,
                        "sources": sources,
                        "chunks": retrieved_chunks,
                    })

                except Exception as e:
                    error_msg = f"❌ Error generating answer: {e}"
                    st.error(error_msg)
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": error_msg,
                    })

# ── Footer ──────────────────────────────────────────────────────────────────
st.markdown("""
<div class="footer">
    Petroleum Engineering RAG · Drilling Engineering Knowledge Base<br>
    Built with Streamlit · FAISS · Sentence-Transformers · Groq
</div>
""", unsafe_allow_html=True)
