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
HF_MODEL = os.environ.get("HF_MODEL", "HuggingFaceH4/zephyr-7b-beta")
HF_TOKEN = os.environ.get("HF_TOKEN", os.environ.get("HUGGING_FACE_HUB_TOKEN"))

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
# Streamlit UI
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="Petroleum Engineering RAG",
    page_icon="🛢️",
    layout="wide",
)

st.title("🛢️ Petroleum Engineering RAG Chatbot")
st.markdown(
    "Ask questions about drilling engineering and get answers grounded in a petroleum engineering textbook."
)

# Sidebar with info
with st.sidebar:
    st.header("About")
    st.markdown(
        "This chatbot uses **Retrieval-Augmented Generation (RAG)** to answer "
        "petroleum engineering questions using a drilling engineering textbook as evidence."
    )
    st.markdown("---")
    st.markdown(f"**Embedding model:** `{EMBEDDING_MODEL_NAME}`")
    st.markdown(f"**LLM model:** `{HF_MODEL}`")
    st.markdown("---")

    if not HF_TOKEN:
        st.warning(
            "⚠️ **HF_TOKEN not set.** Add it as a secret in your Streamlit Cloud settings "
            "for the LLM to work."
        )
    else:
        st.success("✅ HF_TOKEN is configured")

    st.markdown("---")
    st.markdown("Built with [Streamlit](https://streamlit.io) + [FAISS](https://faiss.ai/)")

# Load resources
try:
    with st.spinner("Loading embedding model and building index... (first run may take a few minutes)"):
        chunks_df, faiss_index = load_chunks_and_index()
    st.sidebar.success(f"✅ {len(chunks_df)} chunks loaded, {faiss_index.ntotal} FAISS vectors")
except Exception as e:
    st.error(f"❌ Failed to load resources: {e}")
    st.stop()

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
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

# Chat input
if prompt := st.chat_input("Ask a petroleum engineering question..."):
    # Add user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Generate answer
    with st.chat_message("assistant"):
        if not HF_TOKEN:
            st.error(
                "❌ **HF_TOKEN not set.** Please add your Hugging Face token as a secret in "
                "Streamlit Cloud settings → Secrets.\n\n"
                "Get a token at: https://huggingface.co/settings/tokens"
            )
            st.session_state.messages.append({
                "role": "assistant",
                "content": "❌ HF_TOKEN not configured. Please add it in Streamlit Cloud secrets.",
            })
        else:
            with st.spinner("Searching textbook and generating answer..."):
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

                    # Sources
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

                    # Retrieved chunks
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

                    # Save to history
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
