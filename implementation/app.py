"""
app.py — FastAPI server for the Petroleum Engineering RAG chat application.

Supports two LLM backends:
  - Ollama + DeepSeek (local development)
  - Hugging Face Inference API (cloud deployment on HF Spaces)

Set LLM_BACKEND=ollama or LLM_BACKEND=hf (default: auto-detect).

Usage:
    python implementation/app.py
"""

import sys
import json
import pickle
import logging
import os
from pathlib import Path
from contextlib import asynccontextmanager

import numpy as np
import pandas as pd
import faiss
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------

THIS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = THIS_DIR.parent
PDF_PATH = PROJECT_ROOT / "geokniga-drillingengineeringprasslwl.pdf"
CACHE_DIR = THIS_DIR / ".cache"
CHUNKS_CACHE = CACHE_DIR / "chunks.pkl"
FAISS_CACHE = CACHE_DIR / "faiss_index.bin"
METADATA_CACHE = CACHE_DIR / "metadata.json"

sys.path.insert(0, str(THIS_DIR))

from pipeline_utils import (
    ensure_nltk_resources,
    clean_page_text,
    preprocess_text,
    chunk_text,
    extract_chapter_titles,
    find_chapter_start_pages,
    assign_chapter_labels,
    build_context_package,
    build_better_prompt,
    generate_answer,
    generate_answer_hf,
)

from sentence_transformers import SentenceTransformer

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

EMBEDDING_MODEL_NAME = "BAAI/bge-base-en-v1.5"
OLLAMA_MODEL = "deepseek-r1:1.5b"
HF_MODEL = "HuggingFaceH4/zephyr-7b-beta"
HF_TOKEN = os.environ.get("HF_TOKEN", os.environ.get("HUGGING_FACE_HUB_TOKEN"))
LLM_BACKEND = os.environ.get("LLM_BACKEND", "auto").lower()

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("petro_rag")

# ---------------------------------------------------------------------------
# Global state — populated at startup
# ---------------------------------------------------------------------------

state = {
    "chunks_df": None,
    "embedding_model": None,
    "faiss_index": None,
    "llm_ok": False,
    "llm_backend": "unknown",
}


# ---------------------------------------------------------------------------
# Pipeline execution — builds chunks, embeddings, FAISS from the PDF
# ---------------------------------------------------------------------------

def run_pipeline():
    """Execute the full RAG pipeline from the notebook: PDF -> clean -> chunk -> embed -> FAISS."""
    import fitz

    logger.info("Starting full pipeline execution...")

    if not PDF_PATH.exists():
        raise FileNotFoundError(f"PDF not found: {PDF_PATH}")
    logger.info(f"Loading PDF: {PDF_PATH}")
    pdf_document = fitz.open(str(PDF_PATH))

    raw_pages = []
    for page in pdf_document:
        raw_pages.append(page.get_text())
    logger.info(f"Extracted {len(raw_pages)} pages")

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
    logger.info(f"After filtering: {len(pages_df)} pages remain")

    logger.info("Building chunks...")
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
    logger.info(f"Created {len(chunks_df)} chunks")

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    chunks_df.to_pickle(str(CHUNKS_CACHE))

    metadata = {
        "pdf_mtime": PDF_PATH.stat().st_mtime,
        "num_pages": len(pages_df),
        "num_chunks": len(chunks_df),
    }
    with open(METADATA_CACHE, "w") as f:
        json.dump(metadata, f)
    logger.info("Pipeline artifacts cached to disk")

    pdf_document.close()
    return chunks_df


def load_or_build_chunks():
    """Load cached chunks or run the pipeline to build them."""
    if CHUNKS_CACHE.exists() and METADATA_CACHE.exists():
        with open(METADATA_CACHE) as f:
            metadata = json.load(f)
        if metadata.get("pdf_mtime") == PDF_PATH.stat().st_mtime:
            logger.info("Loading chunks from cache...")
            chunks_df = pd.read_pickle(str(CHUNKS_CACHE))
            logger.info(f"Loaded {len(chunks_df)} cached chunks")
            return chunks_df
        else:
            logger.info("Cache stale (PDF changed) — rebuilding...")
    else:
        logger.info("No cache found — running pipeline...")

    return run_pipeline()


def load_or_build_faiss(embedding_model, chunks_df):
    """Load cached FAISS index or build it from chunk embeddings."""
    if FAISS_CACHE.exists() and METADATA_CACHE.exists():
        with open(METADATA_CACHE) as f:
            metadata = json.load(f)
        if metadata.get("pdf_mtime") == PDF_PATH.stat().st_mtime:
            logger.info("Loading FAISS index from cache...")
            faiss_index = faiss.read_index(str(FAISS_CACHE))
            logger.info(f"Loaded FAISS index with {faiss_index.ntotal} vectors")
            return faiss_index
        else:
            logger.info("FAISS cache stale — rebuilding...")
    else:
        logger.info("No FAISS cache found — building index...")

    logger.info(f"Encoding {len(chunks_df)} chunks with embedding model...")
    chunk_embeddings = embedding_model.encode(
        chunks_df["search_text"].tolist(),
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=True,
    )

    embedding_dimension = chunk_embeddings.shape[1]
    faiss_index = faiss.IndexFlatIP(embedding_dimension)
    faiss_index.add(chunk_embeddings.astype("float32"))
    logger.info(f"Built FAISS index: {faiss_index.ntotal} vectors, {embedding_dimension}d")

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    faiss.write_index(faiss_index, str(FAISS_CACHE))
    logger.info("FAISS index saved to cache")

    return faiss_index


# ---------------------------------------------------------------------------
# LLM backend detection
# ---------------------------------------------------------------------------

def detect_backend():
    """Auto-detect which LLM backend to use."""
    global LLM_BACKEND
    if LLM_BACKEND != "auto":
        return LLM_BACKEND

    # Try Ollama first (local development)
    try:
        import ollama
        models = ollama.list()
        model_names = [m.get("name", "") or m.get("model", "") for m in models.get("models", [])]
        if any("deepseek" in name.lower() for name in model_names):
            LLM_BACKEND = "ollama"
            logger.info(f"Auto-detected: Ollama with '{OLLAMA_MODEL}'")
            return "ollama"
    except Exception:
        pass

    # Fall back to HF Inference API
    LLM_BACKEND = "hf"
    logger.info(f"Auto-detected: Hugging Face Inference API with '{HF_MODEL}'")
    return "hf"


def check_llm_backend():
    """Verify the chosen LLM backend is working."""
    backend = detect_backend()
    if backend == "ollama":
        try:
            import ollama
            models = ollama.list()
            model_names = [m.get("name", "") or m.get("model", "") for m in models.get("models", [])]
            has_model = any("deepseek" in name.lower() for name in model_names)
            if has_model:
                logger.info(f"Ollama OK — models available: {model_names}")
                return True
            else:
                logger.warning(f"Ollama running but deepseek model not found. Models: {model_names}")
                return False
        except Exception as e:
            logger.error(f"Ollama check failed: {e}")
            return False
    else:
        from pipeline_utils import InferenceClient
        if InferenceClient is None:
            logger.warning("huggingface_hub not installed — cannot use HF Inference API")
            return False
        logger.info(f"HF Inference API backend ready (model: {HF_MODEL})")
        return True


# ---------------------------------------------------------------------------
# FastAPI lifespan — startup / shutdown
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup: load all resources. Shutdown: nothing to clean up."""
    logger.info("=" * 60)
    logger.info("Petroleum Engineering RAG — Starting up...")
    logger.info("=" * 60)

    ensure_nltk_resources()

    state["chunks_df"] = load_or_build_chunks()

    logger.info(f"Loading embedding model ({EMBEDDING_MODEL_NAME})...")
    state["embedding_model"] = SentenceTransformer(EMBEDDING_MODEL_NAME)
    logger.info("Embedding model loaded")

    state["faiss_index"] = load_or_build_faiss(state["embedding_model"], state["chunks_df"])

    state["llm_ok"] = check_llm_backend()
    state["llm_backend"] = LLM_BACKEND

    logger.info("=" * 60)
    logger.info("Startup complete!")
    logger.info(f"  Chunks:       {len(state['chunks_df'])}")
    logger.info(f"  FAISS:        {state['faiss_index'].ntotal} vectors")
    logger.info(f"  LLM backend:  {LLM_BACKEND}")
    logger.info(f"  LLM status:   {'OK' if state['llm_ok'] else 'OFFLINE'}")
    logger.info("=" * 60)

    yield

    logger.info("Shutting down...")


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Petroleum Engineering RAG",
    description="Interactive chat for petroleum engineering questions using RAG",
    lifespan=lifespan,
)

app.mount("/static", StaticFiles(directory=str(THIS_DIR / "static")), name="static")
templates = Jinja2Templates(directory=str(THIS_DIR / "templates"))


# ---------------------------------------------------------------------------
# Request / Response models
# ---------------------------------------------------------------------------

class ChatRequest(BaseModel):
    question: str


class ChatResponse(BaseModel):
    answer: str
    sources: list
    retrieved_chunks: list


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    """Serve the chat interface."""
    return templates.TemplateResponse(name="index.html", request=request)


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """Process a petroleum engineering question through the RAG pipeline."""
    question = request.question.strip()

    if not question:
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    # Check LLM availability
    if not state["llm_ok"]:
        state["llm_ok"] = check_llm_backend()
        if not state["llm_ok"]:
            raise HTTPException(
                status_code=503,
                detail=(
                    "LLM backend not available. Please ensure either:\n"
                    "1. Ollama is running with deepseek-r1:1.5b pulled, OR\n"
                    "2. HF_TOKEN environment variable is set for Hugging Face Inference API"
                ),
            )

    try:
        package = build_context_package(
            question,
            state["embedding_model"],
            state["chunks_df"],
            state["faiss_index"],
        )

        # Generate answer using the configured backend
        if LLM_BACKEND == "ollama":
            answer = generate_answer(question, package["context_text"])
        else:
            answer = generate_answer_hf(
                question, package["context_text"],
                model_name=HF_MODEL, hf_token=HF_TOKEN,
            )

        sources = []
        if len(package["selected"]) > 0:
            for _, row in package["selected"].iterrows():
                sources.append({
                    "page": int(row["page_number"]),
                    "chapter": row["chapter"],
                    "chunk_id": row["chunk_id"],
                })

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

        return ChatResponse(
            answer=answer,
            sources=sources,
            retrieved_chunks=retrieved_chunks,
        )

    except Exception as e:
        logger.exception("Error processing question")
        raise HTTPException(status_code=500, detail=f"Pipeline error: {str(e)}")


@app.get("/health")
async def health():
    """Health check endpoint."""
    return {
        "status": "ok" if state.get("llm_ok") else "degraded",
        "llm_backend": state.get("llm_backend", "unknown"),
        "chunks_loaded": state["chunks_df"] is not None,
        "faiss_loaded": state["faiss_index"] is not None,
        "embedding_model_loaded": state["embedding_model"] is not None,
        "llm_available": state["llm_ok"],
        "num_chunks": len(state["chunks_df"]) if state["chunks_df"] is not None else 0,
    }


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
