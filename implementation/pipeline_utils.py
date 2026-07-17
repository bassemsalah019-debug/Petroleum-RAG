"""
pipeline_utils.py — Reusable functions extracted from Petroleum_RAG_Project.ipynb

This module contains all the core pipeline functions from the notebook.
No logic has been rewritten — these are the exact functions used in the RAG pipeline.
"""

import re
import string
import numpy as np
import pandas as pd
import fitz  # PyMuPDF
from pathlib import Path

import nltk
# Download NLTK data at import time (needed for Streamlit Cloud where data isn't pre-installed)
for _nltk_res in ["punkt", "punkt_tab", "stopwords", "wordnet", "omw-1.4"]:
    try:
        nltk.data.find(f"tokenizers/{_nltk_res}" if "punkt" in _nltk_res else f"corpora/{_nltk_res}")
    except LookupError:
        nltk.download(_nltk_res, quiet=True)

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from sentence_transformers import SentenceTransformer
import faiss

try:
    import ollama
except ImportError:
    ollama = None

# Optional: Hugging Face Inference API (used when running on HF Spaces)
try:
    from huggingface_hub import InferenceClient
except ImportError:
    InferenceClient = None


# ---------------------------------------------------------------------------
# Constants (from notebook Cells 21, 23, 32, 40)
# ---------------------------------------------------------------------------

HEADER_LINES = {
    "curtin university of technology",
    "department of petroleum engineering",
    "master of petroleum engineering",
    "drilling engineering",
}

LIGATURE_MAP = {
    "ﬁ": "fi",
    "ﬂ": "fl",
    "ﬀ": "ff",
    "ﬃ": "ffi",
    "ﬄ": "ffl",
}

translator = str.maketrans("", "", string.punctuation)

stop_words = set(stopwords.words("english"))
protected_negation_words = {"no", "not", "nor", "never"}

lemmatizer = WordNetLemmatizer()


# ---------------------------------------------------------------------------
# NLTK resource check
# ---------------------------------------------------------------------------

def ensure_nltk_resources():
    """Download NLTK resources if not already present."""
    resources = ["punkt", "punkt_tab", "stopwords", "wordnet", "omw-1.4"]
    for resource in resources:
        try:
            nltk.data.find(f"tokenizers/{resource}" if "punkt" in resource
                           else f"corpora/{resource}")
        except LookupError:
            nltk.download(resource, quiet=True)


# ---------------------------------------------------------------------------
# Chapter extraction (from notebook Cell 12)
# ---------------------------------------------------------------------------

def extract_chapter_titles(document, toc_page_range=range(0, 8)):
    """
    Scan the table-of-contents pages for lines shaped like '<number> <Title>'
    (e.g. '3 Geomechanics') and return a dict mapping chapter number -> title.
    """
    toc_text = ""
    for i in toc_page_range:
        toc_text += document[i].get_text() + "\n"

    chapter_pattern = re.compile(r"^(\d{1,2})\s+([A-Z][A-Za-z ,]+)$", re.MULTILINE)
    chapter_titles = {}
    for match in chapter_pattern.finditer(toc_text):
        number, title = int(match.group(1)), match.group(2).strip()
        if number not in chapter_titles:
            chapter_titles[number] = title
    return chapter_titles


def find_chapter_start_pages(document):
    """
    Scan every page for a standalone line reading exactly 'Chapter <N>',
    which marks the first page of chapter N. Returns dict: chapter_number -> page_index.
    """
    chapter_starts = {}
    for page_index in range(document.page_count):
        lines = [line.strip() for line in document[page_index].get_text().split("\n") if line.strip()]
        for line in lines:
            match = re.match(r"^Chapter\s+(\d{1,2})$", line)
            if match:
                number = int(match.group(1))
                if number not in chapter_starts:
                    chapter_starts[number] = page_index
    return chapter_starts


def assign_chapter_labels(num_pages, chapter_start_pages, chapter_titles):
    """
    Build a page-index -> 'Chapter N: Title' label for every page in the document,
    by forward-filling each page with the most recent chapter boundary it has passed.
    """
    boundaries = sorted(chapter_start_pages.items(), key=lambda item: item[1])
    labels = []
    current_label = "Front Matter"

    boundary_index = 0
    for page_index in range(num_pages):
        while (boundary_index < len(boundaries)) and (page_index == boundaries[boundary_index][1]):
            chapter_num = boundaries[boundary_index][0]
            current_label = f"Chapter {chapter_num}: {chapter_titles.get(chapter_num, 'Unknown')}"
            boundary_index += 1
        labels.append(current_label)
    return labels


# ---------------------------------------------------------------------------
# Text cleaning (from notebook Cells 21, 23, 25)
# ---------------------------------------------------------------------------

def remove_running_header(text):
    """
    Remove the repeated 4-line running header/footer that appears on nearly
    every page of this PDF. Comparison is case-insensitive.
    """
    kept_lines = []
    for line in text.split("\n"):
        if line.strip().lower() not in HEADER_LINES:
            kept_lines.append(line)
    return "\n".join(kept_lines)


def fix_ligatures(text):
    """Replace PDF font-ligature characters with their expanded, searchable letter pairs."""
    for ligature, replacement in LIGATURE_MAP.items():
        text = text.replace(ligature, replacement)
    return text


def clean_page_text(text):
    """Apply all document-specific cleaning steps to one page's raw text."""
    text = remove_running_header(text)
    text = fix_ligatures(text)
    text = text.strip()
    return text


# ---------------------------------------------------------------------------
# Text preprocessing (from notebook Cells 30–44)
# ---------------------------------------------------------------------------

def remove_urls(text):
    return re.sub(r"http\S+|www\.\S+", "", text)


def remove_punctuation(text):
    return text.translate(translator)


def remove_numbers(text):
    return re.sub(r"\d+", "", text)


def normalize_whitespace(text):
    return re.sub(r"\s+", " ", text).strip()


def remove_stopwords(tokens, preserve_negation=True):
    if preserve_negation:
        return [t for t in tokens if (t not in stop_words) or (t in protected_negation_words)]
    return [t for t in tokens if t not in stop_words]


def lemmatize_tokens(tokens):
    return [lemmatizer.lemmatize(t, pos="v") for t in tokens]


def preprocess_text(
    text,
    lowercase=True,
    remove_url=True,
    remove_punct=True,
    remove_num=False,
    normalize_space=True,
    remove_stop_words=True,
    preserve_negation=True,
    use_lemmatization=True,
):
    """Apply the full Lab 5 preprocessing pipeline to one string of text."""
    if lowercase:
        text = text.lower()
    if remove_url:
        text = remove_urls(text)
    if remove_punct:
        text = remove_punctuation(text)
    if remove_num:
        text = remove_numbers(text)
    if normalize_space:
        text = normalize_whitespace(text)

    if not text:
        return ""

    tokens = word_tokenize(text)

    if remove_stop_words:
        tokens = remove_stopwords(tokens, preserve_negation=preserve_negation)

    if use_lemmatization:
        tokens = lemmatize_tokens(tokens)

    return " ".join(tokens)


# ---------------------------------------------------------------------------
# Chunking (from notebook Cell 72)
# ---------------------------------------------------------------------------

def chunk_text(text, chunk_size=100, overlap=25):
    """
    Split `text` into overlapping chunks of `chunk_size` words, reusing `overlap`
    words between consecutive chunks so ideas at the boundary aren't lost.
    """
    words = text.split()
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if overlap < 0:
        raise ValueError("overlap cannot be negative")
    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size")

    chunks = []
    start = 0
    while start < len(words):
        end = start + chunk_size
        chunks.append(" ".join(words[start:end]))
        if end >= len(words):
            break
        start += chunk_size - overlap
    return chunks


# ---------------------------------------------------------------------------
# Retrieval (from notebook Cells 77, 79)
# ---------------------------------------------------------------------------

def retrieve_top_k_chunks(query, embedding_model, chunks_dataframe, faiss_index, k=8):
    """Retrieve the top-k most similar chunks to `query` using FAISS."""
    query_embedding = embedding_model.encode(
        [query], convert_to_numpy=True, normalize_embeddings=True
    ).astype("float32")
    scores, indices = faiss_index.search(query_embedding, k)

    results = chunks_dataframe.iloc[indices[0]].copy()
    results["score"] = scores[0]
    return results[["chunk_id", "page_number", "chapter", "score", "chunk_text"]].reset_index(drop=True)


def build_context_package(query, embedding_model, chunks_dataframe, faiss_index,
                           retrieval_k=10, max_context_chunks=4,
                           max_chunks_per_page=2, word_budget=220):
    """
    Retrieve candidate chunks for `query`, then filter them into a clean context
    package: deduplicated, capped per source page, and within a word budget.
    """
    candidates = retrieve_top_k_chunks(query, embedding_model, chunks_dataframe, faiss_index, k=retrieval_k)

    selected_rows = []
    seen_texts = set()
    per_page_counts = {}
    used_words = 0

    for _, row in candidates.iterrows():
        normalized = re.sub(r"\s+", " ", row["chunk_text"]).strip().lower()
        if normalized in seen_texts:
            continue

        page_count = per_page_counts.get(row["page_number"], 0)
        if page_count >= max_chunks_per_page:
            continue

        chunk_words = len(row["chunk_text"].split())
        if selected_rows and used_words + chunk_words > word_budget:
            continue

        selected_rows.append(row.to_dict())
        seen_texts.add(normalized)
        per_page_counts[row["page_number"]] = page_count + 1
        used_words += chunk_words

        if len(selected_rows) >= max_context_chunks:
            break

    blocks = []
    for position, row in enumerate(selected_rows, start=1):
        blocks.append(
            f"[Source {position}] (Page {row['page_number']}, {row['chapter']})\n{row['chunk_text']}"
        )

    return {
        "query": query,
        "candidates": candidates,
        "selected": pd.DataFrame(selected_rows),
        "context_text": "\n\n".join(blocks),
        "used_words": used_words,
        "num_sources": len(selected_rows),
    }


# ---------------------------------------------------------------------------
# Prompt engineering (from notebook Cell 81)
# ---------------------------------------------------------------------------

def build_better_prompt(query, context_text):
    """The default prompt builder used by the pipeline."""
    return f'''You are a petroleum engineering assistant answering questions using only the supplied
textbook excerpts as evidence.

Rules:
1. Answer using ONLY the information in the context below.
2. If the context does not contain enough information, say so clearly.
3. Keep your answer concise and directly address the question.
4. Mention which source number(s) you used.

Question: {query}

Context:
{context_text}

Answer:'''


# ---------------------------------------------------------------------------
# Generation (from notebook Cell 83)
# ---------------------------------------------------------------------------

def generate_answer(query, context_text, model_name="deepseek-r1:1.5b", prompt_builder=build_better_prompt):
    """
    Send a grounded prompt to our local Ollama server and return the generated answer.
    """
    prompt = prompt_builder(query, context_text)
    try:
        response = ollama.chat(
            model=model_name,
            messages=[{"role": "user", "content": prompt}],
        )
        return response["message"]["content"]
    except Exception as e:
        return (
            "⚠️ Could not reach Ollama. Make sure Ollama is installed and running, "
            f"and that you've pulled the model with `ollama pull {model_name}`.\n"
            f"Underlying error: {e}"
        )


# ---------------------------------------------------------------------------
# Hugging Face Inference API (for cloud deployment on HF Spaces)
# ---------------------------------------------------------------------------

def generate_answer_hf(query, context_text,
                       model_name="HuggingFaceH4/zephyr-7b-beta",
                       prompt_builder=build_better_prompt,
                       hf_token=None):
    """
    Send a grounded prompt to the Hugging Face Inference API and return the answer.
    Uses the free serverless Inference API — no GPU needed.
    """
    if InferenceClient is None:
        return "⚠️ huggingface_hub is not installed. Run: pip install huggingface_hub"

    prompt = prompt_builder(query, context_text)
    try:
        client = InferenceClient(token=hf_token)
        response = client.text_generation(
            prompt,
            model=model_name,
            max_new_tokens=512,
            temperature=0.3,
            repetition_penalty=1.1,
        )
        return response.strip()
    except Exception as e:
        return (
            f"⚠️ Hugging Face Inference API error.\n"
            f"Model: {model_name}\n"
            f"Underlying error: {e}"
        )


# ---------------------------------------------------------------------------
# Full pipeline (from notebook Cell 85)
# ---------------------------------------------------------------------------

def run_rag_pipeline(query, embedding_model, chunks_dataframe, chunk_faiss_index,
                     k_retrieval=10, max_context_chunks=4, verbose=False):
    """Run the full RAG pipeline: retrieve -> build context -> prompt -> generate."""
    package = build_context_package(
        query, embedding_model, chunks_dataframe, chunk_faiss_index,
        retrieval_k=k_retrieval, max_context_chunks=max_context_chunks
    )
    answer = generate_answer(query, package["context_text"])
    return package, answer
