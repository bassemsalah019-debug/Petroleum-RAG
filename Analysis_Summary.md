# Petroleum Knowledge Assistant — Analysis Summary

> **Notebook:** `Petroleum_RAG_Project.ipynb`
> **Date:** 2026-07-16
> **Status:** ⚠️ Notebook was NOT successfully executed — all 56 code cells produced zero outputs.
> The kernel crashed at Cell 2 (imports) due to a `tf_keras` / Keras 3 compatibility error.
> The error has been fixed (`pip install tf-keras`). **Re-run the notebook to generate actual results.**

---

## 1. Project Overview

| Item | Detail |
|---|---|
| **Goal** | Build a Retrieval-Augmented Generation (RAG) system for a petroleum engineering textbook |
| **Knowledge base** | *Drilling Engineering* — 282-page PDF (Curtin University, Master of Petroleum Engineering) |
| **LLM** | DeepSeek-R1:1.5b via local Ollama |
| **Embedding model** | `all-MiniLM-L6-v2` (384 dimensions) |
| **Vector store** | FAISS (IndexFlatIP — inner product on normalized vectors = cosine similarity) |

---

## 2. Pipeline Architecture

```
PDF (282 pages)
  │
  ├── Text Extraction (PyMuPDF / fitz)
  │     └── raw_pages: list[str]
  │
  ├── DataFrame Construction
  │     └── pages_df: page_number, raw_text, chapter, word_count, char_count
  │
  ├── Text Cleaning
  │     ├── Remove running headers (Curtin University 4-line header)
  │     ├── Fix ligatures (fi, fl, ff, ffi, ffl)
  │     └── Filter pages < 40 words + exclude Front Matter
  │
  ├── Text Preprocessing (Lab 5 pipeline)
  │     ├── Lowercase
  │     ├── Remove URLs
  │     ├── Remove punctuation
  │     ├── Keep numbers (domain-specific: mud weights, pressures, depths)
  │     ├── Normalize whitespace
  │     ├── Tokenize (NLTK word_tokenize)
  │     ├── Remove stopwords (preserve negation: no, not, nor, never)
  │     └── Lemmatize (WordNetLemmatizer)
  │
  ├── Representation
  │     ├── Bag-of-Words (CountVectorizer)
  │     └── TF-IDF (TfidfVectorizer)
  │
  ├── Embeddings & Indexing
  │     ├── Page-level embeddings → FAISS page index
  │     ├── Chunking (100 words, 25-word overlap)
  │     └── Chunk-level embeddings → FAISS chunk index
  │
  ├── Retrieval & Context Building
  │     ├── Retrieve top-k chunks via FAISS
  │     ├── Deduplicate by source page
  │     ├── Cap: max 2 chunks per page, max 4 total
  │     └── Word budget: 220 words
  │
  ├── Prompt Engineering (3 templates: weak, better, strict)
  │
  └── Generation (DeepSeek-R1:1.5b via Ollama)
```

---

## 3. Exploratory Data Analysis (Expected Results)

| Metric | Expected Value |
|---|---|
| Total pages (raw) | 282 |
| Chapters discovered | 12 + Front Matter |
| Mean words/page | ~190–215 |
| Median words/page | ~190–215 |
| Min words/page | ~10 (section dividers, figure pages) |
| Pages after filtering (< 40 words + Front Matter removed) | ~240–260 |

### Chapters Identified
1. Geomechanics
2. Drilling Fluids
3. Drill String Design
4. Drill Bits
5. Drilling Hydraulics
6. Directional Drilling
7. Well Control
8. Casing and Cementing
9. Completion
10. Borehole Problems (Lost Circulation, Stuck Pipe)
11. Kick Control and Blowout Prevention
12. Drilling Operations

---

## 4. Text Representation Results (Expected)

### Bag-of-Words
| Metric | Expected |
|---|---|
| Vocabulary size | ~5,000–10,000 unique terms |
| Matrix shape | (filtered_pages × vocabulary) |
| Sparsity | ~95–99% |

### TF-IDF
| Metric | Expected |
|---|---|
| Vocabulary size | Same as BoW |
| Matrix shape | Same as BoW |

---

## 5. Embedding & Retrieval (Expected)

| Metric | Expected |
|---|---|
| Embedding model | all-MiniLM-L6-v2 |
| Embedding dimension | 384 |
| Page embeddings shape | (~250, 384) |
| Chunk size | 100 words, 25-word overlap |
| Chunk embeddings shape | (~800–1200, 384) |
| FAISS index type | IndexFlatIP (exact search) |

---

## 6. Evaluation Framework

### 6.1 Ground Truth Dataset
- **48 questions** across **16 topics**
- Difficulty distribution: Easy / Medium / Hard
- Each question has: ground truth answer, expected keywords, source page, source chunk ID

| Topic | # Questions | Difficulty Range |
|---|---|---|
| Drilling Fluids | 3 | Easy–Medium |
| Mud Weight | 2 | Medium |
| Formation Pressure | 3 | Easy–Hard |
| Well Control | 3 | Easy–Medium |
| BOP | 5 | Easy–Hard |
| Casing | 4 | Easy–Hard |
| Cementing | 4 | Easy–Medium |
| Drill String | 3 | Easy–Medium |
| Drill Bits | 3 | Easy–Hard |
| Directional Drilling | 3 | Easy–Medium |
| Lost Circulation | 3 | Easy–Medium |
| Stuck Pipe | 4 | Medium–Hard |
| Kick Detection | 3 | Medium–Hard |
| Completion | 1 | Medium |
| Safety | 2 | Easy–Hard |
| Drilling Operations | 2 | Easy–Hard |

### 6.2 Retrieval Quality Metrics (to be computed)
| Metric | Description |
|---|---|
| Average top similarity score | Mean cosine similarity of the #1 retrieved chunk |
| Average chunks retrieved | After context building (dedup + cap) |
| Average context length | Words fed to the LLM |
| Ground-truth page retrieval success rate | % of questions where the correct page appears in retrieved results |

### 6.3 Answer Quality Classification
| Label | Keyword Coverage Threshold |
|---|---|
| ✅ Correct | ≥ 60% of expected keywords found in generated answer |
| ⚠️ Partially Correct | 25–59% keyword coverage |
| ❌ Incorrect | < 25% keyword coverage |
| ⏸️ Not Evaluated | Ollama was unreachable |

### 6.4 Automatic Metrics (6 metrics)
| Metric | What It Measures |
|---|---|
| Exact Match | Binary: generated == ground truth (expected ~0) |
| Keyword Overlap | Fraction of expected keywords present in answer |
| Token Precision | Unique token overlap / unique generated tokens |
| Token Recall | Unique token overlap / unique ground truth tokens |
| Token F1 | Harmonic mean of precision and recall |
| Semantic Similarity | Cosine similarity of embedding vectors (generated vs ground truth) |

### 6.5 Final Summary Scorecard (to be computed)
| Field | Description |
|---|---|
| Total Questions | 48 |
| Correct / Partially Correct / Incorrect / Not Evaluated | Counts per category |
| Exact Match Score | ~0 (free-text generation rarely matches exactly) |
| Average Token F1 | Expected moderate (0.2–0.5) |
| Average Semantic Similarity | Expected good (0.5–0.8) |
| Average Retrieval Similarity | Expected high (0.6–0.9) |
| Overall Retrieval Success Rate | Expected 70–90% |

---

## 7. Three Retrieval Methods Compared

| Method | Strengths | Weaknesses |
|---|---|---|
| **TF-IDF** | Exact keyword matches score high | Fails on paraphrases ("fluid disappears into rock" ≠ "lost circulation") |
| **Semantic (Embeddings)** | Handles paraphrases and synonyms | May retrieve topically related but wrong-chapter passages |
| **RAG (Full Pipeline)** | Grounded answers with source citations | Quality depends on retrieval + LLM (1.5B model may over-summarize) |

---

## 8. Known Limitations

1. **PDF extraction quality** — formulas, figures, and tables are lost in plain-text extraction
2. **Page-level chapter labeling** — coarse granularity; sub-sections not tracked
3. **Residual structural noise** — headers/footers may not be fully removed on all pages
4. **Small local LLM** — DeepSeek-R1:1.5b has only 1.5B parameters; may hallucinate or over-summarize
5. **No hybrid retrieval** — no reranking, no query expansion
6. **Single-document scope** — only one textbook; no cross-document reasoning
7. **Fixed chunk size** — 100 words / 25 overlap; not optimized per topic

---

## 9. Recommended Next Steps

1. **Re-run the entire notebook** — the import error is now fixed; all cells should execute
2. **Verify Ollama is running** — `ollama list` should show `deepseek-r1:1.5b`
3. **Review evaluation results** — check the final scorecard (Cell 112) for actual metrics
4. **Consider:**
   - Trying `deepseek-r1:7b` for better answer quality
   - Experimenting with chunk_size (50, 150, 200) and overlap (10, 50)
   - Adding a reranker (e.g., `cross-encoder/ms-marco-MiniLM-L-6-v2`)
   - Extracting tables/figures separately with specialized tools

---

*This summary was generated from code analysis — no execution outputs were available.*
*Re-run the notebook and update this file with actual metrics.*
