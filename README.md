---
title: Petroleum Engineering RAG Chatbot
emoji: 🛢️
colorFrom: gray
colorTo: blue
sdk: docker
app_port: 7860
pinned: false
license: mit
---

# 🛢️ Petroleum Engineering RAG Chatbot

A Retrieval-Augmented Generation (RAG) chatbot that answers petroleum engineering questions using a drilling engineering textbook as its knowledge source.

**[Try it live on Hugging Face Spaces →](https://huggingface.co/spaces/YOUR_USERNAME/petroleum-rag)**

## How It Works

1. **You ask a question** about drilling engineering
2. **FAISS vector search** finds the most relevant textbook passages
3. **DeepSeek / Zephyr LLM** generates an answer grounded in those passages
4. **Sources are cited** so you can verify the information

## Architecture

```
Question → Embedding (bge-base-en-v1.5) → FAISS Search → Context Building → LLM → Answer
```

- **Embedding Model**: `BAAI/bge-base-en-v1.5` (768 dimensions)
- **Vector Store**: FAISS with cosine similarity (IndexFlatIP)
- **LLM Backend**: Hugging Face Inference API (Zephyr-7B) or Ollama + DeepSeek-R1:1.5b
- **Source Textbook**: *Applied Drilling Engineering* (Bourgoyne et al.)

## Project Structure

```
├── Dockerfile                          # Docker deployment for HF Spaces
├── README.md                           # This file
├── geokniga-drillingengineeringprasslwl.pdf  # Source textbook
├── Petroleum_RAG_Project.ipynb         # Original RAG pipeline notebook
└── implementation/
    ├── app.py                          # FastAPI server
    ├── pipeline_utils.py               # Reusable pipeline functions
    ├── requirements.txt                # Python dependencies
    ├── static/
    │   ├── style.css                   # UI styles
    │   └── app.js                      # Frontend JavaScript
    ├── templates/
    │   └── index.html                  # Chat interface
    └── .cache/                         # Cached pipeline artifacts (auto-generated)
```

## Running Locally

### Option 1: With Ollama (recommended for local use)

```bash
# 1. Install Ollama: https://ollama.com
# 2. Pull the model
ollama pull deepseek-r1:1.5b

# 3. Install dependencies
pip install -r implementation/requirements.txt

# 4. Run
python implementation/app.py
# → Open http://127.0.0.1:8000
```

### Option 2: With Hugging Face Inference API

```bash
# 1. Get a free HF token: https://huggingface.co/settings/tokens
# 2. Install dependencies
pip install -r implementation/requirements.txt

# 3. Set your token and run
export HF_TOKEN=hf_your_token_here
python implementation/app.py
# → Open http://127.0.0.1:8000
```

### Option 3: Docker

```bash
docker build -t petroleum-rag .
docker run -p 7860:7860 petroleum-rag
# → Open http://localhost:7860
```

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/` | Chat interface (HTML) |
| `POST` | `/chat` | Ask a question (JSON) |
| `GET` | `/health` | System status |

### POST /chat

```json
// Request
{ "question": "What is differential sticking?" }

// Response
{
  "answer": "Differential sticking occurs when...",
  "sources": [{ "page": 156, "chapter": "Chapter 5: ...", "chunk_id": "..." }],
  "retrieved_chunks": [{ "chunk_id": "...", "score": 0.82, "text": "..." }]
}
```

## Pipeline Details

The RAG pipeline replicates the notebook exactly:

1. **PDF Extraction** — PyMuPDF extracts text from all pages
2. **Text Cleaning** — Remove headers, fix ligatures, strip whitespace
3. **Chapter Detection** — Regex-based table-of-contents scanning
4. **Chunking** — 100-word chunks with 25-word overlap
5. **Embedding** — `BAAI/bge-base-en-v1.5` encodes chunks + queries
6. **Retrieval** — FAISS IndexFlatIP for cosine similarity search
7. **Context Building** — Deduplication, per-page caps, word budget (220 words)
8. **Generation** — Grounded prompt sent to LLM with source citations

## License

MIT
