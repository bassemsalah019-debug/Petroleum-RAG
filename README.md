# 🛢️ Petroleum Engineering RAG Chatbot

A Retrieval-Augmented Generation (RAG) chatbot that answers petroleum engineering questions using a drilling engineering textbook as its knowledge source. Built from scratch -- no LangChain, no LlamaIndex -- every component is implemented manually for educational clarity.

<p align="center">
  <a href="https://petroleum-rag-ftzg9bv987qglhurjku8yz.streamlit.app/">
    <img src="https://readme-typing-svg.herokuapp.com?font=Poppins&size=30&duration=2500&pause=800&color=F7B500&center=true&vCenter=true&width=700&lines=🚀+Try+the+Petroleum+RAG+Application;🤖+AI-Powered+Document+Question+Answering;📚+Powered+by+FAISS+%2B+BGE+Embeddings+%2B+Qwen7B"/>
  </a>
</p>  

<p align="center">
  <a href="https://petroleum-rag-ftzg9bv987qglhurjku8yz.streamlit.app/">
    <img src="https://img.shields.io/badge/OPEN%20LIVE%20APP-FFB000?style=for-the-badge&logo=streamlit&logoColor=white" />
  </a>
</p>

## How It Works

1. **You ask a question** about drilling engineering
2. **FAISS vector search** finds the most relevant textbook passages
3. **LLM generates an answer** grounded in those passages (Qwen2.5-7B via HF Inference API, or DeepSeek-R1:1.5b via Ollama)
4. **Sources are cited** so you can verify the information

## Architecture

``` text
Question -> Embedding (bge-base-en-v1.5, 768d) -> FAISS Search -> Context Building -> LLM -> Answer
```

| Component               | Detail                                                                                      |
| ----------------------- | ------------------------------------------------------------------------------------------- |
| **Embedding Model**     | `BAAI/bge-base-en-v1.5` (768 dimensions)                                                    |
| **Vector Store**        | FAISS with cosine similarity (IndexFlatIP on normalized vectors)                            |
| **LLM Backend (Cloud)** | Hugging Face Inference API -- `Qwen/Qwen2.5-7B-Instruct` (default)                          |
| **LLM Backend (Local)** | Ollama + `deepseek-r1:1.5b`                                                                 |
| **Source Textbook**     | *Drilling Engineering* -- 282-page PDF (Curtin University, Master of Petroleum Engineering) |
| **Frontend (Cloud)**    | Streamlit (deployed on Hugging Face Spaces)                                                 |
| **Frontend (Local)**    | FastAPI + vanilla HTML/CSS/JS                                                               |

## Project Structure

``` text
├── README.md                                    # This file
├── requirements.txt                             # Streamlit deployment dependencies
├── streamlit_app.py                             # Streamlit frontend (HF Spaces / Streamlit Cloud)
├── geokniga-drillingengineeringprasslwl.pdf     # Source textbook (282 pages)
├── .streamlit/
│   └── config.toml                              # Streamlit theme (dark petroleum gold)
├── implementation/
│   ├── app.py                                   # FastAPI server (local development)
│   ├── pipeline_utils.py                        # Core RAG pipeline functions
│   ├── requirements.txt                         # FastAPI deployment dependencies
│   ├── __init__.py
│   ├── static/
│   │   ├── style.css                            # FastAPI UI styles
│   │   └── app.js                               # FastAPI frontend JavaScript
│   ├── templates/
│   │   └── index.html                           # FastAPI chat interface
│   └── .cache/                                  # Cached pipeline artifacts (auto-generated)
│       ├── chunks.pkl                           # Pre-processed text chunks
│       ├── faiss_index.bin                      # FAISS vector index
│       └── metadata.json                        # Cache validity metadata
├── Petroleum_RAG_Project_Blueprint.md           # Project specification & educational goals
├── Analysis_Summary.md                          # Pipeline analysis & evaluation framework
└── Chat_Summary.md                              # Development conversation log
```

## Running Locally

### Option 1: Streamlit (recommended)

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Set your Hugging Face token (free: https://huggingface.co/settings/tokens)
export HF_TOKEN=hf_your_token_here

# 3. Run
streamlit run streamlit_app.py
# -> Open http://localhost:8501
```

### Option 2: FastAPI with Ollama (fully local, no API key needed)

```bash
# 1. Install Ollama: https://ollama.com
# 2. Pull the model
ollama pull deepseek-r1:1.5b

# 3. Install dependencies
pip install -r implementation/requirements.txt

# 4. Run
python implementation/app.py
# -> Open http://127.0.0.1:8000
```


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

The RAG pipeline is built from scratch in `pipeline_utils.py`:

1. **PDF Extraction** -- PyMuPDF extracts text from all 282 pages
2. **Text Cleaning** -- Remove running headers (Curtin University 4-line header), fix ligatures, strip whitespace
3. **Chapter Detection** -- Regex-based table-of-contents scanning identifies 12 chapters
4. **Page Filtering** -- Remove Front Matter and pages with < 40 words (~240-260 pages remain)
5. **Chunking** -- 100-word chunks with 25-word overlap
6. **Embedding** -- `BAAI/bge-base-en-v1.5` encodes chunks + queries (768 dimensions, normalized)
7. **Retrieval** -- FAISS IndexFlatIP for cosine similarity search (top-k=10 candidates)
8. **Context Building** -- Deduplication, max 2 chunks per page, max 4 total, 220-word budget
9. **Generation** -- Grounded prompt sent to LLM with source citations

### Chapters in the Knowledge Base

| #  | Chapter                                          |
| -- | ------------------------------------------------ |
| 1  | Geomechanics                                     |
| 2  | Drilling Fluids                                  |
| 3  | Drill String Design                              |
| 4  | Drill Bits                                       |
| 5  | Drilling Hydraulics                              |
| 6  | Directional Drilling                             |
| 7  | Well Control                                     |
| 8  | Casing and Cementing                             |
| 9  | Completion                                       |
| 10 | Borehole Problems (Lost Circulation, Stuck Pipe) |
| 11 | Kick Control and Blowout Prevention              |
| 12 | Drilling Operations                              |

## Technologies

| Category               | Technology                                              |
| ---------------------- | ------------------------------------------------------- |
| **Frontend (Cloud)**   | Streamlit                                               |
| **Frontend (Local)**   | FastAPI, Jinja2, vanilla JS                             |
| **PDF Processing**     | PyMuPDF (fitz)                                          |
| **NLP**                | NLTK, scikit-learn                                      |
| **Embeddings**         | sentence-transformers (`BAAI/bge-base-en-v1.5`)         |
| **Vector Search**      | FAISS (faiss-cpu)                                       |
| **LLM (Cloud)**        | Hugging Face Inference API (Qwen2.5-7B-Instruct)        |
| **LLM (Local)**        | Ollama (deepseek-r1:1.5b)                               |
| **Data**               | Pandas, NumPy                                           |

**Forbidden frameworks** (by design -- the project teaches manual RAG construction):
LangChain · LlamaIndex · Chroma · Pinecone

## Evaluation Framework

The project includes a comprehensive evaluation framework (documented in `Analysis_Summary.md`):

- **48 ground-truth questions** across **16 petroleum engineering topics**
- **3 retrieval methods compared**: TF-IDF, Semantic Embeddings, Full RAG
- **6 automatic metrics**: Exact Match, Keyword Overlap, Token Precision/Recall/F1, Semantic Similarity
- **Answer quality classification**: Correct (>=60% keywords), Partially Correct (25-59%), Incorrect (<25%)

## Known Limitations

1. **PDF extraction quality** -- formulas, figures, and tables are lost in plain-text extraction
2. **Small local LLM** -- DeepSeek-R1:1.5b (1.5B parameters) may hallucinate or over-summarize
3. **No hybrid retrieval** -- no reranking or query expansion
4. **Single-document scope** -- only one textbook; no cross-document reasoning
5. **Fixed chunk size** -- 100 words / 25 overlap; not optimized per topic

## License

MIT
