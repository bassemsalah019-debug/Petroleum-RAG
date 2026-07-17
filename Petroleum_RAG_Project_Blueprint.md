# Petroleum RAG Project Blueprint

## Project Title

**Petroleum Knowledge Assistant using Retrieval-Augmented Generation
(RAG)**

------------------------------------------------------------------------

# 1. Project Goal

Build a complete educational RAG project in **one Jupyter Notebook**
using the exact workflow learned in the previous Labs (Preprocessing →
TF-IDF → Embeddings → FAISS → Chunking → Context → Prompt → LLM), but
replace the 20 Newsgroups dataset with a real Petroleum Engineering
document.

The objective is to understand every RAG component, not to use
high-level frameworks.

------------------------------------------------------------------------

# 2. Knowledge Base

## Dataset Location

The project MUST use the following PDF as its ONLY knowledge base.

**Absolute Path (Windows):**

``` text
C:\Users\Admin\Desktop\Petroleum RAG\geokniga-drillingengineeringprasslwl.pdf
```

Claude Code should first verify that this file exists.

Example:

``` python
from pathlib import Path

DATA_PATH = Path(r"C:\Users\Admin\Desktop\Petroleum RAG\geokniga-drillingengineeringprasslwl.pdf")

if not DATA_PATH.exists():
    raise FileNotFoundError(f"PDF not found: {DATA_PATH}")
```

After loading this PDF, execute the entire pipeline:

1.  Extract text using PyMuPDF (fitz)
2.  Build a DataFrame (page/document structure)
3.  Perform EDA
4.  Text preprocessing
5.  Bag of Words
6.  TF-IDF
7.  Keyword Retrieval
8.  SentenceTransformer (all-MiniLM-L6-v2)
9.  FAISS Index
10. Chunking
11. Chunk Retrieval
12. Context Building
13. Prompt Engineering
14. Connect to Ollama (deepseek-r1:1.5b)
15. Evaluate with petroleum engineering questions

Rules: - Use ONLY this PDF. - Do NOT download any external datasets. -
Do NOT use LangChain, LlamaIndex, Chroma, or Pinecone. - Build every
component manually in one educational notebook named
`Petroleum_RAG_Project.ipynb`. - Explain every concept before the code,
matching the style of the previous RAG labs.

## Dataset

The project uses **ONE PDF ONLY**.

**Document**

Drilling Engineering (Curtin University -- Master of Petroleum
Engineering)

This PDF is the entire knowledge base.

Claude must NOT download any other dataset.

No PetroWiki.

No SPE papers.

No additional PDFs.

Everything must be extracted from this single PDF.

------------------------------------------------------------------------

# 3. Technologies

-   Python
-   Pandas
-   NumPy
-   Matplotlib
-   NLTK
-   scikit-learn
-   sentence-transformers
-   FAISS
-   PyMuPDF (fitz)
-   Ollama
-   deepseek-r1:1.5b

Forbidden: - LangChain - LlamaIndex - Chroma - Pinecone

------------------------------------------------------------------------

# 4. Notebook Structure

1.  Introduction
2.  Install & Imports
3.  Load PDF
4.  Extract Text
5.  Build DataFrame
6.  Exploratory Data Analysis
7.  Text Preprocessing
8.  Bag of Words
9.  TF-IDF
10. Keyword Retrieval
11. Embeddings (all-MiniLM-L6-v2)
12. Semantic Retrieval
13. FAISS Index
14. Chunking
15. Chunk Embeddings
16. Chunk Retrieval
17. Context Builder
18. Prompt Engineering
19. Connect to Ollama
20. Ask Petroleum Questions
21. Compare TF-IDF vs Semantic vs RAG
22. Conclusion

------------------------------------------------------------------------

# 5. Educational Rules

Every section must include:

-   Markdown explanation
-   Why this step exists
-   Input
-   Output
-   Expected result
-   Fully commented code

The notebook should teach beginners.

------------------------------------------------------------------------

# 6. RAG Pipeline

PDF ↓ Extract Text ↓ EDA ↓ Preprocessing ↓ Bag of Words ↓ TF-IDF ↓
Keyword Retrieval ↓ Sentence Embeddings ↓ FAISS ↓ Chunking ↓ Chunk
Retrieval ↓ Context Building ↓ Prompt Construction ↓ DeepSeek (Ollama) ↓
Grounded Answer

------------------------------------------------------------------------

# 7. Evaluation Questions

Prepare at least 15 petroleum questions such as:

-   What is lost circulation?
-   What causes differential sticking?
-   What is mud weight?
-   What is the purpose of the BOP?
-   What is tripping?
-   What is well control?
-   What causes a kick?
-   What is casing?
-   What is drilling fluid?
-   How is drill string designed?

Compare: - TF-IDF Retrieval - Embedding Retrieval - Final RAG Answer

------------------------------------------------------------------------

# 8. Claude Code Prompt

Create ONE standalone notebook named:

Petroleum_RAG_Project.ipynb

Requirements:

-   Follow exactly the educational style of the previous NLP/RAG
    notebook.
-   Use the provided PDF as the ONLY knowledge source.
-   Do not download any external data.
-   Explain every concept before writing code.
-   Keep all code executable from top to bottom.
-   Reuse the same preprocessing pipeline.
-   Reuse TF-IDF.
-   Reuse SentenceTransformer(all-MiniLM-L6-v2).
-   Build FAISS manually.
-   Implement chunking manually.
-   Build context manually.
-   Build prompts manually.
-   Connect to Ollama using deepseek-r1:1b or deepseek-r1:1.5b.
-   Finish with qualitative evaluation and conclusions.

The notebook should be portfolio-quality and educational rather than
production-scale.
