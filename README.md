# Turkish RAG Assistant

> **Retrieval-Augmented Generation (RAG) pipeline for Turkish documents**  
> Zero API cost · Fully local inference · LangChain + ChromaDB + HuggingFace

[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python)](https://python.org)
[![LangChain](https://img.shields.io/badge/LangChain-0.2-green)](https://langchain.com)
[![ChromaDB](https://img.shields.io/badge/ChromaDB-0.4-orange)](https://trychroma.com)
[![HuggingFace](https://img.shields.io/badge/HuggingFace-Transformers-yellow)](https://huggingface.co)
[![License](https://img.shields.io/badge/License-MIT-blue)](LICENSE)

---

## What is this?

This project implements a production-ready **RAG (Retrieval-Augmented Generation)** pipeline optimized for Turkish-language documents. It allows you to:

- **Index** any `.txt` or `.pdf` file into a local ChromaDB vector store
- **Retrieve** the most relevant document chunks for a query
- **Generate** grounded answers using a local HuggingFace model (no OpenAI key needed)

The pipeline uses **multilingual sentence embeddings** that handle Turkish text natively, making it suitable for Turkce document QA, enterprise knowledge bases, and research assistants.

---

## Architecture

```
Document(s)
    │
    ▼
TextSplitter (chunk_size=500, overlap=50)
    │
    ▼
HuggingFace Embeddings          ← paraphrase-multilingual-MiniLM-L12-v2
(sentence-transformers)
    │
    ▼
ChromaDB VectorStore            ← persistent local storage
    │
    ▼
Retriever (top-k=3)
    │
    ▼
RetrievalQA Chain (LangChain)   ← PromptTemplate + HuggingFacePipeline
    │
    ▼
Answer + Source Citations
```

---

## Quickstart

```bash
# 1. Clone the repo
git clone https://github.com/Umitsencer/Turkish-RAG-Assistant.git
cd Turkish-RAG-Assistant

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the demo (uses the included sample document)
python demo.py --docs docs/ornek.txt --query "Turkiye'nin baskenti neresidir?"
```

### Expected Output

```
 Turkish RAG Assistant - Baslatiliyor...
[RAG] 6 chunk vektor deposuna eklendi.

Soru : Turkiye'nin baskenti neresidir?
Yanit: Ankara

Kaynak Parcalar:
  [1] docs/ornek.txt -> Baskenti Ankara olup, en buyuk sehri Istanbul'dur...
  [2] docs/ornek.txt -> Turkiye, Avrupa ve Asya kitalarinda yer alan...
```

---

## Programmatic Usage

```python
from src.rag_pipeline import TurkishRAGPipeline

# Initialize
rag = TurkishRAGPipeline(persist_dir="./my_db")

# Index documents (first time)
rag.build_vectorstore(["my_document.pdf", "report.txt"])

# Build QA chain
rag.build_qa_chain(top_k=3)

# Query
result = rag.query("Bu raporda hangi riskler belirtilmistir?")
print(result["answer"])
for src in result["sources"]:
    print(f"  Source: {src['source']}")

# Load existing index (subsequent runs — much faster)
rag2 = TurkishRAGPipeline()
rag2.load_existing_vectorstore()
rag2.build_qa_chain()
```

---

## Project Structure

```
Turkish-RAG-Assistant/
├── src/
│   ├── __init__.py
│   └── rag_pipeline.py        # Core RAG pipeline class
├── tests/
│   ├── __init__.py
│   └── test_pipeline.py       # Unit tests (pytest + mocking)
├── docs/
│   └── ornek.txt              # Sample Turkish document
├── demo.py                    # CLI demo script
├── requirements.txt
└── README.md
```

---

## Tests

```bash
# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=src --cov-report=term-missing

# Run only unit tests (no model download)
pytest tests/ -v -k "TestConstants"
```

### Test Results

```
tests/test_pipeline.py::TestConstants::test_chunk_size_positive        PASSED
tests/test_pipeline.py::TestConstants::test_chunk_overlap_less_than_size PASSED
tests/test_pipeline.py::TestConstants::test_chunk_size_value           PASSED
tests/test_pipeline.py::TestConstants::test_chunk_overlap_value        PASSED
tests/test_pipeline.py::TestRAGPipelineUnit::test_init_sets_persist_dir PASSED
tests/test_pipeline.py::TestRAGPipelineUnit::test_vectorstore_initially_none PASSED
tests/test_pipeline.py::TestRAGPipelineUnit::test_qa_chain_initially_none PASSED
tests/test_pipeline.py::TestRAGPipelineUnit::test_query_raises_without_chain PASSED
tests/test_pipeline.py::TestRAGPipelineUnit::test_similarity_search_raises_without_vectorstore PASSED
tests/test_pipeline.py::TestRAGPipelineUnit::test_build_qa_chain_raises_without_vectorstore PASSED
tests/test_pipeline.py::TestRAGPipelineUnit::test_load_nonexistent_file_raises PASSED

11 passed in 0.43s
```

---

## Key Design Decisions

| Decision | Rationale |
|---|---|
| **ChromaDB** over FAISS | Persistent storage, no re-indexing on restart |
| **Multilingual embeddings** | Native Turkish support without fine-tuning |
| **Local LLM** (HuggingFace) | Zero API cost, works offline, KVKK-compliant |
| **RecursiveCharacterTextSplitter** | Preserves semantic boundaries better than fixed-size |
| **Mock-based tests** | CI runs without downloading 400MB+ models |

---

## Extending the Pipeline

**Swap the generator model** for a more powerful one:
```python
GENERATOR_MODEL = "mistralai/Mistral-7B-Instruct-v0.3"  # requires GPU
GENERATOR_MODEL = "google/flan-t5-base"                  # lightweight CPU
```

**Add more document types**:
```python
from langchain_community.document_loaders import UnstructuredWordDocumentLoader
# Just add it to load_documents() method
```

---

## Related Projects

- [Universal-Churn-Prediction](https://github.com/Umitsencer/Universal-Churn-Prediction) — Gradient Boosting churn model
- [Smart-OCR-Extractor](https://github.com/Umitsencer/Smart-OCR-Extractor) — EasyOCR + OpenCV for Turkish IDs
- [Finansal-Haber-NLP](https://github.com/Umitsencer/Finansal-Haber-NLP) — FinBERT sentiment signals

---

## Author

**Umit Sencer** — Software Engineering Student, Kirklareli University  
TUBiTAK 2209-A Researcher | AI/ML | LLM | Computer Vision

[GitHub](https://github.com/Umitsencer) · [LinkedIn](https://linkedin.com/in/umitsencer)
