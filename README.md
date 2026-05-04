# Turkish RAG Assistant

> **Retrieval-Augmented Generation (RAG) pipeline for Turkish documents**  
> Zero API cost · Fully local inference · LangChain + ChromaDB + HuggingFace

[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python)](https://python.org)
[![LangChain](https://img.shields.io/badge/LangChain-0.2-green)](https://langchain.com)
[![ChromaDB](https://img.shields.io/badge/ChromaDB-0.4-orange)](https://trychroma.com)
[![HuggingFace](https://img.shields.io/badge/HuggingFace-Transformers-yellow)](https://huggingface.co)
[![Tests](https://img.shields.io/badge/tests-14%20passed-brightgreen)](tests/)
[![License](https://img.shields.io/badge/License-MIT-blue)](LICENSE)

---

## What is this?

This project implements a production-ready **RAG (Retrieval-Augmented Generation)** pipeline optimized for Turkish-language documents. It allows you to:

- **Index** any `.txt` or `.pdf` file into a local ChromaDB vector store
- **Retrieve** the most relevant document chunks for a query using multilingual embeddings
- **Generate** grounded answers using a local HuggingFace model — no OpenAI key needed
- **Cite sources** — every answer comes with the source chunk it was derived from

---

## Architecture

```
Document(s) (.txt / .pdf)
        │
        ▼
RecursiveCharacterTextSplitter  (chunk_size=500, overlap=50)
        │
        ▼
HuggingFace Embeddings          ← paraphrase-multilingual-MiniLM-L12-v2
(sentence-transformers)             Native Turkish + 50 language support
        │
        ▼
ChromaDB VectorStore            ← persistent local storage (no re-index on restart)
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
# 1. Clone
git clone https://github.com/Umitsencer/Turkish-RAG-Assistant.git
cd Turkish-RAG-Assistant

# 2. Install
pip install -r requirements.txt

# 3. Run demo (uses included sample document)
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

rag = TurkishRAGPipeline(persist_dir="./my_db")

# First run — index documents
rag.build_vectorstore(["report.pdf", "notes.txt"])
rag.build_qa_chain(top_k=3)

result = rag.query("Bu raporda hangi riskler belirtilmistir?")
print(result["answer"])
for src in result["sources"]:
    print(f"  Source: {src['source']}")

# Subsequent runs — load existing index (much faster)
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
│   └── rag_pipeline.py        # Core TurkishRAGPipeline class
├── tests/
│   └── test_pipeline.py       # 14 unit tests (pytest + mocking)
├── docs/
│   └── ornek.txt              # Sample Turkish document
├── demo.py                    # CLI demo script
├── requirements.txt
└── README.md
```

---

## Tests

```bash
# Run all tests (no model download needed)
pytest tests/ -v

# With coverage report
pytest tests/ --cov=src --cov-report=term-missing
```

### Test Results (Python 3.11, pytest 9.0)

```
============================= test session starts =============================
platform win32 -- Python 3.11.9, pytest-9.0.3

tests/test_pipeline.py::TestConstants::test_chunk_size_positive                        PASSED
tests/test_pipeline.py::TestConstants::test_chunk_overlap_less_than_size               PASSED
tests/test_pipeline.py::TestConstants::test_chunk_size_value                           PASSED
tests/test_pipeline.py::TestConstants::test_chunk_overlap_value                        PASSED
tests/test_pipeline.py::TestRAGPipelineUnit::test_init_sets_persist_dir                PASSED
tests/test_pipeline.py::TestRAGPipelineUnit::test_vectorstore_initially_none           PASSED
tests/test_pipeline.py::TestRAGPipelineUnit::test_qa_chain_initially_none              PASSED
tests/test_pipeline.py::TestRAGPipelineUnit::test_query_raises_without_chain           PASSED
tests/test_pipeline.py::TestRAGPipelineUnit::test_similarity_search_raises_without_vectorstore PASSED
tests/test_pipeline.py::TestRAGPipelineUnit::test_build_qa_chain_raises_without_vectorstore PASSED
tests/test_pipeline.py::TestRAGPipelineUnit::test_load_nonexistent_file_raises         PASSED
tests/test_pipeline.py::TestRAGPipelineUnit::test_load_existing_file                  PASSED
tests/test_pipeline.py::TestRAGPipelineUnit::test_build_vectorstore_sets_vectorstore   PASSED
tests/test_pipeline.py::TestRAGPipelineUnit::test_query_returns_dict_after_setup       PASSED

======================== 14 passed in 0.07s ==============================
```

---

## Key Design Decisions

| Decision | Rationale |
|---|---|
| **ChromaDB** over FAISS | Persistent storage — no re-indexing on restart |
| **Multilingual embeddings** | Native Turkish support without fine-tuning |
| **Local LLM (HuggingFace)** | Zero API cost, works offline, KVKK-compliant |
| **RecursiveCharacterTextSplitter** | Preserves semantic boundaries better than fixed-size |
| **Mock-based tests** | CI runs in <0.1s — no 400MB+ model download |

---

## Swap the Generator Model

```python
# In src/rag_pipeline.py — change one line:
GENERATOR_MODEL = "mistralai/Mistral-7B-Instruct-v0.3"  # powerful, needs GPU
GENERATOR_MODEL = "google/flan-t5-base"                  # lightweight, CPU-only
GENERATOR_MODEL = "Helsinki-NLP/opus-mt-tr-en"           # default (smallest)
```

---

## Related Projects

- [Turkish-NLP-Text-Classifier](https://github.com/Umitsencer/Turkish-NLP-Text-Classifier) — BERT zero-shot text classifier
- [Finansal-Haber-NLP](https://github.com/Umitsencer/Finansal-Haber-NLP) — FinBERT sentiment signals
- [Smart-OCR-Extractor](https://github.com/Umitsencer/Smart-OCR-Extractor) — OCR for Turkish IDs

---

## Author

**Umit Sencer** — Software Engineering Student, Kirklareli University  
TUBiTAK 2209-A Researcher | AI/ML | LLM | Computer Vision

[GitHub](https://github.com/Umitsencer) · [LinkedIn](https://linkedin.com/in/umitsencer)
