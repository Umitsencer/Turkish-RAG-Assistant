"""Turkish RAG (Retrieval-Augmented Generation) Pipeline.

Zero-cost, fully local inference using HuggingFace + ChromaDB.
Author: Umit Sencer | github.com/Umitsencer
"""
from __future__ import annotations
import os
from pathlib import Path
from typing import List

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.llms import HuggingFacePipeline
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from transformers import pipeline as hf_pipeline

EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
GENERATOR_MODEL = "Helsinki-NLP/opus-mt-tr-en"
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50
CHROMA_DIR = "./chroma_db"

_PROMPT_TEMPLATE = """Asagidaki baglam bilgisini kullanarak soruyu yanitla.
Baglamda yanit yoksa 'Bu bilgi belgede bulunmuyor.' de.

Baglam:
{context}

Soru: {question}
Yanit:"""


class TurkishRAGPipeline:
    """End-to-end RAG pipeline for Turkish documents."""

    def __init__(self, persist_dir: str = CHROMA_DIR) -> None:
        self.persist_dir = persist_dir
        self.embeddings = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )
        self.vectorstore: Chroma | None = None
        self.qa_chain = None

    def load_documents(self, paths: List[str]):
        """Load .txt or .pdf files and return LangChain Documents."""
        docs = []
        for p in paths:
            path = Path(p)
            if not path.exists():
                raise FileNotFoundError(f"Dosya bulunamadi: {p}")
            loader = (
                PyPDFLoader(p)
                if path.suffix.lower() == ".pdf"
                else TextLoader(p, encoding="utf-8")
            )
            docs.extend(loader.load())
        return docs

    def build_vectorstore(self, paths: List[str]) -> Chroma:
        """Split documents and index them in ChromaDB."""
        raw_docs = self.load_documents(paths)
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
            separators=["\n\n", "\n", ".", " "],
        )
        chunks = splitter.split_documents(raw_docs)
        self.vectorstore = Chroma.from_documents(
            documents=chunks,
            embedding=self.embeddings,
            persist_directory=self.persist_dir,
        )
        print(f"[RAG] {len(chunks)} chunk vektor deposuna eklendi.")
        return self.vectorstore

    def load_existing_vectorstore(self) -> Chroma:
        """Load an already-persisted ChromaDB."""
        self.vectorstore = Chroma(
            persist_directory=self.persist_dir,
            embedding_function=self.embeddings,
        )
        return self.vectorstore

    def build_qa_chain(self, top_k: int = 3):
        """Build the RetrievalQA chain."""
        if self.vectorstore is None:
            raise RuntimeError("Once build_vectorstore() veya load_existing_vectorstore() cagirin.")
        gen_pipe = hf_pipeline(
            "text2text-generation",
            model=GENERATOR_MODEL,
            max_new_tokens=256,
        )
        llm = HuggingFacePipeline(pipeline=gen_pipe)
        prompt = PromptTemplate(
            input_variables=["context", "question"],
            template=_PROMPT_TEMPLATE,
        )
        self.qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=self.vectorstore.as_retriever(search_kwargs={"k": top_k}),
            chain_type_kwargs={"prompt": prompt},
            return_source_documents=True,
        )
        return self.qa_chain

    def query(self, question: str) -> dict:
        """Ask a question; returns answer + source chunks."""
        if self.qa_chain is None:
            raise RuntimeError("Once build_qa_chain() cagirin.")
        result = self.qa_chain({"query": question})
        return {
            "question": question,
            "answer": result["result"].strip(),
            "sources": [
                {"content": d.page_content[:200], "source": d.metadata.get("source", "?")}
                for d in result.get("source_documents", [])
            ],
        }

    def similarity_search(self, query: str, k: int = 5) -> list:
        """Raw similarity search for debugging retrieval quality."""
        if self.vectorstore is None:
            raise RuntimeError("VectorStore yuklenmedi.")
        return self.vectorstore.similarity_search(query, k=k)
