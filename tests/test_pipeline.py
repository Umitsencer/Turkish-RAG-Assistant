"""Tests for TurkishRAGPipeline.

Run: pytest tests/ -v
"""
import pytest
from unittest.mock import MagicMock, patch

try:
    from src.rag_pipeline import TurkishRAGPipeline, CHUNK_SIZE, CHUNK_OVERLAP
    DEPS_AVAILABLE = True
except ImportError:
    DEPS_AVAILABLE = False
    CHUNK_SIZE = 500
    CHUNK_OVERLAP = 50

requires_deps = pytest.mark.skipif(not DEPS_AVAILABLE, reason="ML deps not installed")


class TestConstants:
    """Always-run tests — no heavy deps needed."""

    def test_chunk_size_positive(self):
        assert CHUNK_SIZE > 0

    def test_chunk_overlap_less_than_size(self):
        assert CHUNK_OVERLAP < CHUNK_SIZE

    def test_chunk_size_value(self):
        assert CHUNK_SIZE == 500

    def test_chunk_overlap_value(self):
        assert CHUNK_OVERLAP == 50


@requires_deps
class TestRAGPipelineUnit:
    """Mock-based unit tests — no model download needed."""

    def _make(self, tmp_path):
        with patch("src.rag_pipeline.HuggingFaceEmbeddings"):
            return TurkishRAGPipeline(persist_dir=str(tmp_path / "db"))

    def test_init_sets_persist_dir(self, tmp_path):
        pipe = self._make(tmp_path)
        assert "db" in pipe.persist_dir

    def test_vectorstore_initially_none(self, tmp_path):
        pipe = self._make(tmp_path)
        assert pipe.vectorstore is None

    def test_qa_chain_initially_none(self, tmp_path):
        pipe = self._make(tmp_path)
        assert pipe.qa_chain is None

    def test_query_raises_without_chain(self, tmp_path):
        pipe = self._make(tmp_path)
        with pytest.raises(RuntimeError, match="build_qa_chain"):
            pipe.query("Test sorusu")

    def test_similarity_search_raises_without_vectorstore(self, tmp_path):
        pipe = self._make(tmp_path)
        with pytest.raises(RuntimeError, match="VectorStore"):
            pipe.similarity_search("test")

    def test_build_qa_chain_raises_without_vectorstore(self, tmp_path):
        pipe = self._make(tmp_path)
        with pytest.raises(RuntimeError, match="build_vectorstore"):
            pipe.build_qa_chain()

    def test_load_nonexistent_file_raises(self, tmp_path):
        pipe = self._make(tmp_path)
        with pytest.raises(FileNotFoundError):
            pipe.load_documents([str(tmp_path / "ghost.txt")])
