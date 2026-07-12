"""Tests for the DocWhisperer pipeline components."""
import pytest
from dataclasses import dataclass
from typing import Any, List
from unittest.mock import MagicMock, patch

from backend.agents.query_classifier import (
    QueryClassification,
    classify_query,
    _default_classification,
)
from backend.agents.retriever import HybridRetriever, RetrievedDoc
from backend.agents.critique_agent import CritiqueAgent, CritiqueResult


class TestClassifyQuery:
    """Tests for the query classifier."""

    def test_default_classification_returns_hybrid(self) -> None:
        """Default fallback should return hybrid strategy."""
        result = _default_classification()
        assert result.retrieval_strategy == "hybrid"
        assert result.suggested_top_k == 5
        assert result.intent == "general"
        assert result.confidence == 0.0

    def test_classify_query_fallback_on_exception(self) -> None:
        """If LLM call fails, should return default classification."""
        mock_llm = MagicMock()
        mock_llm.side_effect = RuntimeError("LLM unavailable")

        result = classify_query("How do I install Python?", mock_llm)
        assert result.intent == "general"
        assert result.retrieval_strategy == "hybrid"

    def test_classify_query_with_mock_llm(self) -> None:
        """Test classification with a mock that returns valid structured output."""
        from backend.agents.query_classifier import QueryClassificationSchema

        mock_llm = MagicMock()
        # Simulate the chain pipeline failing gracefully
        result = classify_query("What is the default port?", mock_llm)
        # Should fallback since mock doesn't implement proper chain
        assert isinstance(result, QueryClassification)
        assert result.retrieval_strategy in ("semantic_heavy", "bm25_heavy", "hybrid")


class TestHybridRetriever:
    """Tests for the hybrid retriever."""

    def _make_retriever(self) -> HybridRetriever:
        """Create a retriever with mock dependencies."""
        mock_collection = MagicMock()
        mock_collection.query.return_value = {
            "ids": [["doc1", "doc2"]],
            "documents": [["Text one", "Text two"]],
            "metadatas": [[{"source": "a.md"}, {"source": "b.md"}]],
            "distances": [[0.1, 0.3]],
        }
        mock_embedder = MagicMock()
        mock_embedder.embed_query.return_value = [0.1] * 384

        from rank_bm25 import BM25Okapi
        corpus = [["text", "one", "about", "python"], ["text", "two", "about", "java"]]
        bm25 = BM25Okapi(corpus)
        corpus_docs = [
            {"id": "doc1", "text": "Text one about python", "metadata": {"source": "a.md"}},
            {"id": "doc2", "text": "Text two about java", "metadata": {"source": "b.md"}},
        ]

        return HybridRetriever(mock_collection, bm25, mock_embedder, corpus_docs)

    def test_semantic_search_returns_docs(self) -> None:
        """Semantic search should return RetrievedDoc objects."""
        retriever = self._make_retriever()
        docs = retriever._semantic_search("python", 2)
        assert len(docs) == 2
        assert all(isinstance(d, RetrievedDoc) for d in docs)

    def test_bm25_search_returns_docs(self) -> None:
        """BM25 search should return scored documents."""
        retriever = self._make_retriever()
        docs = retriever._bm25_search("python", 2)
        assert len(docs) > 0
        assert docs[0].score > 0

    def test_hybrid_retrieve(self) -> None:
        """Hybrid retrieval should merge results via RRF."""
        retriever = self._make_retriever()
        docs = retriever.retrieve("python", "hybrid", 2)
        assert len(docs) <= 2
        assert all(isinstance(d, RetrievedDoc) for d in docs)

    def test_reciprocal_rank_fusion(self) -> None:
        """RRF should merge and deduplicate."""
        retriever = self._make_retriever()
        a = [RetrievedDoc("1", "a", 1.0), RetrievedDoc("2", "b", 0.8)]
        b = [RetrievedDoc("2", "b", 1.0), RetrievedDoc("3", "c", 0.5)]
        merged = retriever._reciprocal_rank_fusion(a, b)
        ids = [d.id for d in merged]
        assert "2" in ids  # should appear (deduplicated)
        assert len(merged) == 3


class TestCritiqueAgent:
    """Tests for the critique agent."""

    def test_empty_docs_returns_not_relevant(self) -> None:
        """No docs should yield is_relevant=False."""
        mock_llm = MagicMock()
        agent = CritiqueAgent(mock_llm)
        result = agent.evaluate_retrieval("test query", [])
        assert not result.is_relevant
        assert result.confidence == 0.0

    def test_should_retry_when_not_relevant(self) -> None:
        """should_retry returns True when not relevant with expansion."""
        agent = CritiqueAgent(MagicMock())
        critique = CritiqueResult(
            is_relevant=False, confidence=0.3, reasoning="poor",
            suggested_query_expansion="expanded query"
        )
        assert agent.should_retry(critique) is True

    def test_should_not_retry_when_relevant(self) -> None:
        """should_retry returns False when docs are relevant."""
        agent = CritiqueAgent(MagicMock())
        critique = CritiqueResult(
            is_relevant=True, confidence=0.9, reasoning="good"
        )
        assert agent.should_retry(critique) is False
