"""Tests for the DocWhisperer FastAPI endpoints."""
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from fastapi.testclient import TestClient


class TestHealthEndpoint:
    """Tests for the /health endpoint."""

    def test_health_returns_ok(self) -> None:
        """Health endpoint should return status ok."""
        from backend.main import app
        client = TestClient(app)
        resp = client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"
        assert "version" in data

    def test_health_includes_service_name(self) -> None:
        """Health endpoint should include service name."""
        from backend.main import app
        client = TestClient(app)
        resp = client.get("/health")
        assert resp.json()["service"] == "docwhisperer"


class TestQueryEndpoint:
    """Tests for the /query endpoint."""

    def test_query_requires_body(self) -> None:
        """POST /query without body should return 422."""
        from backend.main import app
        client = TestClient(app)
        resp = client.post("/query")
        assert resp.status_code == 422

    def test_query_with_mock_pipeline(self) -> None:
        """POST /query should call pipeline and return response."""
        from backend.main import app
        from backend.pipeline.graph import PipelineResult
        import backend.main as main_module

        mock_result = PipelineResult(
            answer="Test answer",
            sources=["doc1"],
            intent="factual_lookup",
            retrieval_strategy="hybrid",
            critique_passed=True,
            iterations=1,
        )

        mock_pipeline = MagicMock()
        mock_pipeline.run = AsyncMock(return_value=mock_result)
        original_pipeline = main_module.pipeline
        main_module.pipeline = mock_pipeline

        try:
            client = TestClient(app)
            resp = client.post("/query", json={"query": "What is X?"})
            assert resp.status_code == 200
            data = resp.json()
            assert data["answer"] == "Test answer"
            assert data["query"] == "What is X?"
        finally:
            main_module.pipeline = original_pipeline


class TestIngestEndpoint:
    """Tests for the /ingest endpoint."""

    def test_ingest_requires_path(self) -> None:
        """POST /ingest without path should return 422."""
        from backend.main import app
        client = TestClient(app)
        resp = client.post("/ingest", json={})
        assert resp.status_code == 422
