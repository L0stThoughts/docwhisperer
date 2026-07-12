"""DocWhisperer FastAPI application.

Exposes REST endpoints for querying the RAG pipeline, ingesting documents,
running health checks, and triggering RAGAS evaluations.
"""
import logging
from collections import deque
from contextlib import asynccontextmanager
from dataclasses import asdict
from typing import Any, Deque, Dict, List, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from backend.config import Settings
from backend.ingestion.ingest import DocumentIngester
from backend.models import QueryRequest, QueryResponse, SourceItem
from backend.pipeline.graph import DocWhispererPipeline

logger = logging.getLogger(__name__)

settings = Settings()

pipeline: Optional[DocWhispererPipeline] = None
ingester: Optional[DocumentIngester] = None
query_history: Deque[Dict[str, Any]] = deque(maxlen=50)


class IngestRequest(BaseModel):
    """Request body for document ingestion."""

    path: str = Field(..., description="Directory path containing documents to ingest")


class IngestResponse(BaseModel):
    """Response for ingestion endpoint."""

    files_processed: int
    chunks_created: int
    errors: List[str]


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize pipeline and ingester on startup."""
    global pipeline, ingester

    logger.info("Initializing DocWhisperer pipeline and ingester...")
    ingester = DocumentIngester(settings)
    pipeline = DocWhispererPipeline(settings)

    # Try to load existing BM25 index
    import os
    bm25_path = os.path.join(settings.chroma_db_dir, "bm25_index.pkl")
    if os.path.exists(bm25_path):
        try:
            from backend.agents.retriever import HybridRetriever
            bm25 = ingester.load_bm25_index(bm25_path)
            retriever = HybridRetriever(
                chroma_collection=ingester.collection,
                bm25_index=bm25,
                embedder=ingester.embedder,
                corpus_docs=ingester.get_corpus_docs(),
            )
            pipeline.set_retriever(retriever)
            logger.info("Loaded existing BM25 index and configured retriever")
        except Exception as exc:
            logger.warning("Failed to load BM25 index: %s", exc)

    logger.info("DocWhisperer ready")
    yield
    logger.info("DocWhisperer shutting down")


app = FastAPI(title="DocWhisperer API", version="1.0.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
async def health() -> Dict[str, str]:
    """Health check endpoint.

    Returns:
        Dict with status and version.
    """
    return {"status": "ok", "version": "1.0.0", "service": "docwhisperer"}


@app.post("/query", response_model=QueryResponse)
async def query_docs(request: QueryRequest) -> QueryResponse:
    """Execute a query against the RAG pipeline.

    Args:
        request: QueryRequest with the user's question.

    Returns:
        QueryResponse with answer and sources.
    """
    if not pipeline:
        raise HTTPException(status_code=503, detail="Pipeline not initialized")

    result = await pipeline.run(request.query)

    sources = [
        SourceItem(id=sid, score=0.0, text=None, metadata=None)
        for sid in result.sources
    ]

    query_history.append({
        "query": request.query,
        "answer": result.answer,
        "sources": result.sources,
        "intent": result.intent,
    })

    return QueryResponse(
        query=request.query,
        answer=result.answer,
        sources=sources,
    )


@app.post("/ingest", response_model=IngestResponse)
async def ingest_docs(request: IngestRequest) -> IngestResponse:
    """Ingest documents from a directory into the vector store.

    Args:
        request: IngestRequest with directory path.

    Returns:
        IngestResponse with ingestion statistics.
    """
    if not ingester:
        raise HTTPException(status_code=503, detail="Ingester not initialized")

    import os
    result = ingester.ingest_directory(request.path)

    # Save BM25 index after ingestion
    bm25_path = os.path.join(settings.chroma_db_dir, "bm25_index.pkl")
    try:
        ingester.save_bm25_index(bm25_path)
        # Update pipeline retriever
        if pipeline:
            from backend.agents.retriever import HybridRetriever
            bm25 = ingester.load_bm25_index(bm25_path)
            retriever = HybridRetriever(
                chroma_collection=ingester.collection,
                bm25_index=bm25,
                embedder=ingester.embedder,
                corpus_docs=ingester.get_corpus_docs(),
            )
            pipeline.set_retriever(retriever)
    except Exception as exc:
        logger.warning("Failed to save/reload BM25 index: %s", exc)
        result.errors.append(f"BM25 index save failed: {exc}")

    return IngestResponse(
        files_processed=result.files_processed,
        chunks_created=result.chunks_created,
        errors=result.errors,
    )


@app.get("/eval")
async def run_eval() -> Dict[str, Any]:
    """Run a quick RAGAS evaluation on recent queries.

    Returns:
        Dict with evaluation metrics or a message if insufficient data.
    """
    if len(query_history) < 1:
        return {"message": "No queries in history to evaluate"}

    try:
        from backend.evaluation.ragas_eval import RAGASEvaluator

        if not settings.openai_api_key:
            return {"message": "RAGAS evaluation requires OPENAI_API_KEY"}

        from langchain_openai import ChatOpenAI, OpenAIEmbeddings

        eval_llm = ChatOpenAI(model="gpt-4o-mini", openai_api_key=settings.openai_api_key)
        eval_embeddings = OpenAIEmbeddings(model="text-embedding-3-small", openai_api_key=settings.openai_api_key)

        evaluator = RAGASEvaluator(eval_llm, eval_embeddings)

        recent = list(query_history)[-5:]
        queries = [q["query"] for q in recent]
        answers = [q["answer"] for q in recent]
        contexts = [[q["answer"]] for q in recent]  # use answer as pseudo-context

        result = evaluator.evaluate_batch(queries, answers, contexts)
        evaluator.save_results(result, settings.ragas_eval_output)

        return {
            "faithfulness": result.faithfulness,
            "answer_relevancy": result.answer_relevancy,
            "context_precision": result.context_precision,
            "evaluated_queries": len(recent),
        }
    except Exception as exc:
        logger.error("Evaluation failed: %s", exc)
        raise HTTPException(status_code=500, detail=f"Evaluation failed: {exc}")
