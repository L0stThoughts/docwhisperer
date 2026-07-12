"""Hybrid retriever combining BM25 lexical search with ChromaDB semantic search.

Uses Reciprocal Rank Fusion to merge results from both retrieval methods.
"""
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from rank_bm25 import BM25Okapi

logger = logging.getLogger(__name__)


@dataclass
class RetrievedDoc:
    """A single retrieved document chunk.

    Attributes:
        id: Unique document/chunk identifier.
        text: The text content of the chunk.
        score: Relevance score (higher is better).
        metadata: Arbitrary metadata from the source document.
    """

    id: str
    text: str
    score: float
    metadata: Dict[str, Any] = field(default_factory=dict)


class HybridRetriever:
    """Hybrid retriever that combines BM25 and semantic vector search.

    Args:
        chroma_collection: A ChromaDB collection instance.
        bm25_index: A BM25Okapi index built from the corpus.
        embedder: An embedding function compatible with ChromaDB queries.
        corpus_docs: List of dicts with 'id', 'text', 'metadata' for BM25 mapping.
    """

    def __init__(
        self,
        chroma_collection: Any,
        bm25_index: Optional[BM25Okapi],
        embedder: Any,
        corpus_docs: Optional[List[Dict[str, Any]]] = None,
    ) -> None:
        self._collection = chroma_collection
        self._bm25 = bm25_index
        self._embedder = embedder
        self._corpus_docs = corpus_docs or []

    def retrieve(self, query: str, strategy: str, top_k: int) -> List[RetrievedDoc]:
        """Retrieve documents using the specified strategy.

        Args:
            query: The search query.
            strategy: One of 'semantic_heavy', 'bm25_heavy', 'hybrid'.
            top_k: Number of results to return.

        Returns:
            Sorted list of RetrievedDoc instances.
        """
        try:
            if strategy == "semantic_heavy":
                semantic = self._semantic_search(query, top_k)
                bm25 = self._bm25_search(query, max(1, top_k // 3)) if self._bm25 else []
                return self._reciprocal_rank_fusion(semantic, bm25)[:top_k]
            elif strategy == "bm25_heavy":
                bm25 = self._bm25_search(query, top_k) if self._bm25 else []
                semantic = self._semantic_search(query, max(1, top_k // 3))
                return self._reciprocal_rank_fusion(bm25, semantic)[:top_k]
            else:  # hybrid
                semantic = self._semantic_search(query, top_k)
                bm25 = self._bm25_search(query, top_k) if self._bm25 else []
                return self._reciprocal_rank_fusion(semantic, bm25)[:top_k]
        except Exception as exc:
            logger.error("Retrieval failed: %s", exc)
            return []

    def _semantic_search(self, query: str, top_k: int) -> List[RetrievedDoc]:
        """Run semantic search via ChromaDB.

        Args:
            query: The search query.
            top_k: Number of results.

        Returns:
            List of RetrievedDoc from semantic search.
        """
        try:
            embedding = self._embedder.embed_query(query)
            results = self._collection.query(
                query_embeddings=[embedding],
                n_results=top_k,
                include=["documents", "metadatas", "distances"],
            )
            docs: List[RetrievedDoc] = []
            if results and results.get("ids"):
                for i, doc_id in enumerate(results["ids"][0]):
                    distance = results["distances"][0][i] if results.get("distances") else 0.0
                    score = 1.0 / (1.0 + distance)
                    docs.append(RetrievedDoc(
                        id=doc_id,
                        text=results["documents"][0][i] if results.get("documents") else "",
                        score=score,
                        metadata=results["metadatas"][0][i] if results.get("metadatas") else {},
                    ))
            return docs
        except Exception as exc:
            logger.error("Semantic search failed: %s", exc)
            return []

    def _bm25_search(self, query: str, top_k: int) -> List[RetrievedDoc]:
        """Run BM25 lexical search.

        Args:
            query: The search query.
            top_k: Number of results.

        Returns:
            List of RetrievedDoc from BM25 search.
        """
        if not self._bm25 or not self._corpus_docs:
            return []
        try:
            tokenized_query = query.lower().split()
            scores = self._bm25.get_scores(tokenized_query)
            top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
            docs: List[RetrievedDoc] = []
            for idx in top_indices:
                if scores[idx] <= 0:
                    continue
                doc = self._corpus_docs[idx]
                docs.append(RetrievedDoc(
                    id=doc.get("id", str(idx)),
                    text=doc.get("text", ""),
                    score=float(scores[idx]),
                    metadata=doc.get("metadata", {}),
                ))
            return docs
        except Exception as exc:
            logger.error("BM25 search failed: %s", exc)
            return []

    def _reciprocal_rank_fusion(
        self, results_a: List[RetrievedDoc], results_b: List[RetrievedDoc], k: int = 60
    ) -> List[RetrievedDoc]:
        """Merge two ranked lists using Reciprocal Rank Fusion.

        Args:
            results_a: First ranked list (primary).
            results_b: Second ranked list (secondary).
            k: RRF constant (default 60).

        Returns:
            Merged and re-ranked list of RetrievedDoc.
        """
        fused_scores: Dict[str, float] = {}
        doc_map: Dict[str, RetrievedDoc] = {}

        for rank, doc in enumerate(results_a):
            fused_scores[doc.id] = fused_scores.get(doc.id, 0.0) + 1.0 / (k + rank + 1)
            doc_map[doc.id] = doc

        for rank, doc in enumerate(results_b):
            fused_scores[doc.id] = fused_scores.get(doc.id, 0.0) + 1.0 / (k + rank + 1)
            if doc.id not in doc_map:
                doc_map[doc.id] = doc

        sorted_ids = sorted(fused_scores, key=lambda x: fused_scores[x], reverse=True)
        return [
            RetrievedDoc(
                id=doc_map[did].id,
                text=doc_map[did].text,
                score=fused_scores[did],
                metadata=doc_map[did].metadata,
            )
            for did in sorted_ids
        ]
