"""LangGraph pipeline wiring all DocWhisperer agents together.

Implements classify → retrieve → critique → (retry) → generate flow.
"""
import logging
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, TypedDict

from langgraph.graph import END, StateGraph

from backend.agents.critique_agent import CritiqueAgent, CritiqueResult
from backend.agents.query_classifier import QueryClassification, classify_query
from backend.agents.response_generator import GeneratedResponse, ResponseGenerator
from backend.agents.retriever import HybridRetriever, RetrievedDoc
from backend.config import Settings

logger = logging.getLogger(__name__)

MAX_CRITIQUE_RETRIES = 2


@dataclass
class PipelineResult:
    """Final result from the DocWhisperer pipeline.

    Attributes:
        answer: The generated answer.
        sources: List of source document IDs.
        intent: Classified query intent.
        retrieval_strategy: Strategy used for retrieval.
        critique_passed: Whether critique evaluation passed.
        iterations: Number of retrieve-critique iterations.
    """

    answer: str
    sources: List[str] = field(default_factory=list)
    intent: str = ""
    retrieval_strategy: str = ""
    critique_passed: bool = False
    iterations: int = 0


class PipelineState(TypedDict, total=False):
    """State passed between pipeline nodes."""

    query: str
    chat_history: List[Any]
    classification: Optional[Dict[str, Any]]
    docs: List[Dict[str, Any]]
    critique: Optional[Dict[str, Any]]
    iteration: int
    response: Optional[Dict[str, Any]]
    expanded_query: Optional[str]


class DocWhispererPipeline:
    """Orchestrates the full RAG pipeline using LangGraph.

    Args:
        settings: Application settings.
        retriever: Optional pre-built HybridRetriever (for testing/reuse).
    """

    def __init__(self, settings: Settings, retriever: Optional[HybridRetriever] = None) -> None:
        self._settings = settings
        self._llm = self._create_llm()
        self._retriever = retriever
        self._classifier_fn = classify_query
        self._critique_agent = CritiqueAgent(self._llm)
        self._response_generator = ResponseGenerator(self._llm)
        self._graph = self.build()

    def _create_llm(self) -> Any:
        """Create the LLM instance, preferring OpenAI, falling back to Ollama.

        Returns:
            A LangChain-compatible chat model.
        """
        if self._settings.openai_api_key:
            from langchain_openai import ChatOpenAI

            return ChatOpenAI(
                model="gpt-4o-mini",
                openai_api_key=self._settings.openai_api_key,
                temperature=0.1,
            )
        else:
            from langchain_community.llms import Ollama

            logger.warning("No OPENAI_API_KEY set, falling back to Ollama")
            return Ollama(
                base_url=self._settings.ollama_url or "http://localhost:11434",
                model="llama3",
            )

    def set_retriever(self, retriever: HybridRetriever) -> None:
        """Set or update the hybrid retriever.

        Args:
            retriever: A configured HybridRetriever instance.
        """
        self._retriever = retriever

    def build(self) -> Any:
        """Build and compile the LangGraph state graph.

        Returns:
            A compiled LangGraph graph.
        """
        graph = StateGraph(PipelineState)

        graph.add_node("classify", self._classify_node)
        graph.add_node("retrieve", self._retrieve_node)
        graph.add_node("critique", self._critique_node)
        graph.add_node("generate", self._generate_node)

        graph.set_entry_point("classify")
        graph.add_edge("classify", "retrieve")
        graph.add_edge("retrieve", "critique")
        graph.add_conditional_edges(
            "critique",
            self._should_retry,
            {"retry": "retrieve", "proceed": "generate"},
        )
        graph.add_edge("generate", END)

        return graph.compile()

    def _classify_node(self, state: PipelineState) -> Dict[str, Any]:
        """Classify the query intent.

        Args:
            state: Current pipeline state.

        Returns:
            Updated state with classification.
        """
        query = state["query"]
        classification = self._classifier_fn(query, self._llm)
        logger.info("Classified query as %s (confidence=%.2f)", classification.intent, classification.confidence)
        return {
            "classification": {
                "intent": classification.intent,
                "confidence": classification.confidence,
                "retrieval_strategy": classification.retrieval_strategy,
                "suggested_top_k": classification.suggested_top_k,
            }
        }

    def _retrieve_node(self, state: PipelineState) -> Dict[str, Any]:
        """Retrieve documents based on classification.

        Args:
            state: Current pipeline state.

        Returns:
            Updated state with retrieved docs.
        """
        if not self._retriever:
            logger.warning("No retriever configured, returning empty results")
            return {"docs": [], "iteration": state.get("iteration", 0) + 1}

        classification = state.get("classification", {})
        strategy = classification.get("retrieval_strategy", "hybrid")
        top_k = classification.get("suggested_top_k", 5)
        query = state.get("expanded_query") or state["query"]

        docs = self._retriever.retrieve(query, strategy, top_k)
        logger.info("Retrieved %d documents", len(docs))
        return {
            "docs": [{"id": d.id, "text": d.text, "score": d.score, "metadata": d.metadata} for d in docs],
            "iteration": state.get("iteration", 0) + 1,
            "expanded_query": None,
        }

    def _critique_node(self, state: PipelineState) -> Dict[str, Any]:
        """Critique the retrieval quality.

        Args:
            state: Current pipeline state.

        Returns:
            Updated state with critique result.
        """
        docs = [
            RetrievedDoc(id=d["id"], text=d["text"], score=d["score"], metadata=d.get("metadata", {}))
            for d in state.get("docs", [])
        ]
        critique = self._critique_agent.evaluate_retrieval(state["query"], docs)
        logger.info("Critique: relevant=%s confidence=%.2f", critique.is_relevant, critique.confidence)
        return {
            "critique": {
                "is_relevant": critique.is_relevant,
                "confidence": critique.confidence,
                "reasoning": critique.reasoning,
                "suggested_query_expansion": critique.suggested_query_expansion,
            },
            "expanded_query": critique.suggested_query_expansion,
        }

    def _should_retry(self, state: PipelineState) -> str:
        """Decide whether to retry retrieval or proceed to generation.

        Args:
            state: Current pipeline state.

        Returns:
            'retry' or 'proceed'.
        """
        critique = state.get("critique", {})
        iteration = state.get("iteration", 0)
        if not critique.get("is_relevant") and iteration < MAX_CRITIQUE_RETRIES and critique.get("suggested_query_expansion"):
            logger.info("Retrying retrieval (iteration %d)", iteration)
            return "retry"
        return "proceed"

    def _generate_node(self, state: PipelineState) -> Dict[str, Any]:
        """Generate the final response.

        Args:
            state: Current pipeline state.

        Returns:
            Updated state with generated response.
        """
        docs = [
            RetrievedDoc(id=d["id"], text=d["text"], score=d["score"], metadata=d.get("metadata", {}))
            for d in state.get("docs", [])
        ]
        chat_history = state.get("chat_history", [])
        response = self._response_generator.generate(state["query"], docs, chat_history)
        return {
            "response": {
                "answer": response.answer,
                "cited_sources": response.cited_sources,
                "token_usage": response.token_usage,
            }
        }

    async def run(self, query: str, chat_history: List[Any] = None) -> PipelineResult:
        """Execute the full pipeline for a query.

        Args:
            query: The user's question.
            chat_history: Optional conversation history.

        Returns:
            PipelineResult with answer, sources, and metadata.
        """
        if chat_history is None:
            chat_history = []

        initial_state: PipelineState = {
            "query": query,
            "chat_history": chat_history,
            "classification": None,
            "docs": [],
            "critique": None,
            "iteration": 0,
            "response": None,
            "expanded_query": None,
        }

        try:
            final_state = await self._graph.ainvoke(initial_state)

            response = final_state.get("response", {})
            classification = final_state.get("classification", {})
            critique = final_state.get("critique", {})

            return PipelineResult(
                answer=response.get("answer", "No answer generated"),
                sources=response.get("cited_sources", []),
                intent=classification.get("intent", ""),
                retrieval_strategy=classification.get("retrieval_strategy", ""),
                critique_passed=critique.get("is_relevant", False),
                iterations=final_state.get("iteration", 0),
            )
        except Exception as exc:
            logger.error("Pipeline execution failed: %s", exc)
            return PipelineResult(answer=f"Pipeline error: {exc}")
