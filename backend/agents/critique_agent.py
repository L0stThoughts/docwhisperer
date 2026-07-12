"""Self-critique agent that evaluates retrieval relevance.

Determines whether retrieved documents are sufficient to answer the query
and suggests query expansion when confidence is low.
"""
import logging
from dataclasses import dataclass
from typing import Any, List, Optional

from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field as PydanticField

from backend.agents.retriever import RetrievedDoc

logger = logging.getLogger(__name__)


class CritiqueSchema(BaseModel):
    """Pydantic schema for LLM critique output."""

    is_relevant: bool = PydanticField(description="Whether the documents are relevant to the query")
    confidence: float = PydanticField(description="Confidence score between 0.0 and 1.0")
    reasoning: str = PydanticField(description="Brief explanation of the assessment")
    suggested_query_expansion: Optional[str] = PydanticField(
        default=None, description="Suggested expanded query if documents are insufficient"
    )


@dataclass
class CritiqueResult:
    """Result of a retrieval critique evaluation.

    Attributes:
        is_relevant: Whether retrieved docs are relevant enough.
        confidence: Confidence score 0-1.
        reasoning: Explanation of the assessment.
        suggested_query_expansion: Expanded query suggestion if needed.
    """

    is_relevant: bool
    confidence: float
    reasoning: str
    suggested_query_expansion: Optional[str] = None


class CritiqueAgent:
    """Agent that evaluates whether retrieved documents can answer a query.

    Args:
        llm: A LangChain-compatible chat model.
    """

    CONFIDENCE_THRESHOLD: float = 0.6

    def __init__(self, llm: Any) -> None:
        self._llm = llm

    def evaluate_retrieval(self, query: str, docs: List[RetrievedDoc]) -> CritiqueResult:
        """Evaluate whether retrieved documents are relevant to the query.

        Args:
            query: The user query.
            docs: List of retrieved documents.

        Returns:
            CritiqueResult with relevance assessment.
        """
        if not docs:
            return CritiqueResult(
                is_relevant=False,
                confidence=0.0,
                reasoning="No documents were retrieved.",
                suggested_query_expansion=query,
            )

        parser = PydanticOutputParser(pydantic_object=CritiqueSchema)

        context_text = "\n---\n".join(
            f"[Doc {d.id}]: {d.text[:500]}" for d in docs[:10]
        )

        prompt = ChatPromptTemplate.from_messages([
            (
                "system",
                "You are a retrieval quality evaluator. Given a query and retrieved documents, "
                "assess whether the documents contain enough relevant information to answer the query.\n"
                "If confidence is below 0.6, set is_relevant to false and suggest a better query.\n"
                "{format_instructions}",
            ),
            (
                "human",
                "Query: {query}\n\nRetrieved Documents:\n{context}",
            ),
        ])

        try:
            chain = prompt | self._llm | parser
            result: CritiqueSchema = chain.invoke({
                "query": query,
                "context": context_text,
                "format_instructions": parser.get_format_instructions(),
            })

            confidence = max(0.0, min(1.0, result.confidence))
            is_relevant = result.is_relevant and confidence >= self.CONFIDENCE_THRESHOLD
            expansion = result.suggested_query_expansion
            if not is_relevant and not expansion:
                expansion = query

            return CritiqueResult(
                is_relevant=is_relevant,
                confidence=confidence,
                reasoning=result.reasoning,
                suggested_query_expansion=expansion if not is_relevant else None,
            )
        except Exception as exc:
            logger.warning("Critique evaluation failed, assuming relevant: %s", exc)
            return CritiqueResult(
                is_relevant=True,
                confidence=0.5,
                reasoning=f"Critique failed with error: {exc}",
            )

    def should_retry(self, critique: CritiqueResult) -> bool:
        """Determine if retrieval should be retried based on critique.

        Args:
            critique: The critique result to evaluate.

        Returns:
            True if retrieval should be retried.
        """
        return not critique.is_relevant and critique.suggested_query_expansion is not None
