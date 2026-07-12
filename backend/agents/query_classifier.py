"""Query classifier agent for DocWhisperer.

Classifies incoming queries into intents and determines retrieval strategy
using an LLM with structured Pydantic output parsing.
"""
import logging
from dataclasses import dataclass, field
from typing import Any

from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field as PydanticField

logger = logging.getLogger(__name__)

VALID_INTENTS = ("factual_lookup", "howto_procedure", "comparison", "debugging", "general")
VALID_STRATEGIES = ("semantic_heavy", "bm25_heavy", "hybrid")

INTENT_STRATEGY_MAP: dict[str, str] = {
    "factual_lookup": "semantic_heavy",
    "howto_procedure": "hybrid",
    "comparison": "hybrid",
    "debugging": "bm25_heavy",
    "general": "hybrid",
}

INTENT_TOPK_MAP: dict[str, int] = {
    "factual_lookup": 3,
    "howto_procedure": 5,
    "comparison": 6,
    "debugging": 5,
    "general": 5,
}


class QueryClassificationSchema(BaseModel):
    """Pydantic schema for LLM structured output."""

    intent: str = PydanticField(description="One of: factual_lookup, howto_procedure, comparison, debugging, general")
    confidence: float = PydanticField(description="Confidence score between 0.0 and 1.0")


@dataclass
class QueryClassification:
    """Result of query classification.

    Attributes:
        intent: The classified intent label.
        confidence: Confidence score 0-1.
        retrieval_strategy: Suggested retrieval strategy.
        suggested_top_k: Suggested number of documents to retrieve.
    """

    intent: str
    confidence: float
    retrieval_strategy: str
    suggested_top_k: int


def _default_classification() -> QueryClassification:
    """Return a safe fallback classification."""
    return QueryClassification(
        intent="general",
        confidence=0.0,
        retrieval_strategy="hybrid",
        suggested_top_k=5,
    )


def classify_query(query: str, llm: Any) -> QueryClassification:
    """Classify a user query into an intent with retrieval strategy.

    Args:
        query: The user's natural language question.
        llm: A LangChain-compatible chat model instance.

    Returns:
        A QueryClassification with intent, confidence, strategy, and top_k.
    """
    parser = PydanticOutputParser(pydantic_object=QueryClassificationSchema)

    prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            "You are a query classification engine. Classify the user query into exactly one intent.\n"
            "Valid intents: factual_lookup, howto_procedure, comparison, debugging, general.\n"
            "{format_instructions}",
        ),
        ("human", "{query}"),
    ])

    try:
        chain = prompt | llm | parser
        result: QueryClassificationSchema = chain.invoke({
            "query": query,
            "format_instructions": parser.get_format_instructions(),
        })

        intent = result.intent if result.intent in VALID_INTENTS else "general"
        confidence = max(0.0, min(1.0, result.confidence))

        return QueryClassification(
            intent=intent,
            confidence=confidence,
            retrieval_strategy=INTENT_STRATEGY_MAP.get(intent, "hybrid"),
            suggested_top_k=INTENT_TOPK_MAP.get(intent, 5),
        )
    except Exception as exc:
        logger.warning("Query classification failed, using fallback: %s", exc)
        return _default_classification()
