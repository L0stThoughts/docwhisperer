"""Response generator agent for DocWhisperer.

Generates final answers from retrieved documents with source citations.
"""
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate

from backend.agents.retriever import RetrievedDoc

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = (
    "You are a precise technical documentation assistant. "
    "Answer ONLY from the provided context. If the answer is not in the context, "
    "say so explicitly — do not guess or hallucinate.\n\n"
    "When referencing information, cite the source document ID inline using [DocID] notation.\n"
    "Be concise, accurate, and well-structured."
)


@dataclass
class GeneratedResponse:
    """Generated answer with provenance.

    Attributes:
        answer: The generated answer text.
        cited_sources: List of document IDs cited in the answer.
        token_usage: Token usage statistics from the LLM call.
    """

    answer: str
    cited_sources: List[str] = field(default_factory=list)
    token_usage: Dict[str, int] = field(default_factory=dict)


class ResponseGenerator:
    """Generates answers from retrieved documents using an LLM.

    Args:
        llm: A LangChain-compatible chat model.
    """

    def __init__(self, llm: Any) -> None:
        self._llm = llm

    def generate(
        self,
        query: str,
        docs: List[RetrievedDoc],
        chat_history: List[Any] = None,
    ) -> GeneratedResponse:
        """Generate an answer for the query using retrieved documents as context.

        Args:
            query: The user's question.
            docs: Retrieved documents to use as context.
            chat_history: Optional list of prior messages for conversational context.

        Returns:
            GeneratedResponse with answer, citations, and token usage.
        """
        if chat_history is None:
            chat_history = []

        context_parts = []
        source_ids = []
        for doc in docs:
            context_parts.append(f"[{doc.id}]: {doc.text}")
            source_ids.append(doc.id)

        context_text = "\n---\n".join(context_parts) if context_parts else "(No documents retrieved)"

        messages: List[Any] = [SystemMessage(content=SYSTEM_PROMPT)]

        for msg in chat_history:
            if isinstance(msg, dict):
                role = msg.get("role", "human")
                content = msg.get("content", "")
                if role == "assistant":
                    messages.append(AIMessage(content=content))
                else:
                    messages.append(HumanMessage(content=content))
            else:
                messages.append(msg)

        messages.append(HumanMessage(
            content=f"Context:\n{context_text}\n\nQuestion: {query}"
        ))

        try:
            result = self._llm.invoke(messages)

            token_usage: Dict[str, int] = {}
            if hasattr(result, "response_metadata"):
                usage = result.response_metadata.get("token_usage", {})
                if usage:
                    token_usage = {
                        "prompt_tokens": usage.get("prompt_tokens", 0),
                        "completion_tokens": usage.get("completion_tokens", 0),
                        "total_tokens": usage.get("total_tokens", 0),
                    }

            answer_text = result.content if hasattr(result, "content") else str(result)

            cited = [sid for sid in source_ids if sid in answer_text]

            return GeneratedResponse(
                answer=answer_text,
                cited_sources=cited if cited else source_ids[:3],
                token_usage=token_usage,
            )
        except Exception as exc:
            logger.error("Response generation failed: %s", exc)
            return GeneratedResponse(
                answer=f"Generation failed: {exc}",
                cited_sources=[],
                token_usage={},
            )
