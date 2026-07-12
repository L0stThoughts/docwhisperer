"""RAGAS evaluation module for DocWhisperer.

Evaluates RAG pipeline outputs using faithfulness, answer relevancy,
context precision, and context recall metrics.
"""
import json
import logging
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from datasets import Dataset
from ragas import evaluate
from ragas.metrics import (
    answer_relevancy,
    context_precision,
    context_recall,
    faithfulness,
)

from backend.config import Settings

logger = logging.getLogger(__name__)


@dataclass
class SingleEvalResult:
    """Evaluation result for a single query.

    Attributes:
        faithfulness: Faithfulness score 0-1.
        answer_relevancy: Answer relevancy score 0-1.
        context_precision: Context precision score 0-1.
        context_recall: Context recall score 0-1.
        timestamp: ISO timestamp of evaluation.
    """

    faithfulness: float = 0.0
    answer_relevancy: float = 0.0
    context_precision: float = 0.0
    context_recall: float = 0.0
    timestamp: str = ""


@dataclass
class EvalResult:
    """Aggregated evaluation result for a batch.

    Attributes:
        faithfulness: Mean faithfulness score.
        answer_relevancy: Mean answer relevancy score.
        context_precision: Mean context precision score.
        context_recall: Mean context recall score.
        timestamp: ISO timestamp of evaluation.
        per_query: Individual results per query.
    """

    faithfulness: float = 0.0
    answer_relevancy: float = 0.0
    context_precision: float = 0.0
    context_recall: float = 0.0
    timestamp: str = ""
    per_query: List[SingleEvalResult] = field(default_factory=list)


class RAGASEvaluator:
    """Evaluates RAG pipeline outputs using RAGAS metrics.

    Args:
        llm: A LangChain-compatible LLM for evaluation.
        embeddings: An embeddings model for semantic metrics.
    """

    def __init__(self, llm: Any, embeddings: Any) -> None:
        self._llm = llm
        self._embeddings = embeddings

    def evaluate_batch(
        self,
        queries: List[str],
        answers: List[str],
        contexts: List[List[str]],
        ground_truths: Optional[List[str]] = None,
    ) -> EvalResult:
        """Evaluate a batch of query-answer pairs.

        Args:
            queries: List of user queries.
            answers: List of generated answers.
            contexts: List of context document lists (one per query).
            ground_truths: Optional list of ground truth answers.

        Returns:
            EvalResult with aggregated and per-query scores.
        """
        timestamp = datetime.now(timezone.utc).isoformat()

        data: Dict[str, List[Any]] = {
            "question": queries,
            "answer": answers,
            "contexts": contexts,
        }
        metrics_list = [faithfulness, answer_relevancy, context_precision]

        if ground_truths:
            data["ground_truth"] = ground_truths
            metrics_list.append(context_recall)

        try:
            dataset = Dataset.from_dict(data)
            result = evaluate(
                dataset,
                metrics=metrics_list,
                llm=self._llm,
                embeddings=self._embeddings,
            )

            scores = result.to_pandas()

            eval_result = EvalResult(
                faithfulness=float(scores.get("faithfulness", [0.0]).mean()),
                answer_relevancy=float(scores.get("answer_relevancy", [0.0]).mean()),
                context_precision=float(scores.get("context_precision", [0.0]).mean()),
                context_recall=float(scores.get("context_recall", [0.0]).mean()) if ground_truths else 0.0,
                timestamp=timestamp,
            )

            for i in range(len(queries)):
                eval_result.per_query.append(SingleEvalResult(
                    faithfulness=float(scores["faithfulness"].iloc[i]) if "faithfulness" in scores else 0.0,
                    answer_relevancy=float(scores["answer_relevancy"].iloc[i]) if "answer_relevancy" in scores else 0.0,
                    context_precision=float(scores["context_precision"].iloc[i]) if "context_precision" in scores else 0.0,
                    context_recall=float(scores["context_recall"].iloc[i]) if "context_recall" in scores and ground_truths else 0.0,
                    timestamp=timestamp,
                ))

            return eval_result
        except Exception as exc:
            logger.error("RAGAS batch evaluation failed: %s", exc)
            return EvalResult(timestamp=timestamp)

    def evaluate_single(
        self, query: str, answer: str, contexts: List[str]
    ) -> SingleEvalResult:
        """Evaluate a single query-answer pair.

        Args:
            query: The user query.
            answer: The generated answer.
            contexts: List of context document texts.

        Returns:
            SingleEvalResult with scores.
        """
        result = self.evaluate_batch([query], [answer], [contexts])
        if result.per_query:
            return result.per_query[0]
        return SingleEvalResult(timestamp=datetime.now(timezone.utc).isoformat())

    def save_results(self, result: EvalResult, output_path: str) -> None:
        """Save evaluation results to a JSON file.

        Args:
            result: The EvalResult to save.
            output_path: File path for the JSON output.
        """
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

        data = {
            "faithfulness": result.faithfulness,
            "answer_relevancy": result.answer_relevancy,
            "context_precision": result.context_precision,
            "context_recall": result.context_recall,
            "timestamp": result.timestamp,
            "per_query": [
                {
                    "faithfulness": r.faithfulness,
                    "answer_relevancy": r.answer_relevancy,
                    "context_precision": r.context_precision,
                    "context_recall": r.context_recall,
                    "timestamp": r.timestamp,
                }
                for r in result.per_query
            ],
        }

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        logger.info("Evaluation results saved to %s", output_path)
