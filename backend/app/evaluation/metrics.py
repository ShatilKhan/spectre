"""Nine applicable evaluation metrics for the document extraction pipeline.

Metrics 8-10 (tool selection, execution success, multi-step coherence)
are excluded as they apply to multi-tool agent systems, not RAG pipelines.
"""

from dataclasses import dataclass, field
from typing import Any


@dataclass
class EvaluationResult:
    """Results from a full evaluation run.

    Metrics 8-10 are excluded as they apply to multi-tool agent
    systems, not RAG pipelines. See docs/assumptions.md for details.
    """

    # Retrieval metrics
    context_relevance: float = 0.0
    context_recall: float = 0.0
    context_precision: float = 0.0
    retrieval_latency_ms: float = 0.0

    # Generation metrics
    answer_faithfulness: float = 0.0
    answer_relevance: float = 0.0
    hallucination_rate: float = 0.0

    # Production metrics
    cost_per_query: float = 0.0
    p99_latency_ms: float = 0.0

    # Metadata
    num_samples: int = 0
    errors: list[str] = field(default_factory=list)


def compute_metrics(
    ground_truth: list[dict[str, Any]],
    predictions: list[dict[str, Any]],
) -> EvaluationResult:
    """Compute evaluation metrics from ground truth and predictions.

    Args:
        ground_truth: List of ground truth label sets.
        predictions: List of system predictions.

    Returns:
        EvaluationResult with all applicable metrics populated.
    """
    # This is a stub — actual computation will be implemented
    # in the benchmark runner with the labeled dataset.
    return EvaluationResult(
        num_samples=len(ground_truth),
    )
