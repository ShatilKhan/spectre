"""Benchmark runner that loads test datasets and evaluates pipeline quality.

Supports CUAD and custom datasets loaded via huggingface datasets library.
"""

from typing import Any

from app.evaluation.metrics import EvaluationResult, compute_metrics


def load_cuad_benchmark(split: str = "test") -> tuple[list[dict], list[dict]]:
    """Load CUAD test set for evaluation.

    Returns:
        Tuple of (ground_truth, predictions) — predictions are empty
        because they need to be computed by running the pipeline.
    """
    try:
        from datasets import load_dataset

        dataset = load_dataset("dvgodoy/CUAD_v1_Contract_Understanding_clause_classification", split=split)
        ground_truth = list(dataset)
        return ground_truth, []
    except ImportError:
        return [], []


def run_benchmark() -> EvaluationResult:
    """Run the full evaluation benchmark.

    Loads test data, runs the extraction pipeline, and computes metrics.
    """
    ground_truth, _ = load_cuad_benchmark()

    if not ground_truth:
        return EvaluationResult(
            num_samples=0,
            errors=["CUAD dataset not available. Run datasets/download.py first."],
        )

    # TODO: run extraction pipeline on each sample
    # predictions = [pipeline.extract(pdf) for pdf in ground_truth]
    predictions = [{} for _ in ground_truth]

    return compute_metrics(ground_truth, predictions)
