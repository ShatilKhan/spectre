"""LLM-as-judge evaluation using Granite 4.1.

Evaluates extraction quality, context relevance, faithfulness,
and hallucination rate using the LLM itself as the judge.
"""

import json
import os
from typing import Any

JUDGE_PROMPT = """You are an evaluation judge for a legal document extraction system.
Evaluate the quality of the extraction against the ground truth.

## Extracted Data
{extracted}

## Ground Truth
{ground_truth}

Score each metric from 0.0 to 1.0 and return as JSON:
{
  "context_relevance": <float>,
  "answer_faithfulness": <float>,
  "answer_relevance": <float>,
  "hallucination_rate": <float>,
  "explanation": "<brief explanation of scores>"
}
"""


def evaluate_extraction(
    extracted: dict[str, Any],
    ground_truth: dict[str, Any],
) -> dict[str, Any]:
    """Evaluate extraction quality using LLM-as-judge.

    Args:
        extracted: What the system extracted.
        ground_truth: The correct values.

    Returns:
        Dict of evaluation metrics.
    """
    from app.extraction.llm_extractor import get_llm

    llm = get_llm()
    prompt = JUDGE_PROMPT.format(
        extracted=json.dumps(extracted, indent=2),
        ground_truth=json.dumps(ground_truth, indent=2),
    )

    response = llm.create_chat_completion(
        messages=[
            {"role": "system", "content": "You are an evaluation judge. Return only valid JSON."},
            {"role": "user", "content": prompt},
        ],
        response_format={"type": "json_object"},
        temperature=0.0,
        max_tokens=1000,
    )

    content = response["choices"][0]["message"]["content"]
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        return {
            "context_relevance": 0.0,
            "answer_faithfulness": 0.0,
            "answer_relevance": 0.0,
            "hallucination_rate": 1.0,
            "error": "Failed to parse judge output",
        }
