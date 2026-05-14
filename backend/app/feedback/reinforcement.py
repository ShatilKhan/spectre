"""Reinforcement learning from operator edits.

Uses correction pairs to generate few-shot examples that improve
future extraction quality.
"""

from typing import Any

from app.feedback.edit_capture import get_recent_corrections


def get_few_shot_examples(n: int = 3) -> list[dict[str, Any]]:
    """Get recent correction pairs formatted as few-shot examples.

    Returns:
        List of examples showing what the model got wrong and the correct value.
    """
    corrections = get_recent_corrections(n)
    examples = []
    for c in corrections:
        for field in c.changed_fields:
            examples.append(
                {
                    "field": field,
                    "model_said": c.original.get(field),
                    "operator_said": c.corrected.get(field),
                    "document_type": c.document_type,
                }
            )
    return examples


def format_examples_for_prompt(examples: list[dict[str, Any]]) -> str:
    """Format correction examples as a prompt suffix for the LLM.

    Args:
        examples: List of correction examples.

    Returns:
        Formatted string to append to the extraction prompt.
    """
    if not examples:
        return ""

    lines = ["\n## Learning from previous corrections\n"]
    for ex in examples:
        lines.append(
            f"- Field '{ex['field']}': model extracted "
            f"'{ex['model_said']}', operator corrected to "
            f"'{ex['operator_said']}'"
        )
    return "\n".join(lines)
