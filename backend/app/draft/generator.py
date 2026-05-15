"""Draft generation with grounded evidence citations using Granite 4.1."""

import json
import os
from typing import Any

from llama_cpp import Llama

from app.extraction.llm_extractor import get_llm

DRAFT_TEMPLATE = """You are a legal document reviewer. Generate an internal review memo
based on the extracted data and source passages provided below.

The memo must be:
- Grounded in the provided evidence (cite source passages inline)
- Structured as a legal memorandum
- Clear enough for a first-pass review

## Extracted Data
{extracted_data}

## Relevant Source Passages
{source_passages}

## Instructions
Generate a memo with these sections:
1. Document Summary — what type of document, parties, date
2. Key Terms — fees, payment schedule, term, governing law
3. Risk Flags — any provisions that need attorney review
4. Evidence Table — each claim cites its source passage

Use inline citations like [Source: page X] after each claim.
"""


def generate_draft(
    extracted_data: dict[str, Any],
    source_passages: list[dict[str, Any]],
) -> str:
    """Generate a grounded draft memo from extracted data and source passages.

    Args:
        extracted_data: Structured fields extracted from the document.
        source_passages: Relevant passages retrieved from the vector store.

    Returns:
        Generated draft text with inline citations.
    """
    llm = get_llm()

    passages_text = "\n\n".join(
        [
            f"[Page {p.get('metadata', {}).get('page', '?')}] {p['document']}"
            for p in source_passages
        ]
    )

    prompt = DRAFT_TEMPLATE.format(
        extracted_data=json.dumps(extracted_data, indent=2),
        source_passages=passages_text,
    )

    response = llm.create_chat_completion(
        messages=[
            {"role": "system", "content": "You are a legal memo generator."},
            {"role": "user", "content": prompt},
        ],
        temperature=0.3,
        max_tokens=3000,
    )

    return response["choices"][0]["message"]["content"] or ""
