"""Draft generation with grounded evidence citations."""

import json
from typing import Any

from app.extraction.llm_extractor import get_llm, is_using_gpu


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
1. Document Summary — what type of document, parties
2. Key Terms — fees, payment schedule, term, governing law
3. Risk Flags — any provisions that need attorney review
4. Evidence Table — each claim cites its source passage

Do NOT use placeholder text — if information is missing, omit it.
Use inline citations like [Source: page X] after each claim.
"""


def generate_draft(
    extracted_data: dict[str, Any],
    source_passages: list[dict[str, Any]],
) -> str:
    """Generate a grounded draft memo."""
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

    if is_using_gpu():
        response = llm.chat.completions.create(
            model="granite4.1:3b",
            messages=[
                {"role": "system", "content": "You are a legal memo generator."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.3,
            max_tokens=3000,
        )
        return _strip_date_line(response.choices[0].message.content or "")
    else:
        response = llm.create_chat_completion(
            messages=[
                {"role": "system", "content": "You are a legal memo generator."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.3,
            max_tokens=3000,
        )
        return _strip_date_line(response["choices"][0]["message"]["content"] or "")


def _strip_date_line(text: str) -> str:
    """Remove date lines from the memo header."""
    import re
    # Remove line containing "Date:" in the header section (first 5 lines)
    lines = text.split('\n')
    cleaned = [l for i, l in enumerate(lines) if i > 4 or 'date' not in l.lower()[:10]]
    return '\n'.join(cleaned).strip()


def generate_draft_stream(
    extracted_data: dict[str, Any],
    source_passages: list[dict[str, Any]],
) -> str:
    """Generate a grounded draft memo. Yields text chunks as they arrive."""
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

    if is_using_gpu():
        stream = llm.chat.completions.create(
            model="granite4.1:3b",
            messages=[
                {"role": "system", "content": "You are a legal memo generator."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.3,
            max_tokens=3000,
            stream=True,
        )
        for chunk in stream:
            content = chunk.choices[0].delta.content or ""
            if content:
                yield content
    else:
        for token in llm.create_completion(
            prompt=prompt,
            temperature=0.3,
            max_tokens=3000,
            stream=True,
        ):
            content = token["choices"][0].get("text", "")
            if content:
                yield content
