"""Draft generation with grounded evidence citations."""

import json
import os

from llama_cpp import Llama
from openai import OpenAI

LLM_MODE = os.getenv("LLM_MODE", "local")
_llm = None

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
    extracted_data: dict,
    source_passages: list[dict],
) -> str:
    """Generate a grounded draft memo from extracted data and source passages.

    Args:
        extracted_data: Structured fields extracted from the document.
        source_passages: Relevant passages retrieved from the vector store.

    Returns:
        Generated draft text with inline citations.
    """
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

    if LLM_MODE == "local":
        from app.extraction.llm_extractor import get_llm

        llm = get_llm()
        response = llm.create_chat_completion(
            messages=[
                {"role": "system", "content": "You are a legal memo generator."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.3,
            max_tokens=3000,
        )
        return response["choices"][0]["message"]["content"]
    else:
        client = OpenAI(
            base_url="https://api.groq.com/openai/v1",
            api_key=os.getenv("GROQ_API_KEY", ""),
        )
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": "You are a legal memo generator."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.3,
            max_tokens=3000,
        )
        return response.choices[0].message.content
