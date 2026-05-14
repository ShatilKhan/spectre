"""LLM-based structured extraction from OCR text.

Uses llama-cpp-python with Granite 4.1 for local inference,
or Groq API as an optional cloud fallback.
"""

import json
import os
from typing import Any

from app.models import ProposalExtraction

LLM_MODE = os.getenv("LLM_MODE", "local")
_llm_instance = None

SYSTEM_PROMPT = """You are a legal document extraction assistant.
Extract structured information from the provided legal document text.
Return only valid JSON matching the requested schema.
If a field cannot be determined, use null.
Do not invent or fabricate information not present in the text."""


def get_llm():
    """Get or create the LLM instance (local or API-based)."""
    global _llm_instance
    if _llm_instance is not None:
        return _llm_instance

    if LLM_MODE == "local":
        from llama_cpp import Llama

        model_path = os.getenv("MODEL_PATH", "/models/granite-4.1-3b-Q4_K_M.gguf")
        _llm_instance = Llama(
            model_path=model_path,
            n_ctx=4096,
            n_threads=4,
            verbose=False,
        )
    elif LLM_MODE == "groq":
        # Use OpenAI-compatible client for Groq
        from openai import OpenAI

        _llm_instance = OpenAI(
            base_url="https://api.groq.com/openai/v1",
            api_key=os.getenv("GROQ_API_KEY", ""),
        )
    return _llm_instance


def extract_fields(raw_text: str) -> dict[str, Any]:
    """Extract structured fields from OCR text using the LLM.

    Args:
        raw_text: OCR-extracted text from the document.

    Returns:
        Dictionary of extracted fields with confidence metadata.
    """
    llm = get_llm()

    extraction_schema = ProposalExtraction.model_json_schema()

    if LLM_MODE == "local":
        response = llm.create_chat_completion(
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": f"Extract fields from this legal document:\n\n{raw_text}",
                },
            ],
            response_format={
                "type": "json_object",
                "schema": extraction_schema,
            },
            temperature=0.0,
            max_tokens=2000,
        )
        content = response["choices"][0]["message"]["content"]
    else:
        response = llm.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": f"Extract fields from this legal document:\n\n{raw_text}",
                },
            ],
            response_format={"type": "json_object"},
            temperature=0.0,
        )
        content = response.choices[0].message.content

    try:
        return json.loads(content)
    except json.JSONDecodeError:
        return {"raw_text": raw_text, "error": "Failed to parse LLM output"}
