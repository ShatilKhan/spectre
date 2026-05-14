"""LLM-based structured extraction from OCR text.

Uses llama-cpp-python with Granite 4.1 for local inference,
or Groq API as an optional cloud fallback. JSON schema is enforced
at the token level via GBNF grammars when using local mode.
"""

import json
import os
from typing import Any, Optional
from pydantic import BaseModel

from app.extraction.schemas import get_schema
from app.feedback.reinforcement import format_examples_for_prompt, get_few_shot_examples

LLM_MODE = os.getenv("LLM_MODE", "local")
_llm_instance = None

BASE_SYSTEM_PROMPT = """You are a legal document extraction assistant. Extract structured information from the provided legal document text.

Rules:
1. Return only valid JSON matching the requested schema.
2. If a field cannot be determined from the text, use null.
3. Do not invent or fabricate information not present in the text.
4. For date fields, use ISO 8601 format (YYYY-MM-DD) when possible.
5. For list fields, include all items mentioned in the text.
6. Extract text verbatim where possible rather than paraphrasing.

Document type: {doc_type}
"""


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
        from openai import OpenAI

        _llm_instance = OpenAI(
            base_url="https://api.groq.com/openai/v1",
            api_key=os.getenv("GROQ_API_KEY", ""),
        )
    return _llm_instance


def extract_fields(
    raw_text: str,
    doc_type: str = "legal_generic",
    use_few_shot: bool = True,
) -> dict[str, Any]:
    """Extract structured fields from OCR text using the LLM.

    Args:
        raw_text: OCR-extracted text from the document.
        doc_type: Classified document type for schema selection.
        use_few_shot: Whether to include correction examples in the prompt.

    Returns:
        Dictionary of extracted fields matching the schema for doc_type.
    """
    llm = get_llm()
    schema = get_schema(doc_type)
    schema_dict = schema.model_json_schema()

    # Build system prompt with doc type and optional corrections
    system_prompt = BASE_SYSTEM_PROMPT.format(doc_type=doc_type)

    if use_few_shot:
        examples = get_few_shot_examples(n=3)
        correction_text = format_examples_for_prompt(examples)
        if correction_text:
            system_prompt += correction_text

    user_prompt = f"Extract fields from this {doc_type}:\n\n{raw_text}"

    if LLM_MODE == "local":
        from llama_cpp import LlamaGrammar

        grammar = LlamaGrammar.from_json_schema(schema_dict)

        response = llm.create_chat_completion(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            grammar=grammar,
            temperature=0.0,
            max_tokens=2000,
        )
        content = response["choices"][0]["message"]["content"]
    else:
        response = llm.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            response_format={"type": "json_object"},
            temperature=0.0,
        )
        content = response.choices[0].message.content

    try:
        parsed = json.loads(content)
        return parsed
    except json.JSONDecodeError:
        return {"error": "Failed to parse LLM output", "raw_snippet": content[:500]}
