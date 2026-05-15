"""LLM-based structured extraction using Granite 4.1 via llama-cpp-python."""

import json
import os
from typing import Any

from llama_cpp import Llama, LlamaGrammar

from app.extraction.schemas import get_schema
from app.feedback.reinforcement import format_examples_for_prompt, get_few_shot_examples

MODEL_PATH = os.getenv("MODEL_PATH", "/models/granite-4.1-3b-Q4_K_M.gguf")
_llm = None

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


def get_llm() -> Llama:
    """Get or create the shared Llama instance (loaded once, reused)."""
    global _llm
    if _llm is None:
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(
                f"Model not found at {MODEL_PATH}. "
                "The model will auto-download on container start."
            )
        _llm = Llama(
            model_path=MODEL_PATH,
            n_ctx=8192,
            n_threads=4,
            verbose=False,
        )
    return _llm


def extract_fields(
    raw_text: str,
    doc_type: str = "legal_generic",
    use_few_shot: bool = True,
) -> dict[str, Any]:
    """Extract structured fields from OCR text using Granite 4.1.

    Uses LlamaGrammar.from_json_schema() for token-level JSON enforcement
    — the model literally cannot output invalid JSON.

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
    grammar = LlamaGrammar.from_json_schema(json.dumps(schema_dict))

    system_prompt = BASE_SYSTEM_PROMPT.format(doc_type=doc_type)

    if use_few_shot:
        examples = get_few_shot_examples(n=3)
        correction_text = format_examples_for_prompt(examples)
        if correction_text:
            system_prompt += correction_text

    user_prompt = f"Extract fields from this {doc_type}:\n\n{raw_text}"

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
    if not content:
        return {"error": "Empty response from LLM"}

    try:
        return json.loads(content)
    except json.JSONDecodeError:
        return {"error": "Failed to parse LLM output", "raw_snippet": content[:500]}
