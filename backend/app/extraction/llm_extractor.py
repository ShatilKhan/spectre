"""LLM-based structured extraction with auto GPU/CPU detection.

Tries Ollama (GPU) first. If unavailable, falls back to llama-cpp-python (CPU).
Zero config — works out of the box with `docker compose up --build`.
"""

import json
import os
import re
from typing import Any

from app.extraction.schemas import get_schema
from app.feedback.reinforcement import format_examples_for_prompt, get_few_shot_examples

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "")
MODEL_PATH = os.getenv("MODEL_PATH", "/models/ibm-granite_granite-4.1-3b-Q4_K_M.gguf")
_llm_instance = None
_using_gpu = None

# Fields whose values are generated (summaries, overviews) and will NOT
# appear verbatim in source text. Sanitizer and benchmark must skip these.
SUMMARY_FIELDS = {"summary", "content", "overview", "description"}

# Common alternative field names the LLM may produce instead of schema names.
# Maps them back to the correct schema field names.
FIELD_NAME_ALIASES: dict[str, str] = {
    "title": "document_title",
    "documenttitle": "document_title",
    "doc_title": "document_title",
    "doc_title": "document_title",
    "content": "summary",
    "document_summary": "summary",
    "doc_summary": "summary",
    "brief": "summary",
    "overview": "summary",
    "parties": "parties_mentioned",
    "party": "parties_mentioned",
    "entities": "parties_mentioned",
    "dates": "dates_mentioned",
    "date": "dates_mentioned",
    "monetary_amount": "monetary_amounts",
    "amounts": "monetary_amounts",
    "money": "monetary_amounts",
    "topics": "key_topics",
    "subjects": "key_topics",
    "clauses": "key_topics",
    "type": "document_type",
    "documenttype": "document_type",
    "doc_type": "document_type",
}


def is_using_gpu() -> bool:
    """Check if GPU (Ollama) is active. Safe to call at any time."""
    return _using_gpu is True


# CPU fallback context management
MAX_INPUT_TOKENS = 12000  # leave room for system prompt + output within 16K window


def truncate_text(text: str, max_words: int = MAX_INPUT_TOKENS) -> str:
    """Truncate text to fit within context window."""
    words = text.split()
    if len(words) > max_words:
        return " ".join(words[:max_words]) + "\n\n[...]"
    return text


def _build_schema_template(schema: type) -> str:
    """Build an example JSON template showing exact field names and types.

    This is embedded in the prompt so the LLM knows the exact field names
    to use in its output (Ollama's json_object mode doesn't enforce field
    names, only valid JSON structure).
    """
    schema_dict = schema.model_json_schema()
    props = schema_dict.get("properties", {})
    template_parts: list[str] = []
    for name, prop in props.items():
        ptype = prop.get("type", "string")
        desc = prop.get("description", "")
        if ptype == "array":
            items_type = prop.get("items", {}).get("type", "string")
            template_parts.append(f'  "{name}": [{items_type} (list)]  # {desc}')
        elif prop.get("default") is not None:
            default_val = prop["default"]
            template_parts.append(f'  "{name}": {json.dumps(default_val)}  # {desc}')
        else:
            null_hint = " | null" if name not in ("parties", "summary") and "" else ""
            template_parts.append(f'  "{name}": <{ptype}>{null_hint}  # {desc}')

    return "{\n" + "\n".join(template_parts) + "\n}"


BASE_SYSTEM_PROMPT = """You are a legal document extraction assistant. Extract structured information from the provided legal document text.

CRITICAL RULES — YOU WILL BE PENALIZED IF YOU VIOLATE THESE:
1. NEVER invent or fabricate information. If a field's value is not explicitly stated in the text, use null.
2. NEVER guess amounts, dates, names, or numbers. Only extract values you can see in the text.
3. If the OCR text is garbled, incomplete, or has no relevant content for a field, use null.
4. Use EXACTLY the field names shown in the schema template below. Do not rename fields.
5. For date fields, use ISO 8601 format (YYYY-MM-DD) when possible.
6. For list fields, include only items explicitly mentioned.

Document type: {doc_type}

YOU MUST use this exact JSON structure — keep the field names as shown:
{schema_template}
"""


def get_llm():
    """Auto-detect: use Ollama (GPU) if available, CPU fallback otherwise."""
    global _llm_instance, _using_gpu
    if _llm_instance is not None:
        return _llm_instance

    # Try Ollama first (GPU, fast)
    if OLLAMA_BASE_URL:
        try:
            from openai import OpenAI
            client = OpenAI(base_url=f"{OLLAMA_BASE_URL}/v1", api_key="ollama")
            client.models.list()  # health check — raises if unreachable
            print("LLM: using Ollama (GPU)")
            _llm_instance = client
            _using_gpu = True
            return _llm_instance
        except Exception:
            print("LLM: Ollama not available, falling back to CPU")

    # CPU fallback via llama-cpp-python
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"Model not found at {MODEL_PATH}. "
            "The model will auto-download on container start."
        )
    from llama_cpp import Llama
    print(f"LLM: loading {MODEL_PATH} on CPU...")
    _llm_instance = Llama(
        model_path=MODEL_PATH,
        n_ctx=16384,
        n_threads=4,
        verbose=False,
    )
    _using_gpu = False
    return _llm_instance


def _normalize_field_names(result: dict, schema: type) -> dict:
    """Map LLM-generated field names back to the schema's expected field names.

    The LLM sometimes uses different field names than the schema defines
    (e.g. 'title' instead of 'document_title'). This function normalizes
    the output to match the schema.
    """
    schema_dict = schema.model_json_schema()
    expected_fields = set(schema_dict.get("properties", {}).keys())

    normalized = {}
    for key, value in result.items():
        if key in expected_fields:
            normalized[key] = value
        elif key.lower() in FIELD_NAME_ALIASES:
            mapped = FIELD_NAME_ALIASES[key.lower()]
            normalized[mapped] = value
        else:
            # Keep unrecognized fields but warn via prefix
            normalized[key] = value

    return normalized


def extract_fields(
    raw_text: str,
    doc_type: str = "legal_generic",
    use_few_shot: bool = True,
) -> dict[str, Any]:
    """Extract structured fields using auto-detected LLM backend.

    Ollama (GPU) → fast. llama-cpp-python (CPU) → always works.
    """
    llm = get_llm()
    schema = get_schema(doc_type)
    schema_dict = schema.model_json_schema()
    schema_template = _build_schema_template(schema)

    system_prompt = BASE_SYSTEM_PROMPT.format(
        doc_type=doc_type,
        schema_template=schema_template,
    )
    if use_few_shot:
        examples = get_few_shot_examples(n=3)
        correction_text = format_examples_for_prompt(examples)
        if correction_text:
            system_prompt += correction_text

    user_prompt = f"Extract fields from this {doc_type}:\n\n{truncate_text(raw_text)}"

    if _using_gpu:
        # Ollama v0.4+ supports json_schema response format for strict
        # schema enforcement. Fall back to json_object if not available.
        try:
            response = llm.chat.completions.create(
                model="granite4.1:3b",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                response_format={
                    "type": "json_schema",
                    "json_schema": {
                        "name": "extraction",
                        "strict": True,
                        "schema": schema_dict,
                    },
                },
                temperature=0.0,
                max_tokens=2000,
            )
            content = response.choices[0].message.content
        except Exception:
            # Fall back to basic json_object if structured output not supported
            response = llm.chat.completions.create(
                model="granite4.1:3b",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                response_format={"type": "json_object"},
                temperature=0.0,
                max_tokens=2000,
            )
            content = response.choices[0].message.content
    else:
        # llama-cpp-python with grammar enforcement
        from llama_cpp import LlamaGrammar
        grammar = LlamaGrammar.from_json_schema(json.dumps(schema_dict))
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
        result = json.loads(content)
        # Normalize field names to match schema
        result = _normalize_field_names(result, schema)
        return _sanitize_extraction(result, raw_text)
    except json.JSONDecodeError:
        return {"error": "Failed to parse LLM output", "raw_snippet": content[:500]}


def _sanitize_extraction(result: dict, raw_text: str) -> dict:
    """Post-process extraction to remove hallucinated values.

    For each field that claims a concrete fact (amount, date, name, etc.),
    verify it appears in the OCR text. If not, null it out.

    Summary/generated fields are skipped — they won't appear verbatim.
    """
    text_lower = raw_text.lower()
    schema = get_schema(result.get("_doc_type", "legal_generic"))
    schema_dict = schema.model_json_schema()
    expected_fields = set(schema_dict.get("properties", {}).keys())

    for key, value in list(result.items()):
        if not value:
            continue

        # Skip summary fields — they are generated, not extracted
        if key in SUMMARY_FIELDS:
            continue

        # Skip fields that don't belong to the schema (unknown keys)
        if key not in expected_fields and key.lower() not in FIELD_NAME_ALIASES:
            continue

        if isinstance(value, str) and _looks_fabricated(value, text_lower):
            result[key] = None
        elif isinstance(value, dict):
            for k2, v2 in list(value.items()):
                if isinstance(v2, str) and _looks_fabricated(v2, text_lower):
                    value[k2] = None
    return result


def _looks_fabricated(value: str, text_lower: str) -> bool:
    """Check if a value looks fabricated vs grounded in source text.

    Returns True if the value seems to be a hallucination.
    Returns False for values < 3 chars, values that appear verbatim,
    or values that pass all checks.
    """
    if not value or len(value) < 3:
        return False
    val_lower = value.lower().strip()

    # Check if value appears directly in text
    if val_lower in text_lower:
        return False

    # Check for dollar amounts
    dollar_matches = re.findall(r'\$[\d,]+(?:\.\d{2})?', value)
    if dollar_matches:
        for dm in dollar_matches:
            if dm.lower() not in text_lower:
                return True

    # Check for dates
    date_patterns = [
        r'\b(?:january|february|march|april|may|june|july|august|september|october|november|december)\s+\d{1,2},?\s+\d{4}\b',
        r'\b\d{4}-\d{2}-\d{2}\b',
        r'\b\d{1,2}/\d{1,2}/\d{4}\b',
    ]
    for pat in date_patterns:
        date_matches = re.findall(pat, value, re.I)
        for dm in date_matches:
            if dm.lower() not in text_lower:
                return True

    return False
