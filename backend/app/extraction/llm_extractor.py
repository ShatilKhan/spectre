"""LLM-based structured extraction with auto GPU/CPU detection.

Tries Ollama (GPU) first. If unavailable, falls back to llama-cpp-python (CPU).
Zero config — works out of the box with `docker compose up --build`.
"""

import json
import os
from typing import Any

from app.extraction.schemas import get_schema
from app.feedback.reinforcement import format_examples_for_prompt, get_few_shot_examples

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "")
MODEL_PATH = os.getenv("MODEL_PATH", "/models/ibm-granite_granite-4.1-3b-Q4_K_M.gguf")
_llm_instance = None
_using_gpu = None



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

    system_prompt = BASE_SYSTEM_PROMPT.format(doc_type=doc_type)
    if use_few_shot:
        examples = get_few_shot_examples(n=3)
        correction_text = format_examples_for_prompt(examples)
        if correction_text:
            system_prompt += correction_text

    user_prompt = f"Extract fields from this {doc_type}:\n\n{truncate_text(raw_text)}"

    if _using_gpu:
        # Ollama / OpenAI-compatible API (128K context, no truncation needed)
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
        return json.loads(content)
    except json.JSONDecodeError:
        return {"error": "Failed to parse LLM output", "raw_snippet": content[:500]}
