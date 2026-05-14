#!/bin/bash
set -e

MODEL_PATH="${MODEL_PATH:-/models/granite-4.1-3b-Q4_K_M.gguf}"
MODEL_DIR="$(dirname "$MODEL_PATH")"

# Auto-download model if not present
if [ ! -f "$MODEL_PATH" ]; then
    echo "Model not found at $MODEL_PATH"
    echo "Downloading Granite 4.1 3B (2.1 GB)..."
    mkdir -p "$MODEL_DIR"
    pip install -q huggingface-hub
    huggingface-cli download \
        bartowski/ibm-granite_granite-4.1-3b-GGUF \
        --include "ibm-granite_granite-4.1-3b-Q4_K_M.gguf" \
        --local-dir "$MODEL_DIR"
    echo "Model downloaded successfully."
fi

# Start the app
exec uvicorn app.main:app --host 0.0.0.0 --port 8000
