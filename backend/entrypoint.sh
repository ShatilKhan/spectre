#!/bin/bash
set -e

MODEL_PATH="${MODEL_PATH:-/models/ibm-granite_granite-4.1-3b-Q4_K_M.gguf}"
MODEL_DIR="$(dirname "$MODEL_PATH")"

# Start uvicorn immediately so health checks pass
uvicorn app.main:app --host 0.0.0.0 --port 8000 &
UVICORN_PID=$!

# Download model in background if missing
if [ ! -f "$MODEL_PATH" ]; then
    echo "Model not found at $MODEL_PATH"
    echo "Downloading Granite 4.1 3B (2.1 GB) in background..."
    mkdir -p "$MODEL_DIR"
    (
        hf download \
            bartowski/ibm-granite_granite-4.1-3b-GGUF \
            ibm-granite_granite-4.1-3b-Q4_K_M.gguf \
            --local-dir "$MODEL_DIR" 2>&1
        echo "Model download completed."
    ) &
fi

# Wait for uvicorn
wait $UVICORN_PID
