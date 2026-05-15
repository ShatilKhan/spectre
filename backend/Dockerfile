FROM python:3.13-slim

# Copy uv binary from official distroless image
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# Install runtime system deps for PaddleOCR + Tesseract
RUN apt-get update && apt-get install -y \
    tesseract-ocr \
    tesseract-ocr-eng \
    libgl1 \
    libglib2.0-0 \
    poppler-utils \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Python deps using uv (run as root to write to /usr/local)
WORKDIR /app
COPY pyproject.toml .
RUN uv pip install --system --no-cache \
    --find-links https://abetlen.github.io/llama-cpp-python/whl/cpu/llama-cpp-python/ \
    -e ".[dev]"

# Create non-root user and set up app
RUN useradd -m -u 1000 user
USER user
WORKDIR /home/user/app

# Copy app code
COPY --chown=user . .

# PaddlePaddle: disable PIR executor (bug in 3.3.x, safe with 3.2.x)
ENV FLAGS_enable_pir_api=0
# Skip model source connectivity check on every startup
ENV PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK=True

# Entrypoint: auto-download model if missing, then start
ENTRYPOINT ["/bin/bash", "entrypoint.sh"]

HEALTHCHECK --interval=30s --timeout=10s --retries=5 \
    CMD curl -f http://localhost:8000/health || exit 1
