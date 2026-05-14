"""Application configuration from environment variables."""

import os
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class Settings:
    """Application settings loaded from environment."""

    # LLM
    llm_mode: str = field(default_factory=lambda: os.getenv("LLM_MODE", "local"))
    groq_api_key: str = field(default_factory=lambda: os.getenv("GROQ_API_KEY", ""))
    model_path: str = field(
        default_factory=lambda: os.getenv(
            "MODEL_PATH", "/models/granite-4.1-3b-Q4_K_M.gguf"
        )
    )

    # Database
    database_url: str = field(
        default_factory=lambda: os.getenv(
            "DATABASE_URL", "sqlite:///app/data/spectre.db"
        )
    )

    # Vector store
    chroma_persist_dir: str = field(
        default_factory=lambda: os.getenv("CHROMA_PERSIST_DIR", "/app/data/chroma")
    )

    # Telemetry
    otlp_endpoint: str = field(
        default_factory=lambda: os.getenv("OTLP_ENDPOINT", "http://otel-collector:4317")
    )

    # Paths
    upload_dir: Path = Path("/app/uploads")
    data_dir: Path = Path("/app/data")

    def __post_init__(self):
        self.upload_dir.mkdir(parents=True, exist_ok=True)
        self.data_dir.mkdir(parents=True, exist_ok=True)


settings = Settings()
