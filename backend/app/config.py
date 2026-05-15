"""Application configuration from environment variables."""

import os
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class Settings:
    """Application settings loaded from environment."""

    # LLM
    llm_mode: str = field(default_factory=lambda: os.getenv("LLM_MODE", "local"))
    model_path: str = field(
        default_factory=lambda: os.getenv(
            "MODEL_PATH", "/models/ibm-granite_granite-4.1-3b-Q4_K_M.gguf"
        )
    )

    # Database
    database_url: str = field(
        default_factory=lambda: os.getenv(
            "DATABASE_URL", "sqlite:////home/user/app/data/spectre.db"
        )
    )

    # Vector store
    chroma_persist_dir: str = field(
        default_factory=lambda: os.getenv(
            "CHROMA_PERSIST_DIR", "/home/user/app/data/chroma"
        )
    )

    # Paths (relative to container working dir /home/user/app)
    base_dir: Path = Path("/home/user/app")
    upload_dir: Path = field(default_factory=lambda: Path("/home/user/app/uploads"))
    data_dir: Path = field(default_factory=lambda: Path("/home/user/app/data"))

    def __post_init__(self):
        self.upload_dir.mkdir(parents=True, exist_ok=True)
        self.data_dir.mkdir(parents=True, exist_ok=True)


settings = Settings()
