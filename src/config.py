"""
Configuration management using Pydantic Settings.
Loads environment variables securely with validation and fallbacks.
"""

from pathlib import Path
from typing import Optional
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field

BASE_DIR = Path(__file__).resolve().parent.parent

class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=str(BASE_DIR / ".env"),
        env_file_encoding="utf-8",
        extra="ignore"
    )

    # Database Configuration (PostgreSQL 16 with pgvector or SQLite fallback)
    DATABASE_URL: str = Field(
        default="postgresql+asyncpg://postgres:postgres_secure_pass@127.0.0.1:5432/research_db",
        description="Async database connection URL"
    )
    POSTGRES_USER: str = "postgres"
    POSTGRES_PASSWORD: str = "postgres_secure_pass"
    POSTGRES_DB: str = "research_db"

    # API Keys & Third-Party Configuration
    OPENALEX_EMAIL: str = Field(
        default="researcher@example.edu",
        description="Polite pool identification email for OpenAlex (10 req/sec)"
    )
    S2_API_KEY: Optional[str] = Field(
        default=None,
        description="Semantic Scholar API Key (optional)"
    )

    # NVIDIA NGC AI Foundation API Configuration
    NVIDIA_API_KEY: Optional[str] = Field(
        default=None,
        description="NVIDIA NGC API Key for LLM classification, chat, and synthesis"
    )
    NVIDIA_MODEL: str = Field(
        default="moonshotai/kimi-k3",
        description="NVIDIA model identifier (e.g. moonshotai/kimi-k3, meta/llama-3.2-11b-vision-instruct)"
    )
    NVIDIA_BASE_URL: str = Field(
        default="https://integrate.api.nvidia.com/v1",
        description="NVIDIA API base URL"
    )

    # Local Directory Paths
    DATA_DIR: Path = BASE_DIR / "data"
    UPLOAD_DIR: Path = BASE_DIR / "data" / "uploads"
    CACHE_DIR: Path = BASE_DIR / "data" / "cache"

    # Algorithm & Threshold Defaults
    EMBEDDING_MODEL_NAME: str = "all-MiniLM-L6-v2"
    CLUSTERING_ALGORITHM: str = "auto"
    MIN_CLUSTER_SIZE: int = 5
    SILHOUETTE_GATE_THRESHOLD: float = 0.15
    SPARSITY_PERCENTAGE: float = 0.02
    MAX_UPLOAD_SIZE_MB: int = 15
    ANALYSIS_TIMEOUT_SECONDS: int = Field(
        default=900,
        description="Frontend/API timeout budget for long-running research analysis requests"
    )

    def ensure_directories(self) -> None:
        """Ensure necessary local storage directories exist and load key fallbacks."""
        self.UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
        self.CACHE_DIR.mkdir(parents=True, exist_ok=True)

        # Fallback inspection if NVIDIA_API_KEY was passed as raw line in .env
        if not self.NVIDIA_API_KEY:
            env_path = BASE_DIR / ".env"
            if env_path.exists():
                try:
                    for line in env_path.read_text(encoding="utf-8").splitlines():
                        line = line.strip()
                        if line.startswith("nvapi-"):
                            self.NVIDIA_API_KEY = line
                            break
                        elif line.startswith("NVIDIA_API_KEY="):
                            self.NVIDIA_API_KEY = line.split("=", 1)[1].strip()
                            break
                except Exception:
                    pass

settings = Settings()
settings.ensure_directories()
