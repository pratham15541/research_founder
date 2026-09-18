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
    NVIDIA_MODEL: Optional[str] = Field(
        default=None,
        description="Primary OpenAI-compatible model identifier"
    )
    NVIDIA_BASE_URL: Optional[str] = Field(
        default=None,
        description="Primary OpenAI-compatible API base URL or full /chat/completions URL"
    )
    LLM_PROVIDER_MAX: int = 8

    # Local Directory Paths
    DATA_DIR: Path = BASE_DIR / "data"
    UPLOAD_DIR: Path = BASE_DIR / "data" / "uploads"
    CACHE_DIR: Path = BASE_DIR / "data" / "cache"

    # File storage: "local" or "s3". Use S3 with LocalStack by setting
    # S3_ENDPOINT_URL=http://127.0.0.1:4566 for host runs or http://localstack:4566 in Docker.
    STORAGE_BACKEND: str = Field(default="local", description="local or s3")
    S3_BUCKET: str = Field(default="research-uploads", description="Bucket for uploaded and downloaded PDFs")
    S3_ENDPOINT_URL: Optional[str] = Field(default=None, description="Custom S3 endpoint, e.g. LocalStack")
    S3_REGION: str = Field(default="us-east-1", description="AWS region for S3-compatible storage")
    S3_PREFIX: str = Field(default="papers", description="Object key prefix for stored PDFs")
    STORAGE_FALLBACK_TO_LOCAL: bool = Field(
        default=True,
        description="Fall back to local disk if S3-compatible storage is unreachable"
    )

    # Dynamic LLM behavior. Fallbacks keep the app runnable without an API key, but results
    # include metadata indicating when a non-LLM fallback was used.
    DYNAMIC_LLM_ENABLED: bool = True
    LLM_REQUIRED: bool = False
    LLM_STRUCTURED_TEMPERATURE: float = 0.35
    LLM_CREATIVE_TEMPERATURE: float = 0.75
    LLM_TIMEOUT_SECONDS: float = Field(
        default=90.0,
        description="Timeout budget in seconds for LLM API calls"
    )

    # External paper APIs. Comma-separated sources: openalex,semantic_scholar,arxiv,pubmed,crossref.
    INGESTION_SOURCES: str = "openalex,semantic_scholar,arxiv,pubmed,crossref"
    OPENALEX_BASE_URL: str = "https://api.openalex.org/works"
    SEMANTIC_SCHOLAR_BASE_URL: str = "https://api.semanticscholar.org/graph/v1/paper/search"
    ARXIV_BASE_URL: str = "https://export.arxiv.org/api/query"
    PUBMED_SEARCH_URL: str = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
    PUBMED_SUMMARY_URL: str = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi"
    CROSSREF_BASE_URL: str = "https://api.crossref.org/works"
    UNPAYWALL_BASE_URL: str = "https://api.unpaywall.org/v2"
    OPENALEX_LIMIT: int = 60
    SEMANTIC_SCHOLAR_LIMIT: int = 30
    SEMANTIC_SCHOLAR_RATE_LIMIT_COOLDOWN_SECONDS: int = 900
    ARXIV_LIMIT: int = 30
    PUBMED_LIMIT: int = 30
    CROSSREF_LIMIT: int = 40

    # Proxy rotation and PDF download controls.
    PROXY_SOURCE_URLS: str = (
        "https://raw.githubusercontent.com/iplocate/free-proxy-list/main/protocols/http.txt,"
        "https://raw.githubusercontent.com/iplocate/free-proxy-list/main/protocols/https.txt"
    )
    PROXY_CACHE_TTL_SECONDS: int = 1800
    PROXY_REFRESH_ON_FAILURES: int = 2
    PROXY_USE_DIRECT_FALLBACK: bool = True
    PDF_DOWNLOAD_MAX_RETRIES: int = 4
    PDF_DOWNLOAD_TIMEOUT_SECONDS: float = 25.0
    PDF_DOWNLOAD_DIRECT_FIRST: bool = True

    # Full-text enrichment
    ENABLE_FULL_TEXT_DOWNLOAD: bool = True
    FULL_TEXT_DOWNLOAD_LIMIT: int = 10
    MIN_EXTRACTED_PDF_TEXT_CHARS: int = 300

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
