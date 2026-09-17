"""
Configuration management using Pydantic Settings.
Loads environment variables securely with validation and fallbacks.
Supports both AWS (Bedrock, S3, OpenSearch) and legacy NVIDIA NGC providers.
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

    # ──────────────────────────────────────────────────────────────────────
    # AWS Configuration (Bedrock, S3, OpenSearch)
    # ──────────────────────────────────────────────────────────────────────
    USE_AWS: bool = Field(
        default=False,
        description="Master toggle: enable AWS services (Bedrock, S3, OpenSearch)"
    )
    USE_LOCALSTACK: bool = Field(
        default=True,
        description="Route all AWS SDK calls to LocalStack (http://localhost:4566)"
    )
    LOCALSTACK_ENDPOINT: str = Field(
        default="http://localhost:4566",
        description="LocalStack gateway URL"
    )
    AWS_REGION: str = Field(
        default="us-east-1",
        description="AWS region for all services"
    )
    AWS_ACCESS_KEY_ID: Optional[str] = Field(
        default="test",
        description="AWS access key (use 'test' for LocalStack)"
    )
    AWS_SECRET_ACCESS_KEY: Optional[str] = Field(
        default="test",
        description="AWS secret key (use 'test' for LocalStack)"
    )
    AWS_SESSION_TOKEN: Optional[str] = Field(
        default=None,
        description="Optional AWS session token for temporary credentials / STS"
    )

    # AWS Bedrock LLM
    AWS_BEDROCK_MODEL_ID: str = Field(
        default="anthropic.claude-3-sonnet-20240229-v1:0",
        description="Primary Bedrock model ID"
    )
    AWS_BEDROCK_FALLBACK_MODEL_ID: str = Field(
        default="amazon.titan-text-premier-v1:0",
        description="Fallback Bedrock model ID"
    )

    # AWS Bedrock Embeddings
    AWS_BEDROCK_EMBEDDING_MODEL_ID: str = Field(
        default="amazon.titan-embed-text-v2:0",
        description="Bedrock embedding model"
    )

    # AWS S3
    S3_BUCKET_NAME: Optional[str] = Field(
        default="research-ai-uploads",
        description="S3 bucket for PDF uploads and cache"
    )

    # AWS OpenSearch Serverless
    OPENSEARCH_ENDPOINT: Optional[str] = Field(
        default=None,
        description="OpenSearch domain endpoint URL (e.g. https://...es.amazonaws.com)"
    )
    OPENSEARCH_INDEX_NAME: str = Field(
        default="research-papers",
        description="OpenSearch index name for vector search"
    )
    OPENSEARCH_USERNAME: Optional[str] = Field(
        default=None,
        description="HTTP Basic Auth username for OpenSearch"
    )
    OPENSEARCH_PASSWORD: Optional[str] = Field(
        default=None,
        description="HTTP Basic Auth password for OpenSearch"
    )

    # ──────────────────────────────────────────────────────────────────────
    # Legacy NVIDIA NGC / OpenRouter Configuration (fallback when AWS is off)
    # ──────────────────────────────────────────────────────────────────────
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

    # ──────────────────────────────────────────────────────────────────────
    # Local Directory Paths
    # ──────────────────────────────────────────────────────────────────────
    DATA_DIR: Path = BASE_DIR / "data"
    UPLOAD_DIR: Path = BASE_DIR / "data" / "uploads"
    CACHE_DIR: Path = BASE_DIR / "data" / "cache"

    # Algorithm & Threshold Defaults
    EMBEDDING_MODEL_NAME: str = "all-MiniLM-L6-v2"
    EMBEDDING_DIMENSION: int = Field(
        default=384,
        description="Embedding vector dimension (384 for MiniLM, 1024 for Titan v2)"
    )
    CLUSTERING_ALGORITHM: str = "auto"
    MIN_CLUSTER_SIZE: int = 5
    SILHOUETTE_GATE_THRESHOLD: float = 0.15
    SPARSITY_PERCENTAGE: float = 0.02
    MAX_UPLOAD_SIZE_MB: int = 15
    ANALYSIS_TIMEOUT_SECONDS: int = Field(
        default=900,
        description="Frontend/API timeout budget for long-running research analysis requests"
    )

    def get_aws_endpoint_url(self) -> Optional[str]:
        """Return the endpoint URL for boto3 clients — LocalStack or None (real AWS)."""
        if self.USE_LOCALSTACK:
            return self.LOCALSTACK_ENDPOINT
        return None

    def is_aws_enabled(self) -> bool:
        """Check if AWS services should be used."""
        return self.USE_AWS

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
