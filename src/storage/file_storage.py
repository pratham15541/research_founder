"""
File storage backends for uploaded and downloaded PDFs.

The parser still needs a local file path, so the S3 backend writes a local
cache copy first and then uploads the same bytes to S3-compatible storage.
"""

import logging
import mimetypes
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from src.config import settings

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class StoredFile:
    backend: str
    local_path: Path
    uri: str
    key: Optional[str] = None
    bucket: Optional[str] = None


class FileStorageBackend:
    """Abstract-ish storage interface used by API uploads and PDF downloads."""

    backend_name = "base"

    def save_bytes(
        self,
        content: bytes,
        filename: str,
        content_type: Optional[str] = None,
        file_id: Optional[str] = None
    ) -> StoredFile:
        raise NotImplementedError


class LocalFileStorageBackend(FileStorageBackend):
    backend_name = "local"

    def save_bytes(
        self,
        content: bytes,
        filename: str,
        content_type: Optional[str] = None,
        file_id: Optional[str] = None
    ) -> StoredFile:
        settings.UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
        suffix = Path(filename).suffix or ".bin"
        safe_id = file_id or Path(filename).stem
        local_path = settings.UPLOAD_DIR / f"{safe_id}{suffix}"
        local_path.write_bytes(content)
        return StoredFile(
            backend=self.backend_name,
            local_path=local_path,
            uri=str(local_path)
        )


class S3FileStorageBackend(FileStorageBackend):
    backend_name = "s3"

    def __init__(self):
        try:
            import boto3
            from botocore.config import Config
        except ImportError as exc:
            raise RuntimeError("boto3 is required for STORAGE_BACKEND=s3") from exc

        self._boto3 = boto3
        self._client = boto3.client(
            "s3",
            endpoint_url=settings.S3_ENDPOINT_URL,
            region_name=settings.S3_REGION,
            aws_access_key_id="test" if settings.S3_ENDPOINT_URL else None,
            aws_secret_access_key="test" if settings.S3_ENDPOINT_URL else None,
            config=Config(s3={"addressing_style": "path"})
        )
        self._ensure_bucket()

    def _ensure_bucket(self) -> None:
        try:
            self._client.head_bucket(Bucket=settings.S3_BUCKET)
        except Exception:
            create_kwargs = {"Bucket": settings.S3_BUCKET}
            if settings.S3_REGION != "us-east-1":
                create_kwargs["CreateBucketConfiguration"] = {
                    "LocationConstraint": settings.S3_REGION
                }
            self._client.create_bucket(**create_kwargs)

    def save_bytes(
        self,
        content: bytes,
        filename: str,
        content_type: Optional[str] = None,
        file_id: Optional[str] = None
    ) -> StoredFile:
        local = LocalFileStorageBackend().save_bytes(content, filename, content_type, file_id)
        guessed_type = content_type or mimetypes.guess_type(filename)[0] or "application/octet-stream"
        key = f"{settings.S3_PREFIX.strip('/')}/{local.local_path.name}"

        try:
            self._client.put_object(
                Bucket=settings.S3_BUCKET,
                Key=key,
                Body=content,
                ContentType=guessed_type
            )
        except Exception:
            if settings.STORAGE_FALLBACK_TO_LOCAL:
                logger.warning(
                    "S3 storage unavailable at %s; kept file locally at %s",
                    settings.S3_ENDPOINT_URL,
                    local.local_path
                )
                return local
            raise

        return StoredFile(
            backend=self.backend_name,
            local_path=local.local_path,
            uri=f"s3://{settings.S3_BUCKET}/{key}",
            key=key,
            bucket=settings.S3_BUCKET
        )


def get_file_storage_backend() -> FileStorageBackend:
    backend = settings.STORAGE_BACKEND.strip().lower()
    if backend == "s3":
        try:
            return S3FileStorageBackend()
        except Exception as exc:
            if settings.STORAGE_FALLBACK_TO_LOCAL:
                logger.warning(
                    "S3 storage backend unavailable at %s; using local storage instead (%s)",
                    settings.S3_ENDPOINT_URL,
                    exc
                )
                return LocalFileStorageBackend()
            raise
    if backend == "local":
        return LocalFileStorageBackend()
    logger.warning("Unknown STORAGE_BACKEND=%s; falling back to local storage.", settings.STORAGE_BACKEND)
    return LocalFileStorageBackend()
