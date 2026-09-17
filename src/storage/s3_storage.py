"""
S3 Storage Backend.
Provides upload/download/list/delete operations against Amazon S3.
Falls back to local filesystem when S3_BUCKET_NAME is not configured.
Routes to LocalStack when USE_LOCALSTACK is enabled.
"""

import logging
from pathlib import Path
from typing import List, Optional

from src.config import settings

logger = logging.getLogger(__name__)


class S3StorageBackend:
    """Manages file storage on S3 with local filesystem fallback."""

    def __init__(self):
        self._client = None
        self._bucket = settings.S3_BUCKET_NAME
        self._use_s3 = settings.is_aws_enabled() and bool(self._bucket)

        if self._use_s3:
            self._init_s3_client()

    def _init_s3_client(self) -> None:
        """Create boto3 S3 client, routing to LocalStack when configured."""
        try:
            import boto3
            kwargs = {
                "service_name": "s3",
                "region_name": settings.AWS_REGION,
            }
            endpoint = settings.get_aws_endpoint_url()
            if endpoint:
                kwargs["endpoint_url"] = endpoint
            if settings.AWS_ACCESS_KEY_ID:
                kwargs["aws_access_key_id"] = settings.AWS_ACCESS_KEY_ID
            if settings.AWS_SECRET_ACCESS_KEY:
                kwargs["aws_secret_access_key"] = settings.AWS_SECRET_ACCESS_KEY
            if settings.AWS_SESSION_TOKEN:
                kwargs["aws_session_token"] = settings.AWS_SESSION_TOKEN

            self._client = boto3.client(**kwargs)
            self._ensure_bucket()
            logger.info(f"S3 client initialized (bucket={self._bucket})")
        except Exception as e:
            logger.warning(f"S3 client initialization failed ({e}). Falling back to local storage.")
            self._use_s3 = False

    def _ensure_bucket(self) -> None:
        """Create the S3 bucket if it does not exist (useful for LocalStack)."""
        try:
            self._client.head_bucket(Bucket=self._bucket)
        except Exception:
            try:
                if settings.AWS_REGION == "us-east-1":
                    self._client.create_bucket(Bucket=self._bucket)
                else:
                    self._client.create_bucket(
                        Bucket=self._bucket,
                        CreateBucketConfiguration={"LocationConstraint": settings.AWS_REGION},
                    )
                logger.info(f"Created S3 bucket: {self._bucket}")
            except Exception as e:
                logger.warning(f"Could not create S3 bucket '{self._bucket}': {e}")

    @property
    def is_s3_active(self) -> bool:
        return self._use_s3 and self._client is not None

    def upload_file(self, local_path: Path, s3_key: Optional[str] = None) -> str:
        """
        Upload a file to S3.
        Returns: S3 URI (s3://bucket/key) or local path string if S3 is off.
        """
        if not s3_key:
            s3_key = f"uploads/{local_path.name}"

        if self.is_s3_active:
            try:
                self._client.upload_file(str(local_path), self._bucket, s3_key)
                uri = f"s3://{self._bucket}/{s3_key}"
                logger.info(f"Uploaded to S3: {uri}")
                return uri
            except Exception as e:
                logger.warning(f"S3 upload failed ({e}). File remains at local path.")

        return str(local_path)

    def download_file(self, s3_key: str, local_path: Path) -> Path:
        """Download a file from S3 to local path."""
        if self.is_s3_active:
            try:
                local_path.parent.mkdir(parents=True, exist_ok=True)
                self._client.download_file(self._bucket, s3_key, str(local_path))
                logger.info(f"Downloaded from S3: s3://{self._bucket}/{s3_key} → {local_path}")
                return local_path
            except Exception as e:
                logger.warning(f"S3 download failed ({e}).")

        return local_path

    def list_files(self, prefix: str = "uploads/") -> List[str]:
        """List all file keys under a prefix."""
        if self.is_s3_active:
            try:
                resp = self._client.list_objects_v2(Bucket=self._bucket, Prefix=prefix)
                return [obj["Key"] for obj in resp.get("Contents", [])]
            except Exception as e:
                logger.warning(f"S3 list_objects failed ({e}).")

        # Local fallback
        local_dir = settings.UPLOAD_DIR
        if local_dir.exists():
            return [str(f.relative_to(local_dir)) for f in local_dir.iterdir() if f.is_file()]
        return []

    def delete_file(self, s3_key: str) -> bool:
        """Delete a file from S3."""
        if self.is_s3_active:
            try:
                self._client.delete_object(Bucket=self._bucket, Key=s3_key)
                logger.info(f"Deleted from S3: s3://{self._bucket}/{s3_key}")
                return True
            except Exception as e:
                logger.warning(f"S3 delete failed ({e}).")
        return False


# Global singleton
storage_backend = S3StorageBackend()
