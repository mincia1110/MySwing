"""S3/MinIO client wrapper for video file storage operations."""

from uuid import uuid4

import boto3
from botocore.exceptions import ClientError

from app.core.config import settings


class S3Client:
    """Wrapper around boto3 S3 client for MinIO/S3 operations."""

    def __init__(self) -> None:
        """Initialize the S3 client with settings from config."""
        self._client = boto3.client(
            "s3",
            endpoint_url=settings.s3_endpoint_url,
            aws_access_key_id=settings.s3_access_key,
            aws_secret_access_key=settings.s3_secret_key,
            region_name=settings.s3_region,
        )
        self._bucket = settings.s3_bucket_name

    def generate_file_key(self, original_filename: str) -> str:
        """Generate a unique S3 file key using UUID.

        Format: uploads/{uuid4}/{original_filename}
        """
        return f"uploads/{uuid4()}/{original_filename}"

    def generate_presigned_upload_url(
        self,
        file_key: str,
        content_type: str,
        expires_in: int = 3600,
    ) -> str:
        """Generate a presigned URL for uploading a file to S3.

        Args:
            file_key: The S3 object key.
            content_type: The MIME type of the file.
            expires_in: URL expiration time in seconds (default: 3600).

        Returns:
            The presigned upload URL string.
        """
        url: str = self._client.generate_presigned_url(
            ClientMethod="put_object",
            Params={
                "Bucket": self._bucket,
                "Key": file_key,
                "ContentType": content_type,
            },
            ExpiresIn=expires_in,
        )
        return url

    def generate_presigned_download_url(
        self,
        file_key: str,
        expires_in: int = 3600,
    ) -> str:
        """Generate a presigned URL for downloading a file from S3.

        Args:
            file_key: The S3 object key.
            expires_in: URL expiration time in seconds (default: 3600).

        Returns:
            The presigned download URL string.
        """
        url: str = self._client.generate_presigned_url(
            ClientMethod="get_object",
            Params={
                "Bucket": self._bucket,
                "Key": file_key,
            },
            ExpiresIn=expires_in,
        )
        return url

    def check_file_exists(self, file_key: str) -> bool:
        """Check if a file exists in the S3 bucket.

        Args:
            file_key: The S3 object key to check.

        Returns:
            True if the file exists, False otherwise.
        """
        try:
            self._client.head_object(Bucket=self._bucket, Key=file_key)
            return True
        except ClientError:
            return False

    def delete_file(self, file_key: str) -> None:
        """Delete a file from the S3 bucket.

        Args:
            file_key: The S3 object key to delete.
        """
        self._client.delete_object(Bucket=self._bucket, Key=file_key)


def get_s3_client() -> S3Client:
    """Factory function to create an S3Client instance."""
    return S3Client()
