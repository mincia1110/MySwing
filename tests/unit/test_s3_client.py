"""Unit tests for S3 client wrapper and presigned URL endpoint."""

import re
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient


class TestFileKeyGeneration:
    """Tests for S3 file key generation strategy."""

    def test_file_key_format_contains_uploads_prefix(self) -> None:
        """File key starts with 'uploads/' prefix."""
        with patch("app.services.s3_client.boto3"):
            from app.services.s3_client import S3Client

            client = S3Client()
            key = client.generate_file_key("test_video.mp4")
            assert key.startswith("uploads/")

    def test_file_key_contains_uuid(self) -> None:
        """File key contains a valid UUID4 segment."""
        uuid_pattern = re.compile(
            r"uploads/[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}/"
        )
        with patch("app.services.s3_client.boto3"):
            from app.services.s3_client import S3Client

            client = S3Client()
            key = client.generate_file_key("swing.mp4")
            assert uuid_pattern.match(key) is not None

    def test_file_key_preserves_original_filename(self) -> None:
        """File key ends with the original filename."""
        with patch("app.services.s3_client.boto3"):
            from app.services.s3_client import S3Client

            client = S3Client()
            key = client.generate_file_key("my_swing_video.mov")
            assert key.endswith("/my_swing_video.mov")

    def test_file_key_uniqueness(self) -> None:
        """Each call generates a unique file key."""
        with patch("app.services.s3_client.boto3"):
            from app.services.s3_client import S3Client

            client = S3Client()
            key1 = client.generate_file_key("video.mp4")
            key2 = client.generate_file_key("video.mp4")
            assert key1 != key2


class TestPresignedUrlGeneration:
    """Tests for presigned URL generation methods."""

    @patch("app.services.s3_client.boto3")
    def test_generate_presigned_upload_url(self, mock_boto3: MagicMock) -> None:
        """Presigned upload URL is generated with correct parameters."""
        mock_client = MagicMock()
        mock_boto3.client.return_value = mock_client
        mock_client.generate_presigned_url.return_value = "https://minio:9000/presigned"

        from app.services.s3_client import S3Client

        client = S3Client()
        url = client.generate_presigned_upload_url(
            file_key="uploads/abc/video.mp4",
            content_type="video/mp4",
            expires_in=3600,
        )

        assert url == "https://minio:9000/presigned"
        mock_client.generate_presigned_url.assert_called_once_with(
            ClientMethod="put_object",
            Params={
                "Bucket": "myswing-videos",
                "Key": "uploads/abc/video.mp4",
                "ContentType": "video/mp4",
            },
            ExpiresIn=3600,
        )

    @patch("app.services.s3_client.boto3")
    def test_generate_presigned_download_url(self, mock_boto3: MagicMock) -> None:
        """Presigned download URL is generated with correct parameters."""
        mock_client = MagicMock()
        mock_boto3.client.return_value = mock_client
        mock_client.generate_presigned_url.return_value = "https://minio:9000/download"

        from app.services.s3_client import S3Client

        client = S3Client()
        url = client.generate_presigned_download_url(
            file_key="uploads/abc/video.mp4",
            expires_in=1800,
        )

        assert url == "https://minio:9000/download"
        mock_client.generate_presigned_url.assert_called_once_with(
            ClientMethod="get_object",
            Params={
                "Bucket": "myswing-videos",
                "Key": "uploads/abc/video.mp4",
            },
            ExpiresIn=1800,
        )

    @patch("app.services.s3_client.boto3")
    def test_check_file_exists_returns_true(self, mock_boto3: MagicMock) -> None:
        """check_file_exists returns True when file exists."""
        mock_client = MagicMock()
        mock_boto3.client.return_value = mock_client
        mock_client.head_object.return_value = {}

        from app.services.s3_client import S3Client

        client = S3Client()
        assert client.check_file_exists("uploads/abc/video.mp4") is True

    @patch("app.services.s3_client.boto3")
    def test_check_file_exists_returns_false(self, mock_boto3: MagicMock) -> None:
        """check_file_exists returns False when file does not exist."""
        from botocore.exceptions import ClientError

        mock_client = MagicMock()
        mock_boto3.client.return_value = mock_client
        mock_client.head_object.side_effect = ClientError(
            {"Error": {"Code": "404", "Message": "Not Found"}},
            "HeadObject",
        )

        from app.services.s3_client import S3Client

        client = S3Client()
        assert client.check_file_exists("uploads/abc/nonexistent.mp4") is False

    @patch("app.services.s3_client.boto3")
    def test_delete_file(self, mock_boto3: MagicMock) -> None:
        """delete_file calls S3 delete_object with correct params."""
        mock_client = MagicMock()
        mock_boto3.client.return_value = mock_client

        from app.services.s3_client import S3Client

        client = S3Client()
        client.delete_file("uploads/abc/video.mp4")

        mock_client.delete_object.assert_called_once_with(
            Bucket="myswing-videos",
            Key="uploads/abc/video.mp4",
        )


class TestPresignedUrlEndpoint:
    """Tests for POST /api/v1/upload/presigned-url endpoint."""

    @patch("app.api.upload.get_s3_client")
    def test_valid_mp4_request(
        self, mock_get_client: MagicMock, client: TestClient
    ) -> None:
        """Valid MP4 request returns presigned URL and file key."""
        mock_s3 = MagicMock()
        mock_get_client.return_value = mock_s3
        mock_s3.generate_file_key.return_value = "uploads/uuid/video.mp4"
        mock_s3.generate_presigned_upload_url.return_value = "https://s3/presigned"

        response = client.post(
            "/api/v1/upload/presigned-url",
            json={"file_name": "video.mp4", "content_type": "video/mp4"},
        )

        assert response.status_code == 200
        data = response.json()
        assert data["upload_url"] == "https://s3/presigned"
        assert data["file_key"] == "uploads/uuid/video.mp4"
        assert data["expires_in"] == 3600

    @patch("app.api.upload.get_s3_client")
    def test_valid_quicktime_request(
        self, mock_get_client: MagicMock, client: TestClient
    ) -> None:
        """Valid QuickTime (MOV) request returns presigned URL."""
        mock_s3 = MagicMock()
        mock_get_client.return_value = mock_s3
        mock_s3.generate_file_key.return_value = "uploads/uuid/swing.mov"
        mock_s3.generate_presigned_upload_url.return_value = "https://s3/presigned"

        response = client.post(
            "/api/v1/upload/presigned-url",
            json={"file_name": "swing.mov", "content_type": "video/quicktime"},
        )

        assert response.status_code == 200

    @patch("app.api.upload.get_s3_client")
    def test_valid_avi_request(
        self, mock_get_client: MagicMock, client: TestClient
    ) -> None:
        """Valid AVI request returns presigned URL."""
        mock_s3 = MagicMock()
        mock_get_client.return_value = mock_s3
        mock_s3.generate_file_key.return_value = "uploads/uuid/clip.avi"
        mock_s3.generate_presigned_upload_url.return_value = "https://s3/presigned"

        response = client.post(
            "/api/v1/upload/presigned-url",
            json={"file_name": "clip.avi", "content_type": "video/x-msvideo"},
        )

        assert response.status_code == 200

    def test_invalid_content_type_rejected_by_schema(
        self, client: TestClient
    ) -> None:
        """Invalid content type is rejected by Pydantic schema validation."""
        response = client.post(
            "/api/v1/upload/presigned-url",
            json={"file_name": "doc.pdf", "content_type": "application/pdf"},
        )

        assert response.status_code == 422

    def test_missing_file_name_rejected(self, client: TestClient) -> None:
        """Missing file_name field returns 422."""
        response = client.post(
            "/api/v1/upload/presigned-url",
            json={"content_type": "video/mp4"},
        )

        assert response.status_code == 422

    def test_missing_content_type_rejected(self, client: TestClient) -> None:
        """Missing content_type field returns 422."""
        response = client.post(
            "/api/v1/upload/presigned-url",
            json={"file_name": "video.mp4"},
        )

        assert response.status_code == 422
