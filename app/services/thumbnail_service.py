"""Thumbnail generation service for uploaded videos.

Extracts the first frame from a video file and generates a JPEG thumbnail.
"""

import logging
import os
import tempfile
from pathlib import Path

import cv2
import numpy as np

from app.services.s3_client import S3Client

logger = logging.getLogger(__name__)

# Thumbnail dimensions
THUMBNAIL_WIDTH = 320
THUMBNAIL_HEIGHT = 180
THUMBNAIL_JPEG_QUALITY = 85


def generate_thumbnail(video_file_path: str) -> bytes:
    """Extract the first frame from a video and generate a JPEG thumbnail.

    Reads the first frame of the video, resizes it to 320x180 pixels,
    and encodes it as JPEG.

    Args:
        video_file_path: Path to the video file.

    Returns:
        JPEG-encoded thumbnail as bytes.

    Raises:
        ValueError: If the video cannot be opened or the first frame cannot be read.
    """
    cap = cv2.VideoCapture(video_file_path)
    try:
        if not cap.isOpened():
            raise ValueError(f"Cannot open video file: {video_file_path}")

        ret, frame = cap.read()
        if not ret or frame is None:
            raise ValueError(
                f"Cannot read first frame from video: {video_file_path}"
            )

        # Resize to thumbnail dimensions
        thumbnail = cv2.resize(
            frame,
            (THUMBNAIL_WIDTH, THUMBNAIL_HEIGHT),
            interpolation=cv2.INTER_AREA,
        )

        # Encode as JPEG
        encode_params = [cv2.IMWRITE_JPEG_QUALITY, THUMBNAIL_JPEG_QUALITY]
        success, encoded = cv2.imencode(".jpg", thumbnail, encode_params)
        if not success:
            raise ValueError("Failed to encode thumbnail as JPEG")

        return encoded.tobytes()
    finally:
        cap.release()


def generate_thumbnail_from_s3(file_key: str, s3_client: S3Client) -> str:
    """Download video from S3, generate thumbnail, upload it back to S3.

    Downloads the video to a temporary file, generates a thumbnail from
    the first frame, uploads the thumbnail to S3, and returns the S3 key.

    Args:
        file_key: S3 key of the uploaded video file.
        s3_client: S3Client instance for download/upload operations.

    Returns:
        S3 key of the uploaded thumbnail image.

    Raises:
        ValueError: If the video cannot be processed.
    """
    # Determine thumbnail key based on video key
    video_path = Path(file_key)
    thumbnail_key = f"{video_path.parent}/thumbnail_{video_path.stem}.jpg"

    # Download video to temp file
    with tempfile.NamedTemporaryFile(
        suffix=video_path.suffix, delete=False
    ) as tmp_video:
        tmp_video_path = tmp_video.name

    try:
        # Download the video from S3
        s3_client._client.download_file(
            Bucket=s3_client._bucket,
            Key=file_key,
            Filename=tmp_video_path,
        )

        # Generate thumbnail
        thumbnail_bytes = generate_thumbnail(tmp_video_path)

        # Upload thumbnail to S3
        s3_client._client.put_object(
            Bucket=s3_client._bucket,
            Key=thumbnail_key,
            Body=thumbnail_bytes,
            ContentType="image/jpeg",
        )

        return thumbnail_key
    finally:
        # Clean up temp file
        if os.path.exists(tmp_video_path):
            os.unlink(tmp_video_path)
