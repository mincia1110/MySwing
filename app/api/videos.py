"""Video metadata and thumbnail API endpoints.

Provides endpoints for extracting metadata and generating thumbnails
from uploaded video files (Requirement 1.7).
"""

import asyncio
import logging
import os
import tempfile
from pathlib import Path, PurePosixPath
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy import select

from app.api.dependencies import get_current_user_id
from app.core.config import settings
from app.db.models import VideoTable
from app.db.session import async_session_factory
from app.schemas.video import ResolutionResponse, VideoMetadataWithThumbnailResponse
from app.services.s3_client import get_s3_client
from app.services.thumbnail_service import generate_thumbnail_from_s3
from app.services.video_validator import extract_metadata, validate_single_swing_input_policy

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/videos", tags=["videos"])


def _validate_upload_file_key(file_key: str) -> None:
    """Reject object keys outside the upload namespace or with traversal."""
    path = PurePosixPath(file_key)
    parts = path.parts
    if (
        not file_key
        or file_key.startswith("/")
        or "\\" in file_key
        or ".." in parts
        or len(parts) < 3
        or parts[0] != "uploads"
    ):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid upload file_key.",
        )


@router.post(
    "/{file_key:path}/metadata",
    response_model=VideoMetadataWithThumbnailResponse,
    status_code=status.HTTP_200_OK,
    summary="Extract metadata and generate thumbnail",
    description=(
        "Extracts video metadata and generates a preview thumbnail "
        "from the first frame. Must respond within 5 seconds (Requirement 1.7)."
    ),
)
async def get_video_metadata(
    file_key: str,
    current_user_id: UUID = Depends(get_current_user_id),
) -> VideoMetadataWithThumbnailResponse:
    """Extract metadata and generate thumbnail for an uploaded video.

    Downloads the video from S3, extracts metadata (file_name, duration,
    resolution, file_size), generates a thumbnail from the first frame,
    uploads the thumbnail to S3, saves a video record to the DB,
    and returns the combined response.

    Must respond within 5 seconds per Requirement 1.7.
    """
    _validate_upload_file_key(file_key)
    s3_client = get_s3_client()

    # Verify the file exists and is within the server-side size limit before
    # downloading it for expensive metadata extraction.
    try:
        object_head = await asyncio.to_thread(s3_client.head_object, file_key)
    except Exception:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Video file not found: {file_key}",
        )

    content_length = object_head.get("ContentLength")
    if isinstance(content_length, int) and content_length > settings.max_file_size_bytes:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail={
                "status": "file_too_large",
                "message": "Uploaded video exceeds the maximum allowed size.",
                "max_file_size_bytes": settings.max_file_size_bytes,
                "file_size_bytes": content_length,
            },
        )

    # Existing metadata records are authoritative for ownership. Check before
    # downloading or processing the object.
    factory = async_session_factory()
    async with factory() as session:
        existing = await session.execute(
            select(VideoTable).where(VideoTable.file_key == file_key)
        )
        video_record = existing.scalar_one_or_none()
        if video_record is not None and video_record.user_id != current_user_id:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Video file belongs to a different user.",
            )

    # Download video to temp file for metadata extraction
    video_path = Path(file_key)
    tmp_video_path = ""
    try:
        with tempfile.NamedTemporaryFile(
            suffix=video_path.suffix, delete=False
        ) as tmp_video:
            tmp_video_path = tmp_video.name

        # boto3 download_file is sync I/O; offload to a thread to avoid
        # blocking the event loop.
        await asyncio.to_thread(
            s3_client._client.download_file,
            s3_client._bucket,
            file_key,
            tmp_video_path,
        )

        # Extract metadata (OpenCV is sync; offload to thread).
        metadata = await asyncio.to_thread(extract_metadata, tmp_video_path)

        # Generate thumbnail and upload to S3 (also sync; offload to thread).
        thumbnail_url = None
        try:
            thumbnail_key = await asyncio.to_thread(
                generate_thumbnail_from_s3, file_key, s3_client
            )
            thumbnail_url = s3_client.generate_presigned_download_url(
                thumbnail_key, expires_in=3600
            )
        except (ValueError, Exception) as e:
            # Thumbnail generation is non-critical; log and continue
            logger.warning("Thumbnail generation failed for %s: %s", file_key, e)

        # Save video record to DB (upsert by file_key)
        try:
            factory = async_session_factory()
            async with factory() as session:
                # Check if video record already exists
                existing = await session.execute(
                    select(VideoTable).where(VideoTable.file_key == file_key)
                )
                video_record = existing.scalar_one_or_none()
                if video_record is None:
                    # Determine format from file extension
                    ext = Path(file_key).suffix.lstrip(".").lower() or "mp4"
                    video_record = VideoTable(
                        user_id=current_user_id,
                        file_key=file_key,
                        file_name=metadata.file_name,
                        file_size_bytes=metadata.file_size_bytes,
                        duration_seconds=metadata.duration_seconds,
                        resolution_width=metadata.resolution_width,
                        resolution_height=metadata.resolution_height,
                        frame_rate=metadata.frame_rate,
                        format=ext,
                    )
                    session.add(video_record)
                elif video_record.user_id != current_user_id:
                    raise HTTPException(
                        status_code=status.HTTP_403_FORBIDDEN,
                        detail="Video file belongs to a different user.",
                    )
                else:
                    # Update existing record
                    video_record.file_name = metadata.file_name
                    video_record.file_size_bytes = metadata.file_size_bytes
                    video_record.duration_seconds = metadata.duration_seconds
                    video_record.resolution_width = metadata.resolution_width
                    video_record.resolution_height = metadata.resolution_height
                    video_record.frame_rate = metadata.frame_rate
                await session.commit()
                logger.info("Video record saved for file_key=%s", file_key)
        except HTTPException:
            raise
        except Exception as e:
            logger.warning("Failed to save video record for %s: %s", file_key, e)

        input_validation = validate_single_swing_input_policy(metadata.duration_seconds)
        return VideoMetadataWithThumbnailResponse(
            file_name=metadata.file_name,
            duration_seconds=metadata.duration_seconds,
            resolution=ResolutionResponse(
                width=metadata.resolution_width,
                height=metadata.resolution_height,
            ),
            file_size_bytes=metadata.file_size_bytes,
            thumbnail_url=thumbnail_url,
            input_validation=input_validation.to_dict(),
        )

    except ValueError as e:
        input_validation = validate_single_swing_input_policy(None)
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail={
                "status": "metadata_unavailable",
                "message": f"Cannot process video file: {e}",
                "recommendation": input_validation.recommendation,
                "input_validation": input_validation.to_dict(),
            },
        )
    except HTTPException:
        raise
    except Exception:
        logger.exception("Unexpected error processing video metadata: %s", file_key)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal error processing video metadata",
        )
    finally:
        if tmp_video_path and os.path.exists(tmp_video_path):
            os.unlink(tmp_video_path)
