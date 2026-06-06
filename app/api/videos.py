"""Video metadata and thumbnail API endpoints.

Provides endpoints for extracting metadata and generating thumbnails
from uploaded video files (Requirement 1.7).
"""

import asyncio
import logging
import os
import tempfile
from pathlib import Path
from uuid import UUID

from fastapi import APIRouter, HTTPException, Query, status
from sqlalchemy import select

from app.db.models import VideoTable
from app.db.session import async_session_factory
from app.schemas.video import ResolutionResponse, VideoMetadataWithThumbnailResponse
from app.services.s3_client import get_s3_client
from app.services.thumbnail_service import generate_thumbnail_from_s3
from app.services.video_validator import extract_metadata

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/videos", tags=["videos"])

# Default user ID for MVP (single-user mode)
_DEFAULT_USER_ID = "00000000-0000-0000-0000-000000000001"


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
    user_id: str = Query(default=_DEFAULT_USER_ID, description="User ID for video ownership"),
) -> VideoMetadataWithThumbnailResponse:
    """Extract metadata and generate thumbnail for an uploaded video.

    Downloads the video from S3, extracts metadata (file_name, duration,
    resolution, file_size), generates a thumbnail from the first frame,
    uploads the thumbnail to S3, saves a video record to the DB,
    and returns the combined response.

    Must respond within 5 seconds per Requirement 1.7.
    """
    s3_client = get_s3_client()

    # Verify the file exists in S3 (sync boto3 call -> offload to thread)
    if not await asyncio.to_thread(s3_client.check_file_exists, file_key):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Video file not found: {file_key}",
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
                        user_id=UUID(user_id),
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
        except Exception as e:
            logger.warning("Failed to save video record for %s: %s", file_key, e)

        return VideoMetadataWithThumbnailResponse(
            file_name=metadata.file_name,
            duration_seconds=metadata.duration_seconds,
            resolution=ResolutionResponse(
                width=metadata.resolution_width,
                height=metadata.resolution_height,
            ),
            file_size_bytes=metadata.file_size_bytes,
            thumbnail_url=thumbnail_url,
        )

    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Cannot process video file: {e}",
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Unexpected error processing video metadata: %s", file_key)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal error processing video metadata",
        )
    finally:
        if tmp_video_path and os.path.exists(tmp_video_path):
            os.unlink(tmp_video_path)
