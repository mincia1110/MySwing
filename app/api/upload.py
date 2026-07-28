"""Upload API endpoints for presigned URL generation."""

from pathlib import PurePosixPath
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, status

from app.api.dependencies import get_current_user_id
from app.schemas.video import PresignedUrlRequest, PresignedUrlResponse
from app.services.s3_client import get_s3_client

router = APIRouter(prefix="/upload", tags=["upload"])

ALLOWED_VIDEO_CONTENT_TYPES = {
    "video/mp4",
    "video/quicktime",
    "video/x-msvideo",
}
CONTENT_TYPE_EXTENSIONS = {
    "video/mp4": {".mp4"},
    "video/quicktime": {".mov"},
    "video/x-msvideo": {".avi"},
}


@router.post(
    "/presigned-url",
    response_model=PresignedUrlResponse,
    status_code=status.HTTP_200_OK,
    summary="Generate presigned upload URL",
    description="Generate a presigned S3 URL for direct video file upload.",
)
async def create_presigned_upload_url(
    request: PresignedUrlRequest,
    current_user_id: UUID = Depends(get_current_user_id),
) -> PresignedUrlResponse:
    """Generate a presigned URL for uploading a video file to S3.

    Validates that the content_type is a supported video format,
    generates a unique file key, and returns the presigned upload URL.
    """
    if request.content_type not in ALLOWED_VIDEO_CONTENT_TYPES:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported content type: {request.content_type}. "
            f"Supported types: {', '.join(sorted(ALLOWED_VIDEO_CONTENT_TYPES))}",
        )

    normalized_name = request.file_name.replace("\\", "/")
    extension = PurePosixPath(normalized_name).suffix.lower()
    if extension not in CONTENT_TYPE_EXTENSIONS[request.content_type]:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="File extension does not match the declared video content type.",
        )

    s3_client = get_s3_client()
    try:
        file_key = s3_client.generate_file_key(
            request.file_name,
            owner_id=current_user_id,
        )
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        ) from exc
    expires_in = 3600

    upload_url = s3_client.generate_presigned_upload_url(
        file_key=file_key,
        content_type=request.content_type,
        expires_in=expires_in,
    )

    return PresignedUrlResponse(
        upload_url=upload_url,
        file_key=file_key,
        expires_in=expires_in,
    )
