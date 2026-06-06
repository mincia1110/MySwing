"""Register an uploaded video in the DB (bridges S3 upload → DB record).

Usage:
    python -m scripts.register_video <file_key> <user_id>

Example:
    python -m scripts.register_video "uploads/cd7c840a-c8f1-465a-a8b0-7818caf27cba/swing.mp4" "00000000-0000-0000-0000-000000000001"
"""
import sys
import uuid
import tempfile
import os

def main():
    if len(sys.argv) < 3:
        print("Usage: python -m scripts.register_video <file_key> <user_id>")
        sys.exit(1)

    file_key = sys.argv[1]
    user_id = uuid.UUID(sys.argv[2])

    from app.db.session import sync_session_factory
    from app.db.models import VideoTable
    from app.services.s3_client import S3Client

    s3 = S3Client()

    # Check file exists in S3
    if not s3.check_file_exists(file_key):
        print(f"ERROR: File not found in S3: {file_key}")
        sys.exit(1)

    # Download to temp file to extract metadata
    tmp = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
    tmp_path = tmp.name
    tmp.close()

    try:
        s3._client.download_file(s3._bucket, file_key, tmp_path)

        from app.services.video_validator import extract_metadata
        metadata = extract_metadata(tmp_path)

        session = sync_session_factory()
        try:
            # Check if already registered
            existing = session.query(VideoTable).filter(
                VideoTable.file_key == file_key
            ).first()
            if existing:
                print(f"Video already registered: id={existing.id}")
                return

            video = VideoTable(
                user_id=user_id,
                file_key=file_key,
                file_name=metadata.file_name,
                file_size_bytes=metadata.file_size_bytes,
                duration_seconds=metadata.duration_seconds,
                resolution_width=metadata.resolution_width,
                resolution_height=metadata.resolution_height,
                frame_rate=metadata.frame_rate,
                format=metadata.format,
            )
            session.add(video)
            session.commit()
            print(f"Video registered: id={video.id}")
            print(f"  file_key: {file_key}")
            print(f"  duration: {metadata.duration_seconds:.1f}s")
            print(f"  resolution: {metadata.resolution_width}x{metadata.resolution_height}")
            print(f"  fps: {metadata.frame_rate}")
        except Exception as e:
            session.rollback()
            print(f"DB Error: {e}")
        finally:
            session.close()
    finally:
        os.unlink(tmp_path)

if __name__ == "__main__":
    main()
