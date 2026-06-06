"""Run a local mirror-invariance sample analysis pair.

This helper is intentionally a script, not production API code. It uploads a
baseline and generated mirror video to the local MinIO bucket, creates DB rows,
runs the Celery task eagerly in-process, and writes a JSON summary.
"""

from __future__ import annotations

import argparse
import json
import mimetypes
import uuid
from pathlib import Path
from typing import Any

from app.db.models import AnalysisTable, UserProfileTable, UserTable, VideoTable
from app.db.session import sync_session_factory
from app.services.s3_client import S3Client
from app.services.video_validator import extract_metadata
from app.tasks.pipeline import analyze_swing_task

DEFAULT_USER_ID = uuid.UUID("00000000-0000-0000-0000-000000000001")


def ensure_user_profile(session, user_id: uuid.UUID, batting_direction: str) -> None:
    user = session.query(UserTable).filter(UserTable.id == user_id).first()
    if user is None:
        user = UserTable(
            id=user_id,
            email="mirror-benchmark@example.local",
            name="Mirror Benchmark",
        )
        session.add(user)
        session.flush()

    profile = (
        session.query(UserProfileTable)
        .filter(UserProfileTable.user_id == user_id)
        .first()
    )
    if profile is None:
        profile = UserProfileTable(
            user_id=user_id,
            height=175.0,
            weight=75.0,
            bat_length=33.0,
            bat_weight=30.0,
            batting_direction=batting_direction,
            camera_direction="side",
            age_group="adult",
            level="recreational",
        )
        session.add(profile)
    else:
        profile.batting_direction = batting_direction
        profile.height = profile.height or 175.0
        profile.bat_length = profile.bat_length or 33.0


def upload_video(path: Path, label: str) -> str:
    s3 = S3Client()
    suffix = path.suffix.lower().lstrip(".") or "mp4"
    key = f"benchmarks/mirror/{label}-{uuid.uuid4()}.{suffix}"
    content_type = mimetypes.guess_type(path.name)[0] or "video/mp4"
    s3._client.upload_file(
        str(path),
        s3._bucket,
        key,
        ExtraArgs={"ContentType": content_type},
    )
    return key


def create_analysis_for_video(
    session,
    *,
    user_id: uuid.UUID,
    path: Path,
    file_key: str,
) -> uuid.UUID:
    metadata = extract_metadata(str(path))
    video = VideoTable(
        user_id=user_id,
        file_key=file_key,
        file_name=path.name,
        file_size_bytes=path.stat().st_size,
        duration_seconds=metadata.duration_seconds,
        resolution_width=metadata.resolution_width,
        resolution_height=metadata.resolution_height,
        frame_rate=metadata.frame_rate,
        format=metadata.format or path.suffix.lower().lstrip(".") or "mp4",
    )
    session.add(video)
    session.flush()

    analysis = AnalysisTable(user_id=user_id, video_id=video.id, status="pending")
    session.add(analysis)
    session.flush()
    return analysis.id


def run_one(path: Path, label: str, batting_direction: str) -> dict[str, Any]:
    user_id = DEFAULT_USER_ID
    file_key = upload_video(path, label)

    session = sync_session_factory()
    try:
        ensure_user_profile(session, user_id, batting_direction)
        analysis_id = create_analysis_for_video(
            session,
            user_id=user_id,
            path=path,
            file_key=file_key,
        )
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()

    result = analyze_swing_task.apply(args=[str(analysis_id)]).get()
    return {
        "label": label,
        "path": str(path),
        "file_key": file_key,
        "batting_direction": batting_direction,
        "analysis_id": str(analysis_id),
        "task_result": result,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--mirror", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    baseline = run_one(args.baseline, "baseline-rhb", "right")
    mirror = run_one(args.mirror, "generated-hflip-lhb", "left")

    payload = {"baseline": baseline, "mirror": mirror}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
