# MySwing - AI Baseball Swing Analysis

English | [한국어](README.ko.md)

MySwing is a file-upload based baseball swing analysis service. It uses computer vision with MediaPipe Pose and wrist/elbow-based bat estimation to detect body posture and bat motion, then generates biomechanics and baseball-theory driven analysis reports.

> This project is currently designed for local development and validation with Docker Compose. Production deployments must configure storage, CORS, file visibility, authentication, and authorization policies separately.

## Features

- **Video upload**: Direct S3-compatible upload with presigned URLs (mp4/mov/avi, up to 500 MB)
- **Pose estimation**: MediaPipe Pose 33-landmark detection with multi-frame tracking and interpolation
- **Bat estimation**: Wrist/elbow keypoint-based bat head estimation through `WristBatEstimator`
- **Swing classification**: Six phases: Stance -> Load -> Stride -> Rotation -> Impact -> Follow-through
- **Biomechanics analysis**: Bat speed, attack angle, kinematic chain, rotation analysis, and hand path efficiency
- **Swing evaluation**: Level-aware reference comparison plus modern hitting principle checks
- **Left-handed hitter support**: Mirrors left-handed videos into the canonical right-handed coordinate system and exposes normalization metadata
- **Report generation**: H.264 overlay video, metrics table, drill recommendations, and trend charts

## Architecture

```text
Frontend -> FastAPI REST API -> Celery Worker
                |                    |
             PostgreSQL           MinIO/S3
                |
              Redis
```

## Tech Stack

| Layer | Technology |
|-------|------------|
| Backend | FastAPI, Python 3.12, SQLAlchemy, Pydantic |
| Task Queue | Celery + Redis |
| CV/ML | MediaPipe Pose, OpenCV |
| Video | ffmpeg (H.264 encoding) |
| Storage | PostgreSQL, MinIO (S3-compatible) |
| Frontend | React, TypeScript, Vite |
| Testing | pytest, Hypothesis (property-based testing), Vitest |
| Infra | Docker Compose |

## Quick Start

### Prerequisites

- Python 3.12+
- Docker & Docker Compose
- ffmpeg
- Node.js 18+ for the frontend
- Linux, macOS, or WSL recommended

### 1. Start Infrastructure

```bash
docker compose up -d postgres redis minio minio-init
```

Default local services:

| Service | URL/port |
|---------|----------|
| PostgreSQL | `localhost:5432` |
| Redis | `localhost:6379` |
| MinIO API | `http://localhost:9000` |
| MinIO Console | `http://localhost:9001` |

The credentials and keys in `docker-compose.yml` and `.env.example` are local development defaults. Replace them before public deployment or external access.

### 2. Configure Backend

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev,ml]"

cp .env.example .env
alembic upgrade head
python scripts/seed_reference_data.py
```

### 3. Run Services

```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

celery -A app.core.celery_app:celery_app worker \
  --loglevel=info \
  -Q default,video_processing,analysis \
  --include=app.tasks.pipeline
```

Health checks:

```bash
curl http://localhost:8000/health
curl http://localhost:8000/api/v1/status
```

API documentation is available through FastAPI:

- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`

### 4. Run Frontend

```bash
cd frontend
npm install
npm run dev
```

The default frontend dev server runs at `http://localhost:5173`; the API server runs at `http://localhost:8000`.
Vite proxies `/api` requests to the backend. If the backend runs elsewhere, set the following in `frontend/.env.local`:

```bash
VITE_API_BASE_URL=http://localhost:8000/api/v1
```

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| POST | `/api/v1/upload/presigned-url` | Create an upload URL |
| POST | `/api/v1/users/{id}/profile` | Create or update a user profile |
| GET | `/api/v1/users/{id}/profile` | Read a user profile |
| POST | `/api/v1/videos/{file_key}/metadata` | Extract metadata and register an uploaded video |
| POST | `/api/v1/analyses` | Create an analysis job with `file_key` and `user_id` |
| GET | `/api/v1/analyses/{id}/status` | Read analysis status |
| GET | `/api/v1/analyses/{id}/report` | Read the generated report |
| GET | `/api/v1/analyses/{id}/overlay` | Read the overlay video URL |
| GET | `/api/v1/analyses/{id}/metrics` | Read metrics data |
| GET | `/api/v1/users/{id}/analyses` | Read analysis history |
| GET | `/api/v1/users/{id}/trends` | Read trend data |

## Analysis Pipeline

```text
Video Upload -> Preprocessing -> Pose Estimation -> Wrist-based Bat Estimation
    -> Swing Classification -> Biomechanics Analysis
    -> Swing Evaluation -> Report Generation
```

Each stage can fail independently. The pipeline uses graceful degradation to return the best available result when possible.

### Metrics

| Metric | Unit | Description |
|--------|------|-------------|
| bat_speed | km/h | Estimated bat speed in the impact zone |
| attack_angle | degrees | Bat path angle at impact. The API/report value is a positive magnitude used for reference-range comparison; low-level diagnostics retain the signed angle. |
| hip_shoulder_separation | degrees | Maximum hip-shoulder separation, available mainly from front/rear camera views |
| hand_path_efficiency | ratio | Hand path efficiency from 0.0 to 1.0 |
| kinematic_chain | degrees/s | Peak angular velocities across joints |
| stride_length | cm | Foot spacing at the start of rotation |
| cog_sway / cog_drop | cm | Horizontal sway and vertical drop of the center of gravity |
| head_stability | cm | Maximum head displacement during the swing |
| front_knee_flexion | degrees | Front knee angle at stride landing |
| spine_angle | degrees | Spine angle relative to vertical during load |

### Drill Recommendation Logic

When a metric is outside its reference range, MySwing recommends direction-specific correction drills:

| Metric | When below range | When above range |
|--------|------------------|------------------|
| attack_angle | Upward swing path training | Level swing correction |
| bat_speed | Power and explosiveness training | Control and contact-quality training |
| hand_path_efficiency | Compact hand path training | Full extension training |

## User Profile

The following profile data improves analysis accuracy:

- **Required**: height in cm, bat length in inches, batting direction
- **Optional**: weight, camera direction, age group, level, bat weight

## Filming Guide

| Item | Recommendation |
|------|----------------|
| Camera position | Front or rear view from the third-base or first-base side |
| Resolution | 1280x720 or higher |
| Frame rate | 30 fps or higher, 60 fps recommended |
| Lighting | Bright environment, at least 40 lux |
| Framing | Full body visible from head to ankles |
| Length | Up to 5 minutes |
| File format | MP4, MOV, AVI |

> Side-view footage limits rotation analysis such as `hip_shoulder_separation`.
> Portrait videos are supported, but full-body landscape footage produces the best results.
> For left-handed hitters, set `batting_direction: left`; the video is mirrored and analyzed in the canonical right-handed coordinate system. Check `analysis_metadata` for `analysis_coordinate_system`, `canonical_batting_direction`, and `video_normalization`.

## Tests

Backend:

```bash
python -m pytest tests/ -q

python -m pytest \
  tests/unit/test_biomechanics_calibration.py \
  tests/unit/test_biomechanics_angles.py \
  tests/unit/test_biomechanics_orchestrator.py \
  tests/unit/test_impact_frame_estimation.py \
  tests/unit/test_swing_classifier.py \
  tests/unit/test_pipeline_integration.py \
  tests/unit/test_mirror_invariance_benchmark.py -q

python -m pytest tests/ -k "property" -v
```

Frontend:

```bash
cd frontend
npm run lint
npm run test:run
```

## Project Structure

```text
MySwing/
├── app/
│   ├── api/            # FastAPI routers for upload, analyses, history, profile
│   ├── core/           # Configuration, Celery, database wiring
│   ├── db/             # SQLAlchemy models and sessions
│   ├── models/         # Domain dataclass models
│   ├── pipeline/       # CV/ML pipeline modules
│   ├── schemas/        # Pydantic request/response schemas
│   ├── services/       # Business logic for S3, validation, quality checks
│   └── tasks/          # Celery pipeline orchestration
├── frontend/           # React + TypeScript
├── tests/              # pytest + Hypothesis
├── alembic/            # Database migrations
├── scripts/            # Development and operations helpers
├── docker-compose.yml
└── pyproject.toml
```

## Development Notes

- `.env.example` contains local Docker Compose defaults.
- Real video analysis requires `mediapipe`; include the `.[ml]` extra when installing the backend.
- Bat trajectory currently uses wrist/elbow keypoint estimation rather than an object detection model.
- Uploaded source and overlay videos are stored in S3-compatible storage. Local development uses MinIO.

## License

Apache License 2.0. See [LICENSE](LICENSE).
