# MySwing - AI 야구 스윙 분석 서비스

[English](README.md) | 한국어

파일 업로드 기반 야구 스윙 분석 서비스. Computer Vision(MediaPipe Pose + 손목/팔꿈치 기반 배트 추정)으로 신체 포즈와 배트를 인식하여 스포츠공학적/야구이론 기반 분석 리포트를 제공합니다.

> 현재 프로젝트는 로컬 개발/검증용 Docker Compose 구성을 기준으로 합니다. 실제 배포 환경에서는 저장소, CORS, 파일 공개 범위, 인증/권한 정책을 별도로 설정해야 합니다.

## 주요 기능

- **비디오 업로드**: Presigned URL 기반 S3 직접 업로드 (mp4/mov/avi, 최대 500MB)
- **포즈 추정**: MediaPipe Pose 33개 랜드마크 감지 + 다중 프레임 추적/보간
- **배트 추정**: 손목/팔꿈치 keypoint 기반 배트 헤드 추정 (WristBatEstimator)
- **스윙 분류**: 6단계 (Stance -> Load -> Stride -> Rotation -> Impact -> Follow-through)
- **생체역학 분석**: 배트 스피드, 공격각(attack_angle; API 노출값은 양수 bat-path magnitude), 운동 연쇄, 회전 분석, 핸드 패스 효율
- **스윙 평가**: 레벨별 참조 데이터 비교 + 현대 타격 원칙 평가
- **좌타자 지원**: 영상 좌우반전으로 canonical RHB 좌표계 분석 -> 결과 metadata에 정규화/좌표계 정보 노출
- **리포트 생성**: 오버레이 비디오(H.264), 메트릭 테이블, 드릴 추천, 트렌드 차트

## 아키텍처

```text
Frontend -> FastAPI REST API -> Celery Worker
                |                    |
             PostgreSQL           MinIO/S3
                |
              Redis
```

## 기술 스택

| 레이어 | 기술 |
|--------|------|
| Backend | FastAPI, Python 3.12, SQLAlchemy, Pydantic |
| Task Queue | Celery + Redis |
| CV/ML | MediaPipe Pose, OpenCV |
| Video | ffmpeg (H.264 인코딩) |
| Storage | PostgreSQL, MinIO (S3 호환) |
| Frontend | React, TypeScript, Vite |
| Testing | pytest, Hypothesis (Property-Based Testing), Vitest |
| Infra | Docker Compose |

## 빠른 시작

### 사전 요구사항

- Python 3.12+
- Docker & Docker Compose
- ffmpeg
- Node.js 18+ (Frontend)
- Linux/macOS 또는 WSL 권장

### 1. 인프라 실행

```bash
docker compose up -d postgres redis minio minio-init
```

로컬 기본 서비스:

| 서비스 | URL/포트 |
|--------|----------|
| PostgreSQL | `localhost:5432` |
| Redis | `localhost:6379` |
| MinIO API | `http://localhost:9000` |
| MinIO Console | `http://localhost:9001` |

`docker-compose.yml`과 `.env.example`의 계정/키는 로컬 개발용 기본값입니다. 공개 배포나 외부 접근 환경에서는 반드시 별도 값으로 교체하세요.

### 2. Backend 설정

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev,ml]"

cp .env.example .env
alembic upgrade head
python scripts/seed_reference_data.py
```

### 3. 서비스 실행

```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

celery -A app.core.celery_app:celery_app worker \
  --loglevel=info \
  -Q default,video_processing,analysis \
  --include=app.tasks.pipeline
```

실행 확인:

```bash
curl http://localhost:8000/health
curl http://localhost:8000/api/v1/status
```

API 문서는 FastAPI 기본 문서 화면에서 확인할 수 있습니다:

- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`

### 4. Frontend 실행

```bash
cd frontend
npm install
npm run dev
```

기본 프론트엔드 개발 서버는 `http://localhost:5173`, API 서버는 `http://localhost:8000`에서 실행됩니다.
Vite 개발 서버는 `/api` 요청을 백엔드로 프록시합니다. 백엔드를 다른 주소에서 실행한다면 `frontend/.env.local`에 아래 값을 지정하세요:

```bash
VITE_API_BASE_URL=http://localhost:8000/api/v1
```

## API 엔드포인트

| Method | Path | 설명 |
|--------|------|------|
| POST | `/api/v1/upload/presigned-url` | 업로드 URL 생성 |
| POST | `/api/v1/users/{id}/profile` | 사용자 프로필 생성/수정 |
| GET | `/api/v1/users/{id}/profile` | 프로필 조회 |
| POST | `/api/v1/videos/{file_key}/metadata` | 업로드 영상 metadata 추출/DB 등록 |
| POST | `/api/v1/analyses` | 분석 작업 생성 (`file_key`, `user_id`) |
| GET | `/api/v1/analyses/{id}/status` | 분석 상태 조회 |
| GET | `/api/v1/analyses/{id}/report` | 리포트 조회 |
| GET | `/api/v1/analyses/{id}/overlay` | 오버레이 비디오 URL |
| GET | `/api/v1/analyses/{id}/metrics` | 메트릭 데이터 |
| GET | `/api/v1/users/{id}/analyses` | 분석 이력 |
| GET | `/api/v1/users/{id}/trends` | 트렌드 데이터 |

## 분석 파이프라인

```text
Video Upload -> Preprocessing -> Pose Estimation -> Wrist-based Bat Estimation
    -> Swing Classification -> Biomechanics Analysis
    -> Swing Evaluation -> Report Generation
```

각 단계는 독립적으로 실패할 수 있으며, Graceful Degradation으로 가능한 범위 내 결과를 제공합니다.

### 측정 메트릭

| 메트릭 | 단위 | 설명 |
|--------|------|------|
| bat_speed | km/h | 임팩트 존 배트 스피드 (손목 기반 추정) |
| attack_angle | degrees | 임팩트 순간 배트 경로 각도. API/report 노출값은 참조 범위(+5~+25 degrees) 비교용 양수 magnitude이며, 내부 low-level 계산기는 signed diagnostic angle을 유지 |
| hip_shoulder_separation | degrees | 골반-어깨 최대 분리각 (정면 촬영 시만) |
| hand_path_efficiency | ratio | 핸드 패스 효율 (0.0~1.0) |
| kinematic_chain | degrees/s | 각 관절 최대 각속도 |
| stride_length | cm | rotation 시작 시 양발 간격 |
| cog_sway / cog_drop | cm | 무게중심 좌우 흔들림 / 수직 하강 |
| head_stability | cm | 스윙 중 머리 최대 이탈 거리 |
| front_knee_flexion | degrees | stride 착지 시 앞무릎 각도 |
| spine_angle | degrees | load 시 척추-수직선 각도 |

### 드릴 추천 로직

메트릭이 참조 범위를 벗어나면 방향에 따라 다른 교정 드릴을 추천합니다:

| 메트릭 | 기준 미만 시 | 기준 초과 시 |
|--------|-------------|-------------|
| attack_angle | 상향 스윙 경로 훈련 | 레벨 스윙 교정 |
| bat_speed | 파워/폭발력 훈련 | 컨트롤/정확도 훈련 |
| hand_path_efficiency | 컴팩트 핸드 패스 훈련 | 풀 익스텐션 훈련 |

## 사용자 프로필

분석 정확도를 위해 다음 정보가 필요합니다:

- **필수**: 신장(cm), 배트 길이(inch), 타격 방향(좌/우)
- **선택**: 체중, 카메라 방향, 연령대, 레벨, 배트 무게

## 촬영 가이드

| 항목 | 권장 |
|------|------|
| 카메라 위치 | 정면 또는 후면 (3루/1루 측) |
| 해상도 | 1280x720 이상 |
| 프레임레이트 | 30fps 이상 (60fps 권장) |
| 조명 | 밝은 환경 (40 lux 이상) |
| 프레이밍 | 전신이 보이도록 (머리~발목) |
| 길이 | 5분 이내 |
| 파일 형식 | MP4, MOV, AVI |

> 측면 촬영 시 회전 분석(hip_shoulder_separation)이 제한됩니다.
> 세로 영상도 지원되지만, 전신이 보이는 가로 영상에서 최적의 결과를 얻을 수 있습니다.
> 좌타자는 프로필에 `batting_direction: left` 설정 시 자동으로 영상 반전 후 canonical RHB 좌표계에서 분석됩니다. 리포트의 `analysis_metadata`에서 `analysis_coordinate_system`, `canonical_batting_direction`, `video_normalization` 정보를 확인할 수 있습니다.

## 테스트

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

## 프로젝트 구조

```text
MySwing/
├── app/
│   ├── api/            # FastAPI 라우터 (upload, analyses, history, profile)
│   ├── core/           # 설정, Celery, DB 연결
│   ├── db/             # SQLAlchemy 모델, 세션
│   ├── models/         # 도메인 데이터 모델 (dataclass)
│   ├── pipeline/       # CV/ML 파이프라인 모듈
│   ├── schemas/        # Pydantic 요청/응답 스키마
│   ├── services/       # 비즈니스 로직 (S3, 검증, 품질 체크)
│   └── tasks/          # Celery 태스크 (파이프라인 오케스트레이션)
├── frontend/           # React + TypeScript
├── tests/              # pytest + Hypothesis
├── alembic/            # DB 마이그레이션
├── scripts/            # 개발/운영 보조 스크립트
├── docker-compose.yml
└── pyproject.toml
```

## 개발 메모

- `.env.example`은 로컬 Docker Compose 기본값을 담고 있습니다.
- 실제 영상 분석에는 `mediapipe`가 필요하므로 backend 설치 시 `.[ml]` extra를 포함해야 합니다.
- 배트 궤적은 현재 객체 감지 모델이 아니라 손목/팔꿈치 keypoint 기반 추정 경로를 사용합니다.
- 업로드된 원본/오버레이 영상은 S3 호환 스토리지에 저장됩니다. 로컬 개발에서는 MinIO를 사용합니다.

## 라이선스

Apache License 2.0. 자세한 내용은 [LICENSE](LICENSE)를 참고하세요.
