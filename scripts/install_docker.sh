#!/bin/bash
# Docker Engine 설치 스크립트 (Ubuntu 24.04 WSL)
# 실행: sudo bash scripts/install_docker.sh

set -e

echo "=== Docker Engine 설치 시작 ==="

# 기존 Docker 패키지 제거
apt-get remove -y docker docker-engine docker.io containerd runc 2>/dev/null || true

# 필수 패키지 설치
apt-get update
apt-get install -y ca-certificates curl gnupg

# Docker GPG 키 추가
install -m 0755 -d /etc/apt/keyrings
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | gpg --dearmor -o /etc/apt/keyrings/docker.gpg
chmod a+r /etc/apt/keyrings/docker.gpg

# Docker 저장소 추가
echo \
  "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] https://download.docker.com/linux/ubuntu \
  $(. /etc/os-release && echo "$VERSION_CODENAME") stable" | \
  tee /etc/apt/sources.list.d/docker.list > /dev/null

# Docker 설치
apt-get update
apt-get install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin

# 현재 사용자를 docker 그룹에 추가 (sudo 없이 docker 사용 가능)
usermod -aG docker kim_minchul

# Docker 서비스 시작
service docker start

echo "=== Docker 설치 완료 ==="
echo ""
echo "다음 명령으로 확인:"
echo "  docker --version"
echo "  docker compose version"
echo ""
echo "※ 그룹 변경 적용을 위해 WSL을 재시작하세요:"
echo "  (Windows에서) wsl --shutdown"
echo "  그 후 다시 WSL 터미널 열기"
echo ""
echo "그 다음 프로젝트 실행:"
echo "  cd /home/kim_minchul/workspace/MySwing"
echo "  docker compose up -d"
echo "  source .venv/bin/activate"
echo "  pip install asyncpg mediapipe"
echo "  alembic upgrade head"
echo "  python -m scripts.seed_reference_data"
echo "  uvicorn app.main:app --reload --host 0.0.0.0 --port 8000"
