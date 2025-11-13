FROM python:3.10-slim

# 시스템 패키지 설치 (opencv가 요구하는 패키지)
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# requirements 먼저 복사
COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

# 전체 프로젝트 복사
COPY . .

# FastAPI 서버 실행
CMD ["python", "server.py"]
