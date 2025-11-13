FROM python:3.10-slim

# 기본 패키지 업데이트 및 OpenCV 필수 라이브러리 설치
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# 앱 디렉토리 생성
WORKDIR /app

# 모든 파일 복사
COPY . .

# pip 업그레이드
RUN pip install --upgrade pip

# Python 의존성 설치
RUN pip install -r requirements.txt

# FastAPI 서버 실행
CMD ["python", "server.py"]
