FROM python:3.10-slim

# -----------------------------
# 1) 필수 라이브러리 설치
# -----------------------------
RUN apt-get update && apt-get install -y \
    libglib2.0-0 \
    libgl1 \
    wget \
    && rm -rf /var/lib/apt/lists/*

# -----------------------------
# 2) 작업 디렉토리
# -----------------------------
WORKDIR /app

# -----------------------------
# 3) Python 패키지 설치
# -----------------------------
COPY requirements.txt /app/
RUN pip install --no-cache-dir -r requirements.txt

# -----------------------------
# 4) 프로젝트 소스 코드 복사
# -----------------------------
COPY . /app/

# -----------------------------
# 5) YOLO weights 자동 다운로드
#    GitHub Release에서 70MB 파일 받아서 /app 위치에 저장
# -----------------------------
RUN mkdir -p /app/yolo/data/food/weights && \
    wget -O /app/yolo/data/food/weights/food-dark-yolov3-tiny_3l-v3-2_24000.weights \
    https://github.com/wind2-storm/food_classification/releases/download/1/food-dark-yolov3-tiny_3l-v3-2_24000.weights

# -----------------------------
# 6) WebAPI Flask 서버 실행
# -----------------------------
EXPOSE 3000
CMD ["python", "WebAPI.py"]
