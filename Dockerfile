FROM python:3.10-slim

# 필수 패키지 설치
RUN apt-get update && apt-get install -y \
    libglib2.0-0 \
    libgl1 \
    wget \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# 파이썬 라이브러리 설치
COPY requirements.txt /app/
RUN pip install --no-cache-dir -r requirements.txt

# 프로젝트 전체 복사
COPY . /app/

# YOLO weights 다운로드 (GitHub Release)
RUN mkdir -p /app/yolo/data/food/weights && \
    wget https://github.com/wind2-storm/food_classification/releases/download/1/food-dark-yolov3-tiny_3l-v3-2_24000.weights \
    -O /app/yolo/data/food/weights/food-dark-yolov3-tiny_3l-v3-2_24000.weights

# WebAPI 실행
EXPOSE 3000
CMD ["python", "WebAPI.py"]
