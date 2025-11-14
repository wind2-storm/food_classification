FROM python:3.10-slim

RUN apt-get update && apt-get install -y \
    libglib2.0-0 \
    libgl1 \
    wget \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt /app/

RUN pip install --no-cache-dir -r requirements.txt

# Copy all project files
COPY . /app/

# Google Drive Large File Download (weights)
RUN mkdir -p /app/yolo/data/food/weights
RUN fileid="1XGFmxWA30nXWwpoQW_qPbv3yg2hD83oW" && \
    filename="food-dark-yolov3-tiny_3l-v3-2_24000.weights" && \
    wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id='$fileid -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1/p')" \
    -O /app/yolo/data/food/weights/$filename && rm -rf /tmp/cookies.txt

CMD ["python", "WebAPI.py"]
