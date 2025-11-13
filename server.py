import cv2
import numpy as np
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
import uvicorn
import os

app = FastAPI()

# --- Load YOLO Model ---
BASE_DIR = "yolo/data/food"
WEIGHTS_PATH = f"{BASE_DIR}/weights/food-dark-yolov3-tiny_3l-v3-2_last.weights"
CFG_PATH = f"yolo/config/food-dark-yolov3-tiny.cfg"
NAMES_PATH = f"{BASE_DIR}/food-classes.names"

# Load classes
with open(NAMES_PATH, "r", encoding="utf-8") as f:
    CLASSES = [line.strip() for line in f.readlines()]

# Load network
net = cv2.dnn.readNet(WEIGHTS_PATH, CFG_PATH)
layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers().flatten()]

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    image_data = np.frombuffer(await file.read(), np.uint8)
    img = cv2.imdecode(image_data, cv2.IMREAD_COLOR)

    height, width, _ = img.shape
    blob = cv2.dnn.blobFromImage(img, 0.00392, (416, 416), swapRB=True, crop=False)

    net.setInput(blob)
    outs = net.forward(output_layers)

    class_ids = []
    confidences = []

    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:
                class_ids.append(class_id)
                confidences.append(float(confidence))

    if len(class_ids) == 0:
        return JSONResponse({"result": None, "confidence": 0})

    # 가장 높은 확률 음식 선택
    idx = np.argmax(confidences)
    food_name = CLASSES[class_ids[idx]]
    confidence = confidences[idx]

    return {"result": food_name, "confidence": confidence}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
