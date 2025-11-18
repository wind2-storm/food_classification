# -*- encoding: utf-8 -*-
from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename
import cv2
import numpy as np
import json
import base64
import os
import time

# 실행 위치를 프로젝트 루트로 고정
os.chdir(os.path.dirname(os.path.abspath(__file__)))

from food_classifier_yolo import food_classifier_Json

app = Flask(__name__)

def recognize(filename):
    image = cv2.imread(filename)
    return food_classifier_Json(image=image)

def recognizeBase64(base64_code):
    file_bytes = np.asarray(bytearray(base64.b64decode(base64_code)),dtype=np.uint8)
    image_data_ndarray = cv2.imdecode(file_bytes,1)
    return food_classifier_Json(image_data_ndarray)

@app.route('/uploader', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        f = request.files['file']
        filepath = "./images_rec/" + secure_filename(f.filename)
        f.save(filepath)

        t0 = time.time()
        res = recognize(filepath)
        print("elapsed time:", time.time() - t0)

        # JSON 문자열 → dict → JSON 응답
        return jsonify(json.loads(res))

    return render_template('upload.html')

if __name__ == '__main__':
    app.run("0.0.0.0", port=3000)
