import cv2
import time
import numpy as np
from flask import Flask, render_template, Response, jsonify
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

app = Flask(__name__)

MODEL_PATH = "mask_detector.h5"
FACE_PROTO = "face_detector/deploy.prototxt"
FACE_MODEL = "face_detector/res10_300x300_ssd_iter_140000.caffemodel"
CONFIDENCE_THRESHOLD = 0.5

faceNet = cv2.dnn.readNet(FACE_PROTO, FACE_MODEL)
maskNet = load_model(MODEL_PATH)

streaming = False
mask_count = 0
no_mask_count = 0

def detect_and_predict_mask(frame):
    global mask_count, no_mask_count
    (h, w) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300),
                                 (104.0, 177.0, 123.0))
    faceNet.setInput(blob)
    detections = faceNet.forward()

    faces = []
    locs = []

    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > CONFIDENCE_THRESHOLD:
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")
            (startX, startY) = (max(0, startX), max(0, startY))
            (endX, endY) = (min(w - 1, endX), min(h - 1, endY))

            face = frame[startY:endY, startX:endX]
            if face.size == 0:
                continue

            face_rgb = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
            face_rgb = cv2.resize(face_rgb, (224, 224))
            face_rgb = img_to_array(face_rgb)
            face_rgb = preprocess_input(face_rgb)

            faces.append(face_rgb)
            locs.append((startX, startY, endX, endY))

    if faces:
        preds = maskNet.predict(np.array(faces), batch_size=32)
        for (box, pred) in zip(locs, preds):
            (startX, startY, endX, endY) = box
            (mask, withoutMask) = pred
            label = "Mask" if mask > withoutMask else "No Mask"
            color = (0, 255, 0) if label == "Mask" else (0, 0, 255)

            if label == "Mask":
                mask_count += 1
            else:
                no_mask_count += 1

            label_text = f"{label}: {max(mask, withoutMask) * 100:.2f}%"
            cv2.putText(frame, label_text, (startX, startY - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
            cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)
    return frame

def gen_frames():
    global mask_count, no_mask_count
    cap = cv2.VideoCapture(0)
    time.sleep(2.0)
    while True:
        if not streaming:
            time.sleep(0.1)
            continue
        success, frame = cap.read()
        if not success:
            break
        frame = cv2.resize(frame, (640, 480))
        frame = detect_and_predict_mask(frame)
        ret, buffer = cv2.imencode('.jpg', frame)
        if not ret:
            continue
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/toggle_stream', methods=['POST'])
def toggle_stream():
    global streaming, mask_count, no_mask_count
    streaming = not streaming
    mask_count = 0
    no_mask_count = 0
    return jsonify({'streaming': streaming})

@app.route('/counts')
def counts():
    return jsonify({'mask': mask_count, 'no_mask': no_mask_count})

if __name__ == '__main__':
    app.run(debug=True)
