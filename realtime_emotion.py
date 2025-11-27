import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from cnn import *
from ultralytics import YOLO
import numpy as np

EMOTIONS = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model_path = "emotion_cnn_best.pth"
yolo_model = 'yolov8n.pt'

# CNN model
cnn_model = Conv_Net()
cnn_model.load_state_dict(torch.load(model_path, map_location=device))
cnn_model.to(device)
cnn_model.eval()

# YOLO Model
yolo_model = YOLO(yolo_model)

# Preprocess image
def preprocess(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (48, 48))
    tensor = torch.tensor(resized, dtype=torch.float32).unsqueeze(0).unsqueeze(0)  # (1,1,H,W)
    tensor = (tensor / 255.0 - 0.5) / 0.5  # normalize [-1,1]
    return tensor.to(device)

# Predict the emotion
def predict(img):
    tensor = preprocess(img)
    with torch.no_grad():
        output = cnn_model(tensor)
        pred = torch.argmax(output, dim=1).item()
    return EMOTIONS[pred]

# The webcam (real-time)
cap = cv2.VideoCapture(0)

print("Press 'q' to quit.")
while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Detect faces using YOLO
    results = yolo_model(frame)[0]

    for result in results:
        for box, cls in zip(result.boxes.xyxy, result.boxes.cls): 
            if int(cls) == 0:
                x1, y1, x2, y2 = map(int, box.cpu().numpy())
                img = frame[y1:y2, x1:x2]
            
                if img.size == 0:
                    continue
                
                emotion = predict(img)

                # Draw bounding box and label
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, emotion, (x1, y1-10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    # Show frame
    cv2.imshow('Real-Time Emotion Detector', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()