import cv2
import torch
import os
import random

VIDEO = "pothole.mp4"
OUT_DIR = "D:\ARM COMPETATION\road_anomaly_project\EdgeAI-Road-Anomaly-Detection\verifier\data\normal"
PATCH = 64

os.makedirs(OUT_DIR, exist_ok=True)

# Load YOLO
model = torch.hub.load(
    repo_or_dir="../yolov5",
    model="custom",
    path="../best.pt",
    source="local"
)
model.conf = 0.4   # HIGH confidence → avoid weak detections

cap = cv2.VideoCapture(VIDEO)
count = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    h, w, _ = frame.shape
    results = model(frame)
    detections = results.xyxy[0]

    # Skip frames WITH detections
    if len(detections) > 0:
        continue

    # Take random road patches
    for _ in range(3):
        x = random.randint(int(0.2*w), int(0.8*w - PATCH))
        y = random.randint(int(0.5*h), int(h - PATCH))

        patch = frame[y:y+PATCH, x:x+PATCH]
        if patch.shape[0] == PATCH:
            fname = f"normal_{count}.jpg"
            cv2.imwrite(os.path.join(OUT_DIR, fname), patch)
            count += 1

cap.release()
print(f"✅ Collected {count} NORMAL samples")
