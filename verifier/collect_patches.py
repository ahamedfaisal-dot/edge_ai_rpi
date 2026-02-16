import cv2
import torch
import os

# =============================
# CONFIG (CHANGE ONLY THESE)
# =============================
VIDEO_PATH = "pothole.mp4"   # change video name
LABEL = "pothole"            # "pothole" or "normal"
MAX_PATCHES = 500            # limit to avoid overload

# =============================
# Paths
# =============================
SAVE_DIR = f"verifier/data/{LABEL}"
os.makedirs(SAVE_DIR, exist_ok=True)

# =============================
# Load YOLOv5
# =============================
model = torch.hub.load(
    repo_or_dir="../yolov5",
    model="custom",
    path="../best.pt",
    source="local",
    force_reload=False
)

model.conf = 0.25

# =============================
# Video
# =============================
cap = cv2.VideoCapture(VIDEO_PATH)

if not cap.isOpened():
    print("❌ Cannot open video")
    exit()

print("✅ Video opened")

count = 0
frame_id = 0

# =============================
# MAIN LOOP
# =============================
while cap.isOpened() and count < MAX_PATCHES:
    ret, frame = cap.read()
    if not ret:
        break

    frame_id += 1
    h, w, _ = frame.shape

    # --- Road ROI (important!) ---
    crop_top = int(0.45 * h)
    crop_left = int(0.15 * w)
    crop_right = int(0.85 * w)
    roi = frame[crop_top:h, crop_left:crop_right]

    results = model(roi)
    detections = results.xyxy[0]

    for det in detections:
        if count >= MAX_PATCHES:
            break

        x1, y1, x2, y2, conf, cls = det.tolist()
        x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])

        # shift back to original frame
        x1 += crop_left
        x2 += crop_left
        y1 += crop_top
        y2 += crop_top

        # safety check
        if x2 <= x1 or y2 <= y1:
            continue

        patch = frame[y1:y2, x1:x2]

        # ignore tiny patches
        if patch.shape[0] < 20 or patch.shape[1] < 20:
            continue

        filename = f"{LABEL}_{frame_id}_{count}.jpg"
        save_path = os.path.join(SAVE_DIR, filename)

        cv2.imwrite(save_path, patch)
        count += 1

    if frame_id % 20 == 0:
        print(f"Collected {count} patches...")

cap.release()
print(f"\n✅ DONE: {count} patches saved in {SAVE_DIR}")
