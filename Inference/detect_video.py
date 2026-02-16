import cv2
import torch
import time
from temporal_voting import MultiFrameVoter
from fallback_anomaly import FallbackAnomalyDetector

# -----------------------------
# False-positive filters
# -----------------------------
MIN_BOX_AREA_RATIO = 0.001   # too small → noise
MAX_BOX_AREA_RATIO = 0.15    # too large → trees/sky
MIN_BOX_Y_RATIO = 0.45       # must be on road (lower half)

# -----------------------------
# Load YOLOv5 model
# -----------------------------
model = torch.hub.load(
    repo_or_dir="../yolov5",
    model="custom",
    path="../yolov5/runs/train/road_anomaly_yolov5n_final/weights/best.pt",
    source="local",
    force_reload=True
)

model.conf = 0.25
class_names = model.names

# -----------------------------
# Modules
# -----------------------------
voter = MultiFrameVoter(window_size=3, vote_threshold=2)
fallback = FallbackAnomalyDetector(threshold=2.8)

# -----------------------------
# Video input
# -----------------------------
cap = cv2.VideoCapture("pothole.mp4")
if not cap.isOpened():
    print("❌ Video not opened")
    exit()

print("✅ Video opened")

# -----------------------------
# Counters
# -----------------------------
total_frames = 0
yolo_hits = 0
fallback_hits = 0

# -----------------------------
# FPS
# -----------------------------
prev_time = time.time()

# =============================
# MAIN LOOP
# =============================
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    h, w, _ = frame.shape
    frame_area = h * w
    total_frames += 1

    # -----------------------------
    # ROAD ROI
    # -----------------------------
    crop_top = int(0.45 * h)
    crop_left = int(0.15 * w)
    crop_right = int(0.85 * w)

    roi = frame[crop_top:h, crop_left:crop_right]

    # -----------------------------
    # YOLO inference
    # -----------------------------
    results = model(roi)
    raw_detections = results.xyxy[0]

    valid_detections = []

    for det in raw_detections:
        x1, y1, x2, y2 = map(int, det[:4])
        conf = float(det[4])
        cls = int(det[5])

        # shift back to original frame
        x1 += crop_left
        x2 += crop_left
        y1 += crop_top
        y2 += crop_top

        box_area = (x2 - x1) * (y2 - y1)
        box_center_y = (y1 + y2) / 2

        # ---- Filter 1: size ----
        if box_area < MIN_BOX_AREA_RATIO * frame_area:
            continue
        if box_area > MAX_BOX_AREA_RATIO * frame_area:
            continue

        # ---- Filter 2: position ----
        if box_center_y < MIN_BOX_Y_RATIO * h:
            continue

        valid_detections.append((x1, y1, x2, y2, conf, cls))

    detected = len(valid_detections) > 0
    confirmed = voter.update(detected)

    # -----------------------------
    # Train fallback on early frames
    # -----------------------------
    if total_frames <= 30:
        fallback.fit(frame)

    # -----------------------------
    # YOLO CONFIRMED (GREEN)
    # -----------------------------
    if confirmed and detected:
        yolo_hits += 1

        for x1, y1, x2, y2, conf, cls in valid_detections:
            label = f"{class_names[cls]} {conf:.2f}"
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(
                frame, label,
                (x1, y1 - 6),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6, (0, 255, 0), 2
            )

        cv2.putText(
            frame, "YOLO CONFIRMED",
            (30, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1, (0, 255, 0), 3
        )

    # -----------------------------
    # FALLBACK (BLUE)
    # -----------------------------
    else:
        score = fallback.score(frame)

        if score > fallback.threshold and detected:
            fallback_hits += 1

            for x1, y1, x2, y2, conf, cls in valid_detections:
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
                cv2.putText(
                    frame, "FALLBACK",
                    (x1, y1 - 6),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (255, 0, 0), 2
                )

            cv2.putText(
                frame, "FALLBACK ANOMALY",
                (30, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1, (255, 0, 0), 3
            )

    # -----------------------------
    # FPS
    # -----------------------------
    curr_time = time.time()
    fps = int(1 / max(curr_time - prev_time, 1e-6))
    prev_time = curr_time

    cv2.putText(
        frame, f"FPS: {fps}",
        (30, 80),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8, (0, 255, 255), 2
    )

    # -----------------------------
    # Display
    # -----------------------------
    cv2.imshow("Road Anomaly Detection", frame)
    if cv2.waitKey(20) & 0xFF == ord("q"):
        break

# -----------------------------
# Cleanup
# -----------------------------
cap.release()
cv2.destroyAllWindows()

# -----------------------------
# FINAL SUMMARY
# -----------------------------
print("\n==== FINAL RESULTS ====")
print("Total frames      :", total_frames)
print("YOLO detections   :", yolo_hits)
print("Fallback detects  :", fallback_hits)
print("======================")


