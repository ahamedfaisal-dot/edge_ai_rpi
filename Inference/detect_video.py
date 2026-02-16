import cv2
import torch
import time
import sys
import os
from collections import deque

sys.path.append(os.path.abspath(".."))

from temporal_voting import MultiFrameVoter
from fallback_anomaly import FallbackAnomalyDetector
from verifier.cnn_verifier import CNNVerifier

# =============================
# FILTERS (TUNED FOR STABILITY)
# =============================
MIN_BOX_AREA_RATIO = 0.001
MAX_BOX_AREA_RATIO = 0.08
MIN_BOX_Y_RATIO = 0.50

MIN_BOX_WIDTH = 25
MIN_BOX_HEIGHT = 25

# Tightened aspect ratio bounds (was 0.35–3.5)
MIN_ASPECT_RATIO = 0.4
MAX_ASPECT_RATIO = 3.0

# Border rejection margin (pixels)
BORDER_MARGIN = 5

# Inter-frame area stability threshold (50% max change)
AREA_CHANGE_THRESHOLD = 0.5

# =============================
# LOAD YOLOv5
# =============================
model = torch.hub.load(
    repo_or_dir="../yolov5",
    model="custom",
    path="../best.pt",
    source="local",
    force_reload=False
)

# Raised from 0.30 to 0.35 for fewer low-confidence candidates
model.conf = 0.35
class_names = model.names

# =============================
# LOAD CNN VERIFIER
# =============================
cnn = CNNVerifier(
    model_path="../verifier/cnn_model.pth",
    device="cuda" if torch.cuda.is_available() else "cpu",
    threshold=0.90  # High threshold for confident detections only
)

# =============================
# MULTI-FRAME VOTERS
# =============================
# YOLO-level voter (existing)
yolo_voter = MultiFrameVoter(window_size=3, vote_threshold=2)

# CNN-level voter (NEW: requires 2/3 frames for confirmation)
cnn_voter = MultiFrameVoter(window_size=3, vote_threshold=2)

# =============================
# FALLBACK DETECTOR (STRICTER)
# =============================
fallback = FallbackAnomalyDetector(threshold=5.0)  # Raised from 4.0

# Fallback temporal voting (9-frame window, require 5 votes)
fallback_votes = deque(maxlen=9)
fallback_cooldown = 0  # Frames to wait after a detection

# =============================
# INTER-FRAME TRACKING
# =============================
prev_boxes = []  # Store previous frame's valid boxes for stability check

def compute_iou(box1, box2):
    """Compute IoU between two boxes (x1,y1,x2,y2)."""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    
    inter_area = max(0, x2 - x1) * max(0, y2 - y1)
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    
    union_area = box1_area + box2_area - inter_area
    return inter_area / (union_area + 1e-6)

def find_matching_prev_box(box, prev_boxes, iou_threshold=0.3):
    """Find best matching box from previous frame."""
    best_iou = 0
    best_box = None
    for prev_box in prev_boxes:
        iou = compute_iou(box, prev_box)
        if iou > best_iou:
            best_iou = iou
            best_box = prev_box
    if best_iou >= iou_threshold:
        return best_box
    return None

def box_area(box):
    return (box[2] - box[0]) * (box[3] - box[1])

def box_touches_border(box, w, h, margin=BORDER_MARGIN):
    """Check if box touches frame border within margin."""
    x1, y1, x2, y2 = box
    return (x1 <= margin or y1 <= margin or 
            x2 >= w - margin or y2 >= h - margin)

# =============================
# VIDEO
# =============================
cap = cv2.VideoCapture("WhatsApp Video 2026-01-27 at 7.50.30 PM.mp4")
assert cap.isOpened(), "❌ Video not opened"

# Statistics
total_frames = 0
yolo_candidates = 0
cnn_verified = 0
cnn_rejected = 0
fallback_hits = 0
prev_time = time.time()
cnn_scores = []  # Track all CNN scores for analysis

# =============================
# MAIN LOOP
# =============================
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    total_frames += 1
    h, w, _ = frame.shape
    frame_area = h * w

    # -------- ROAD ROI --------
    crop_top = int(0.45 * h)
    crop_left = int(0.15 * w)
    crop_right = int(0.85 * w)

    roi = frame[crop_top:h, crop_left:crop_right]

    # -------- YOLO --------
    results = model(roi)
    detections = results.xyxy[0]

    valid_boxes = []

    for det in detections:
        x1, y1, x2, y2, conf, cls = det.tolist()
        x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])

        # Shift back to full frame coordinates
        x1 += crop_left
        x2 += crop_left
        y1 += crop_top
        y2 += crop_top

        bw = x2 - x1
        bh = y2 - y1
        area = bw * bh
        center_y = (y1 + y2) / 2

        # ---- STANDARD FILTERS ----
        if area < MIN_BOX_AREA_RATIO * frame_area:
            continue
        if area > MAX_BOX_AREA_RATIO * frame_area:
            continue
        if center_y < MIN_BOX_Y_RATIO * h:
            continue
        if bw < MIN_BOX_WIDTH or bh < MIN_BOX_HEIGHT:
            continue

        aspect_ratio = bw / (bh + 1e-6)
        if aspect_ratio > MAX_ASPECT_RATIO or aspect_ratio < MIN_ASPECT_RATIO:
            continue

        # ---- NEW: Border rejection ----
        if box_touches_border((x1, y1, x2, y2), w, h):
            continue

        # ---- NEW: Area stability check ----
        current_box = (x1, y1, x2, y2)
        prev_match = find_matching_prev_box(current_box, prev_boxes)
        if prev_match is not None:
            prev_area = box_area(prev_match)
            curr_area = box_area(current_box)
            area_change = abs(curr_area - prev_area) / (prev_area + 1e-6)
            if area_change > AREA_CHANGE_THRESHOLD:
                continue  # Reject unstable box

        valid_boxes.append((x1, y1, x2, y2, conf))

    # Update previous boxes for next frame
    prev_boxes = [(b[0], b[1], b[2], b[3]) for b in valid_boxes]

    # YOLO temporal voting
    detected = len(valid_boxes) > 0
    yolo_confirmed = yolo_voter.update(detected)

    # -------- FALLBACK WARMUP --------
    if total_frames <= 30:
        fallback.fit(roi)

    cnn_confirmed_any = False
    frame_had_yolo_boxes = len(valid_boxes) > 0

    # =============================
    # YOLO + CNN VERIFICATION
    # =============================
    if yolo_confirmed and detected:
        cnn_votes_this_frame = []
        
        for x1, y1, x2, y2, yolo_conf in valid_boxes:
            yolo_candidates += 1

            # Draw YOLO candidate (yellow)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 2)
            cv2.putText(frame, f"YOLO ({yolo_conf:.2f})",
                        (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (0, 255, 255), 2)

            patch = frame[y1:y2, x1:x2]
            if patch.size == 0:
                continue

            # Get CNN score
            verified, cnn_score = cnn.verify_with_score(patch)
            cnn_scores.append(cnn_score)  # Track for analysis
            cnn_votes_this_frame.append(verified)

            if verified:
                cnn_verified += 1  # Count raw CNN pass
                # Draw CNN verified (green)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)
                cv2.putText(frame, f"CNN VERIFIED ({cnn_score:.2f})",
                            (x1, y1 - 25),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.5, (0, 255, 0), 2)
            else:
                # Draw CNN rejected (red)
                cnn_rejected += 1
                cv2.putText(frame, f"CNN REJECTED ({cnn_score:.2f})",
                            (x1, y1 - 25),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.5, (0, 0, 255), 2)

        # CNN temporal voting: any box verified this frame?
        any_verified_this_frame = any(cnn_votes_this_frame)
        cnn_temporal_confirmed = cnn_voter.update(any_verified_this_frame)
        
        if cnn_temporal_confirmed and any_verified_this_frame:
            cnn_confirmed_any = True

    # =============================
    # FALLBACK (STRICT MODE)
    # =============================
    # ONLY run fallback if YOLO produced ZERO valid boxes AND no CNN confirmation
    if fallback_cooldown > 0:
        fallback_cooldown -= 1
    
    if not frame_had_yolo_boxes and not cnn_confirmed_any:
        score = fallback.score(roi)
        fallback_votes.append(score > fallback.threshold)

        # Require 5/9 votes for fallback detection + cooldown
        if sum(fallback_votes) >= 5 and fallback_cooldown == 0:
            fallback_hits += 1
            fallback_cooldown = 15  # Wait 15 frames before next fallback
            cv2.putText(frame, "FALLBACK ANOMALY",
                        (30, 40),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1, (255, 0, 0), 3)
    else:
        # Reset fallback votes if YOLO detected something
        fallback_votes.clear()

    # -------- FPS --------
    fps = int(1 / max(time.time() - prev_time, 1e-6))
    prev_time = time.time()

    cv2.putText(frame, f"FPS: {fps}",
                (30, 80),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8, (0, 255, 255), 2)

    cv2.imshow("Road Anomaly Detection", frame)
    if cv2.waitKey(20) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()

print("\n==== FINAL RESULTS ====")
print("Frames processed :", total_frames)
print("YOLO candidates  :", yolo_candidates)
print("CNN verified     :", cnn_verified)
print("CNN rejected     :", cnn_rejected)
print("Fallback detects :", fallback_hits)
if cnn_scores:
    import statistics
    print("\n---- CNN Score Analysis ----")
    print(f"Min score   : {min(cnn_scores):.3f}")
    print(f"Max score   : {max(cnn_scores):.3f}")
    print(f"Mean score  : {statistics.mean(cnn_scores):.3f}")
    print(f"Median score: {statistics.median(cnn_scores):.3f}")
print("======================")