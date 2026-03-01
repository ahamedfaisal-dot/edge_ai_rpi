# Edge AI Road Anomaly Detection System for Raspberry Pi — Final ONNX Pipeline

## Project Overview

This project implements a road anomaly detection system optimized for edge devices, specifically Raspberry Pi 4. The system uses an ONNX-based deep learning detector combined with structured post-processing, temporal voting, and statistical anomaly validation to achieve stable and efficient real-world performance.

---

The main branch represents the final optimized pipeline after multiple architectural iterations, focusing on speed, robustness, and deployment readiness.

final output : [View Real-World Bike Testing Video (Google Drive)](https://drive.google.com/file/d/1fmIOCZe0yc1K5nB2kklY15GKl-EBwTE5/view?usp=sharing)
---
## Anomalies Our Project will detect

The proposed system is designed to detect multiple categories of road surface anomalies, including longitudinal cracks, transverse cracks, alligator cracks, and potholes. These damage types are commonly classified into different severity levels such as D00, D10, D20, and D40, representing increasing degrees of structural deterioration. Longitudinal and transverse cracks indicate surface-level stress and material fatigue, while alligator cracks represent interconnected crack patterns caused by repeated vehicular loading. Potholes correspond to severe surface failure resulting from prolonged crack propagation and environmental exposure. By accurately identifying and classifying these anomalies, the system enables efficient monitoring of road conditions and supports timely maintenance decisions.

---

## Branch Evolution Summary

This repository contains three major pipeline stages across different branches:

**First_approach — Baseline**
- YOLO detection + ROI cropping
- Basic rule filters
- Simple temporal voting
- Unsupervised fallback
- Minimal optimization

**Second_approach — Multi-Stage Verification**
- YOLO candidate generator
- Multi-stage strict filters
- CNN appearance verifier
- Dual temporal voting
- Strong false-positive reduction

**main — Final ONNX Edge Pipeline (This Branch)**
- ONNX Runtime inference
- Optimized preprocessing
- Structured post-processing + NMS
- Temporal voting
- Statistical anomaly fallback
- Edge deployment optimized

> **Output videos and performance results from all pipeline stages running on Raspberry Pi 4 hardware are documented in the [`Output_videos_in_PI4`](Output_videos_in_PI4) folder.**

---

# System Architecture — Final Pi Camera Pipeline

The architecture below reflects the **live Pi Camera deployment** (`run_with_pi_cam.py`) inside [`Final_Output_With_PI4_Cam_on_realworld/`](./Final_Output_With_PI4_Cam_on_realworld/README.md), which is the fully field-validated pipeline running on Raspberry Pi 4 with a real camera feed.

## High-Level Pipeline

```
rpicam-vid (subprocess)
   → YUV420 stdout pipe
   → YUV → BGR conversion
   → Frame Downscaling
   → Preprocessing (resize / normalise)
   → YOLOv5 ONNX Inference
   → Post-Processing (conf + area + aspect filters)
   → NMS
   → Temporal Voting
   → Visualisation + Logging (log / CSV / SQLite / JSON / JPEG)
```

---

## Architecture Diagram — Full Detailed Pipeline

```
┌──────────────────────────────────────────────────────────────────┐
│                   rpicam-vid SUBPROCESS                          │
│  Command:                                                        │
│    rpicam-vid --width 640 --height 480                           │
│               --framerate 5 --codec yuv420                       │
│               --nopreview --timeout 0 --output -                 │
│                                                                  │
│  • Launched via subprocess.Popen                                 │
│  • Raw YUV420 (I420) data written to stdout pipe                 │
│  • Uses libcamera driver — works with Pi Camera v2               │
│  • 2-second warm-up after launch before reading starts           │
│  • Exits if rpicam-vid process terminates unexpectedly           │
└───────────────────────────┬──────────────────────────────────────┘
                            │  stdout pipe (YUV420 bytes)
                            ▼
┌──────────────────────────────────────────────────────────────────┐
│                   FRAME READING (read_frame)                     │
│  • FRAME_BYTES = 640 × 480 × 3 // 2  (YUV420 I420 layout)        │
│  • Uses readinto loop into a fixed bytearray buffer              │
│  • Guarantees exactly one full frame per call                    │
│  • Returns None if pipe closes (triggers clean shutdown)         │
│  • Converts: YUV420 → BGR via cv2.COLOR_YUV2BGR_I420             │
└───────────────────────────┬──────────────────────────────────────┘
                            │  BGR frame (480 × 640 × 3)
                            ▼
┌──────────────────────────────────────────────────────────────────┐
│                      FRAME DOWNSCALING                           │
│  • Scale factor: FRAME_SCALE = 0.45                              │
│    640 → ~288 px wide,  480 → ~216 px tall                       │
│  • Interpolation: INTER_NEAREST (fastest on RPi CPU)             │
│  • Reduces pixel count ~80% before inference                     │
└───────────────────────────┬──────────────────────────────────────┘
                            │
                            ▼
┌──────────────────────────────────────────────────────────────────┐
│                     PREPROCESSING (preprocess)                   │
│  1. Resize to 320×320  (INFER_SIZE × INFER_SIZE)                 │
│     using INTER_NEAREST                                          │
│  2. BGR → RGB  (flip channel axis [::-1])                        │
│  3. HWC → CHW  (transpose to PyTorch tensor layout)              │
│  4. uint8 → float32,  divide by 255  → [0.0, 1.0]                │
│  5. Add batch dim: shape becomes (1, 3, 320, 320)                │
└───────────────────────────┬──────────────────────────────────────┘
                            │
                            ▼
┌──────────────────────────────────────────────────────────────────┐
│                   YOLOv5 ONNX INFERENCE                          │
│  • Model file  : best.onnx  (YOLOv5, exported to ONNX)           │
│  • Provider    : CPUExecutionProvider                            │
│  • intra_op_num_threads = 4  (all 4 RPi4 cores)                  │
│  • inter_op_num_threads = 1                                      │
│  • Execution mode : ORT_SEQUENTIAL                               │
│  • Graph opt level: ORT_ENABLE_ALL                               │
│  • Output shape: (1, N, ≥5)                                      │
│    Each row = [x_centre, y_centre, width, height, conf, cls...]  │
└───────────────────────────┬──────────────────────────────────────┘
                            │
                            ▼
┌──────────────────────────────────────────────────────────────────┐
│              POST-PROCESSING + FILTERS (postprocess)             │
│                                                                  │
│  Filter 1 — Confidence:                                          │
│    Discard if conf < CONF_THRES (0.40)                           │
│    (raised from 0.20 to eliminate weak false positives)          │
│                                                                  │
│  Filter 2 — Scale & convert:                                     │
│    sx = frame_w / 320,  sy = frame_h / 320                       │
│    Convert centre-format → corners (x1, y1, x2, y2)              │
│    Clamp to frame boundary                                       │
│                                                                  │
│  Filter 3 — Size:                                                │
│    Discard if box width or height < 10 px or > 1000 px           │
│                                                                  │
│  Filter 4 — Area ratio:                                          │
│    Discard if box area > 30% of total frame area                 │
│    (prevents huge false boxes covering entire scene)             │
│                                                                  │
│  Filter 5 — Aspect ratio:                                        │
│    aspect = box_width / box_height                               │
│    Discard if aspect < 0.3  (too tall/thin)                      │
│    Discard if aspect > 3.0  (too wide/flat)                      │
│    (potholes are roughly equant or slightly wide)                │
└───────────────────────────┬──────────────────────────────────────┘
                            │
                            ▼
┌──────────────────────────────────────────────────────────────────┐
│                  NON-MAXIMUM SUPPRESSION (nms)                   │
│  • Sort boxes by confidence (descending)                         │
│  • Greedy pick: always keep the highest-confidence box           │
│  • Remove overlapping boxes where IoU > IOU_THRES (0.4)          │
│  • Repeat until box list is empty                                │
│                                                                  │
│    IoU = Intersection Area / Union Area                          │
│                                                                  │
│  Purpose: deduplicate multiple detections on same pothole        │
└───────────────────────────┬──────────────────────────────────────┘
                            │
                            ▼
┌──────────────────────────────────────────────────────────────────┐
│              TEMPORAL VOTING (MultiFrameVoter)                   │
│  Imported from: temporal_voting.py (parent project)              │
│                                                                  │
│  • Maintains a sliding window of the last 3 frames               │
│  • Detection confirmed only if ≥ 2 of 3 frames detect something  │
│  • Window size = 3,  vote threshold = 2  (configurable)          │
│  • Rejects single-frame camera glitches / lighting artifacts     │
│  • No "sticky latch" — clean binary output each frame            │
└───────────────────────────┬──────────────────────────────────────┘
                            │
                ┌───────────┴──────────────┐
                │                          │
                ▼                          ▼
┌──────────────────────────┐  ┌────────────────────────────────────┐
│    VISUALISATION (draw)  │  │   DETECTION LOGGING (log_detection)│
│                          │  │                                    │
│ • Green boxes on frame   │  │ Triggered only when BOTH:          │
│   for every detection    │  │   confirmed=True AND detections>0  │
│ • "POTHOLE X.XX" label   │  │                                    │
│   above each box         │  │ Outputs:                           │
│ • Red "POTHOLE DETECTED!"│  │  .log  — Python logging (INFO)     │
│   banner at top-left     │  │  .csv  — row per confirmed event   │
│ • HUD (bottom of frame): │  │  .db   — SQLite WAL, buffered 10x  │
│   – FPS (cyan)           │  │  .jpg  — JPEG snapshot of frame    │
│   – Frame # (yellow)     │  │          (85% quality)             │
│   – Total dets (magenta) │  │                                    │
│   – Proc time ms (orange)│  │ Columns logged:                    │
│ • Resized to 640×360     │  │  timestamp, frame_id, num_boxes,   │
│   for stable display     │  │  max_conf, boxes_json, fps,        │
│ • Window: "Pothole       │  │  process_ms                        │
│   Detection"             │  │                                    │
└──────────────────────────┘  └────────────────────────────────────┘
                            │
                            ▼
┌──────────────────────────────────────────────────────────────────┐
│                       FPS THROTTLE                               │
│  • TARGET_FRAME_TIME = 1.0 / 5 = 200 ms                          │
│  • Sleep remainder after each loop iteration                     │
│  • cv2.waitKey handles both sleep and key press check            │
│  • Console log every 2 seconds: frame/fps/proc/detections        │
└──────────────────────────────────────────────────────────────────┘
                            │
                 [q pressed / Ctrl+C / pipe closed]
                            ▼
┌──────────────────────────────────────────────────────────────────┐
│                    CLEAN SHUTDOWN (finally)                      │
│  1. cam_proc.terminate() + cam_proc.wait()  — kill rpicam-vid    │
│  2. cv2.destroyAllWindows()                                      │
│  3. Flush SQLite buffer, close CSV file                          │
│  4. Write summary_<SESSION>.json                                 │
│  5. Print final statistics to console                            │
└──────────────────────────────────────────────────────────────────┘
```
---

# Stage Explanations and Use Cases

## Frame Preprocessing

Steps:
- Downscale frame for speed
- Resize to model input size (320×320)
- Convert BGR → RGB
- Normalize pixels to [0,1]

Use:
- Ensures model input consistency
- Reduces computation
- Improves inference stability

---

## ONNX Model Inference

- Model: YOLO-based detector exported to ONNX
- Backend: ONNX Runtime CPU provider
- Output: bounding boxes + confidence

Why ONNX:
- Faster CPU inference
- Smaller runtime footprint
- Cross-platform deployment
- Better edge suitability than full PyTorch runtime

---

## Post-Processing Filters

Applied to raw model outputs:

- Confidence threshold (> 0.2)
- Bounding box scaling to original frame
- Bounding box dimension validation (10–1000 px)

Use:
- Remove weak detections
- Remove unrealistic boxes
- Improve reliability before NMS

---

## Non-Maximum Suppression (NMS)

Purpose: remove duplicate overlapping detections.

Algorithm:
1. Sort boxes by confidence
2. Keep highest confidence box
3. Remove overlapping boxes above IoU threshold
4. Repeat

**IoU Formula**

Intersection over Union:

```
IoU = Intersection Area / Union Area
```

Use:
- Prevent multiple boxes on same pothole
- Improve counting accuracy

---

## Temporal Voting

Multi-frame confirmation logic.

Configuration:
- Window size = 3 frames
- Required votes = 2

Algorithm:
- Track detection presence per frame
- Confirm only if votes ≥ threshold

Use:
- Remove flicker detections
- Reduce transient false positives
- Improve temporal stability

---

# Statistical Anomaly Detection (Fallback Layer)

Runs only when detector finds no valid boxes.

---

## Mahalanobis Distance Detector

Statistical outlier detection using feature distance.

**Formula**

```
D_M(x) = √[(x − μ)ᵀ Σ⁻¹ (x − μ)]
```

Where:
- x = feature vector
- μ = mean vector
- Σ = covariance matrix
- Σ⁻¹ = inverse covariance

Use:
- Compare region features vs normal road patterns
- Reject abnormal statistical deviations
- Reduce false positives from shadows and markings

Regularization used:
```
Σ = Σ + 1e-6 I
```

(prevents singular covariance)

---

## Grayscale Z-Score Detector

Lightweight anomaly scoring using brightness deviation.

**Formula**

```
Z = |value − mean| / std
```

Anomaly if:
```
Z > threshold
```

Use:
- Detect unusual intensity patterns
- Fast fallback method
- Very low compute cost

---

# System Components

## runvid.py — Main Pipeline

Responsibilities:
- Video capture
- Preprocessing
- ONNX inference
- Post-processing
- NMS
- Temporal voting
- Visualization
- FPS control

Key parameters:

| Parameter | Value |
|------------|---------|
TARGET_FPS | 5 |
FRAME_SCALE | 0.45 |
CONF_THRES | 0.2 |
IOU_THRES | 0.4 |
INFER_SIZE | 320 |

---

## mahalanobis_detector.py

- Statistical anomaly model
- Feature collection + covariance
- Distance scoring

Methods:
- fit()
- finalize()
- score()

---

## fallback_anomaly.py

- Grayscale mean feature extraction
- Running mean + std
- Z-score anomaly scoring

---

# Performance Optimization Techniques

## Frame Optimization
- Downscale to 45%
- Fast interpolation mode

## Inference Optimization
- Input size 320×320
- Batch size 1
- CPU thread tuning
- ONNX graph optimization enabled

## Memory Optimization
- Single-frame buffers
- Rolling window metrics
- Contiguous arrays

---

# Performance Achievements

| Test Phase | Hardware | FPS | Performance vs Laptop | Notes |
|-------------|------------|------|------------------------|-------|
Development | Laptop | 25–45 | 100% | Baseline |
Initial Edge | Raspberry Pi | 1 | ~2% | Unoptimized |
First Optimization | Raspberry Pi | 4 | ~9% | 4× gain |
Final Optimization | Raspberry Pi | 7 | ~16% | 7× gain |
Real-World Test | Raspberry Pi | ~7 | ~16% | Field validated |

---

# Comparison vs Earlier Architectures

| Aspect | First | Second | Main |
|---------|---------|----------|---------|
Runtime | PyTorch | PyTorch | ONNX |
Verification | Basic | CNN + filters | Filters + stats |
Temporal logic | Basic | Dual | Optimized |
False positives | Medium | Low | Low + efficient |
Edge speed | Poor | Moderate | Strong |
Deployment | Limited | Moderate | Ready |

---

# File Structure

```
onnx/
  best.onnx
  runvid.py
  mahalanobis_detector.py
  fallback_anomaly.py
  pothole2.mp4
README.md
```

---

# Installation

```
pip install opencv-python numpy onnxruntime
```

Optional GPU:

```
pip install onnxruntime-gpu
```

---

# Usage

```
python runvid.py
```

Controls:
- q → quit
- Ctrl+C → safe stop

---
# Dataset-Links
1>https://universe.roboflow.com/road-damage-detection/rdd-rs9zu
2>https://universe.roboflow.com/new-workspace-cwddl/rdd-2020
---

# Authors

Ahamed Faisal A  
Sanji Krishna M P  
Subiksha A

Target Hardware: Raspberry Pi 4  
Project Status: Active Development
