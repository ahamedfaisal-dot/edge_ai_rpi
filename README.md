# Road Anomaly Detection on Edge — First Approach (Baseline Pipeline)

## Project Overview

This branch contains the first implementation of a real-time road anomaly detection system optimized for Raspberry Pi 4. The pipeline detects both known anomalies (such as potholes and cracks) and unknown road irregularities using a hybrid supervised and unsupervised detection strategy.

This version represents the baseline architecture before later optimization and multi-stage verification improvements.

---

## Objectives

- Achieve at least 5 FPS on Raspberry Pi 4  
- Reduce false positives  
- Detect both known and previously unseen road anomalies  
- Maintain a lightweight, edge-friendly pipeline  

---

## Technologies Used

- Python 3  
- YOLOv5n (primary detector)  
- OpenCV  
- PyTorch  
- NumPy  
- Raspberry Pi 4 (target platform)

---
## System Architecture — First Approach (Baseline Pipeline)

The first approach uses a lightweight single-stage detection pipeline with rule-based filtering, temporal validation, and an unsupervised fallback detector. The design focuses on simplicity, edge-device feasibility, and basic false-positive reduction.

### Processing Pipeline Diagram

```
┌──────────────────────────────────────────────┐
│               DASHCAM VIDEO INPUT            │
└─────────────────────────┬────────────────────┘
                          ▼
┌──────────────────────────────────────────────┐
│            ROI CROPPING (ROAD ONLY)          │
└─────────────────────────┬────────────────────┘
                          ▼
┌──────────────────────────────────────────────┐
│              YOLOv5n DETECTION               │
└─────────────────────────┬────────────────────┘
                          ▼
┌──────────────────────────────────────────────┐
│        BOUNDING BOX RULE FILTERS             │
└─────────────────────────┬────────────────────┘
                          ▼
┌──────────────────────────────────────────────┐
│        TEMPORAL VOTING (MULTI-FRAME)         │
└─────────────────────────┬────────────────────┘
                          ▼
┌──────────────────────────────────────────────┐
│   FALLBACK ANOMALY DETECTOR (UNSUPERVISED)   │
└─────────────────────────┬────────────────────┘
                          ▼
┌──────────────────────────────────────────────┐
│        FINAL DECISION & VISUALIZATION        │
└──────────────────────────────────────────────┘
```

---

## Stage-by-Stage Explanation and Use Cases

### Dashcam Video Input

**What it does**
- Captures live or recorded road video
- Provides frame-by-frame input to the detection pipeline

**Use case**
- Vehicle-mounted cameras
- Road survey recordings
- Edge AI monitoring systems

**Why it matters**
All downstream detection depends on frame quality and continuity.

---

### ROI Cropping (Road-Only Region)

**What it does**
- Crops the frame to keep only the road region
- Removes sky, buildings, trees, and roadside objects

**Use case**
- When camera angle is fixed and road occupies a known region
- Edge systems where compute must be minimized

**Why it is used**
- Reduces false positives from non-road objects
- Reduces pixels processed → improves speed
- Focuses detection only where potholes can exist

---

### YOLOv5n Detection (Primary Detector)

**What it does**
- Runs supervised object detection
- Detects known anomaly classes like potholes and cracks
- Produces bounding boxes with confidence scores

**Use case**
- Fast candidate detection on edge hardware
- Real-time anomaly spotting

**Why it is used**
- YOLOv5n is lightweight and fast
- Suitable for Raspberry Pi–class devices
- Provides high recall for anomaly candidates

**Important note**
This stage is permissive — later filters clean incorrect detections.

---

### Bounding Box Rule Filters

**What it does**
Applies rule-based checks on detected boxes:

- Confidence threshold
- Minimum and maximum size limits
- Position constraints (road region only)

**Use case**
- Remove detections that are geometrically unrealistic
- Filter noise before temporal validation

**Why it is used**
YOLO may detect:
- Shadows
- Lane markings
- Small noise patches

Filters remove these early to reduce false positives.

---

### Temporal Voting (Multi-Frame Validation)

**What it does**
- Checks whether a detection appears across multiple consecutive frames
- Confirms only if detection is repeated

**Use case**
- Video-based detection systems
- Dashcam streams with continuous frames

**Why it is used**
Single-frame detections are often noise due to:
- Lighting flicker
- Motion blur
- Sensor noise

Temporal voting improves stability and reliability.

Example rule:
Detection must appear in at least 2 out of 3 frames.

---

### Fallback Anomaly Detector (Unsupervised)

**What it does**
- Runs only if YOLO produces no valid detections
- Uses statistical or visual deviation checks
- Flags unusual road patterns

**Use case**
- Detect unknown anomaly types
- Handle anomalies not present in YOLO training data

**Why it is used**
Supervised models can only detect trained classes.  
Fallback detection provides limited coverage for unseen anomalies.

Examples:
- Broken patches
- Debris
- Surface irregularities

---

### Final Decision and Visualization

**What it does**
- Draws bounding boxes on confirmed detections
- Displays FPS and counters
- Shows detection summary

**Use case**
- Real-time monitoring dashboards
- Demo and evaluation runs
- Edge device visualization

**Why it is used**
Provides human-interpretable output and runtime feedback.

---

## Practical Pipeline Behavior

In real operation:

- ROI cropping reduces irrelevant regions
- YOLO proposes anomaly candidates
- Rule filters remove obvious mistakes
- Temporal voting removes flicker detections
- Fallback detection handles unseen anomaly patterns
- Only stable detections are displayed

This makes the baseline pipeline lightweight, explainable, and edge-feasible.

---

## False Positive Reduction Methods

- Road-only ROI cropping  
- YOLO confidence threshold filtering  
- Bounding box size filtering  
- Bounding box position filtering  
- Multi-frame temporal voting  
- Fallback anomaly verification  

---

## Repository Structure (First Approach)

```
yolov5/        YOLOv5 framework
inference/     Detection pipeline scripts
best.pt        Trained YOLOv5n weights
requirements.txt
README.html
```

---

## Setup

Clone repository:

```bash
git clone <your-repo-link>
cd road_anomaly_project
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Place model weights:

```
yolov5/runs/train/road_anomaly_yolov5n_final/weights/best.pt
```

---

## Running Inference

```bash
cd inference
python detect_video.py
```

Press `Q` to exit the display window.

---

## Notes

This branch contains the baseline pipeline.  
Later branches introduce multi-stage verification and stronger false-positive reduction mechanisms.

Refer to the main branch and second approach branch for optimized and advanced architectures.

---

## Project Context

Developed as part of an Edge AI road anomaly detection project targeting real-time deployment on Raspberry Pi–class devices.
