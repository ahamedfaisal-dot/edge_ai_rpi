# Edge AI Road Anomaly Detection System for Raspberry Pi — Final ONNX Pipeline

## Project Overview

This project implements a real-time pothole and road anomaly detection system optimized for edge devices, specifically Raspberry Pi 4. The system uses an ONNX-based deep learning detector combined with structured post-processing, temporal voting, and statistical anomaly validation to achieve stable and efficient real-world performance.

The main branch represents the final optimized pipeline after multiple architectural iterations, focusing on speed, robustness, and deployment readiness.

---

## Branch Evolution Summary

This repository contains three major pipeline stages:

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

---

# System Architecture — Main Branch

## High-Level Pipeline

```
Video Input
   → Frame Preprocessing
   → ONNX Model Inference
   → Post-Processing Filters
   → Non-Maximum Suppression
   → Temporal Voting
   → Optional Statistical Anomaly Detection
   → Visualization
```

---

## Architecture Diagram

```
VIDEO INPUT
   │
   ▼
FRAME PREPROCESSING
• Downscale (45%)
• Resize to 320×320
• BGR → RGB
• Normalize [0,1]
   │
   ▼
ONNX MODEL INFERENCE
• YOLO-based ONNX model
• CPU execution provider
• Raw detections
   │
   ▼
POST-PROCESSING
• Confidence filtering
• Bounding box scaling
• Size validation
• NMS
   │
   ▼
TEMPORAL VOTING
• 3-frame window
• 2/3 consensus
   │
   ▼
OPTIONAL ANOMALY DETECTION
• Mahalanobis detector
• Grayscale Z-score detector
   │
   ▼
VISUALIZATION
• Boxes + scores
• FPS + metrics
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

# Authors

Ahamed Faisal A  
Sanji Krishna M P  
Subiksha A

Target Hardware: Raspberry Pi 4  
Project Status: Active Development
