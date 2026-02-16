# Edge AI Road Anomaly Detection System — Main Branch (Final Optimized ONNX Pipeline)

## Project Overview

This project implements a real-time road anomaly and pothole detection system optimized for edge devices, specifically Raspberry Pi 4. The pipeline combines ONNX-based deep learning detection with structured post-processing, temporal voting, and statistical fallback anomaly validation to achieve stable performance under real-world road conditions.

The main branch represents the final optimized architecture after iterative improvements over the first and second approaches, with a strong focus on speed, deployment readiness, and reduced false positives.

---

## Repository Branch Evolution

This repository contains multiple branches representing the evolution of the detection pipeline.

### main — Final Optimized ONNX Pipeline (This Branch)
- ONNX Runtime–based inference
- Edge-optimized preprocessing and execution
- Structured post-processing + NMS
- Temporal voting
- Statistical fallback anomaly detection
- Deployment-ready pipeline

### First_approach — Baseline Pipeline
- YOLO-based detection
- ROI cropping + rule filters
- Basic temporal voting
- Unsupervised fallback
- Minimal optimization

### Second_approach — Multi-Stage Verification Pipeline
- YOLO candidate generator
- Multi-stage strict filters
- CNN appearance verifier
- Dual temporal voting
- Strong false-positive reduction
- Heavier but more strict pipeline

---

## System Architecture — Main Branch

This branch uses an ONNX-based edge inference pipeline with standardized preprocessing, structured filtering, temporal validation, and optional statistical anomaly detection.

### End-to-End Architecture Diagram

```
┌──────────────────────────────────────────────┐
│                  VIDEO INPUT                 │
│              (Dashcam / File)                │
└─────────────────────────┬────────────────────┘
                          ▼
┌──────────────────────────────────────────────┐
│              FRAME PREPROCESSING             │
│  • Downscale frame                           │
│  • Resize to 320×320                         │
│  • BGR → RGB conversion                      │
│  • Normalize to [0,1]                        │
└─────────────────────────┬────────────────────┘
                          ▼
┌──────────────────────────────────────────────┐
│              ONNX MODEL INFERENCE            │
│  • YOLO-based ONNX model                     │
│  • CPU execution provider                    │
│  • Candidate bounding boxes                  │
└─────────────────────────┬────────────────────┘
                          ▼
┌──────────────────────────────────────────────┐
│               POST-PROCESSING                │
│  • Confidence filtering                      │
│  • Box scaling to frame                      │
│  • Size validation                           │
│  • Non-Maximum Suppression                   │
└─────────────────────────┬────────────────────┘
                          ▼
┌──────────────────────────────────────────────┐
│               TEMPORAL VOTING                │
│  • 3-frame window                            │
│  • 2/3 consensus rule                        │
│  • Flicker rejection                         │
└─────────────────────────┬────────────────────┘
                          ▼
┌──────────────────────────────────────────────┐
│        OPTIONAL ANOMALY DETECTION            │
│  • Mahalanobis statistical detector          │
│  • Grayscale deviation detector              │
│  • Runs only if YOLO = no detections         │
└─────────────────────────┬────────────────────┘
                          ▼
┌──────────────────────────────────────────────┐
│                VISUALIZATION                 │
│  • Bounding boxes                            │
│  • Confidence scores                         │
│  • FPS and metrics                           │
└──────────────────────────────────────────────┘
```

---

## Stage Use and Purpose

### Frame Preprocessing
Standardizes every frame before inference.

Use:
- Ensures model input consistency  
- Reduces compute load  
- Matches ONNX model expectations  

---

### ONNX Model Inference
Runs YOLO-based detector in ONNX Runtime.

Use:
- Lightweight edge inference  
- Faster startup and execution vs PyTorch runtime  
- Cross-platform deployment  

---

### Post-Processing + NMS
Cleans raw model outputs.

Use:
- Remove weak detections  
- Remove unrealistic box sizes  
- Merge duplicate overlapping boxes  
- Improve output quality  

---

### Temporal Voting
Validates detections across frames.

Use:
- Reject single-frame noise  
- Improve stability  
- Reduce false positives  

Rule:
Detection must appear in at least 2 of last 3 frames.

---

### Statistical Fallback Anomaly Detection
Runs only if detector finds nothing.

Methods:
- Mahalanobis distance outlier detection  
- Grayscale Z-score deviation detection  

Use:
- Catch unknown anomaly patterns  
- Provide backup safety layer  

---

## Comparison with Earlier Approaches

| Aspect | First Approach | Second Approach | Main Branch |
|---------|----------------|----------------|-------------|
YOLO role | Final detector | Candidate generator | Candidate detector |
Verification | Rule filters | Rule + CNN verifier | Structured filters + stats fallback |
Temporal logic | Basic | YOLO + CNN voting | Optimized temporal voting |
Model runtime | PyTorch | PyTorch | ONNX Runtime |
False positive control | Moderate | Very strict | Strong + efficient |
Edge performance | Low | Moderate | Optimized |
Deployment readiness | Limited | Moderate | High |

---

## Key Achievements and Performance Progress

| Test Phase | Hardware | FPS | Performance vs Laptop | Notes |
|-------------|------------|------|------------------------|-------|
Development | Laptop | 25–45 | 100% baseline | Optimal conditions |
Initial Edge | Raspberry Pi | 1 | ~2% | Unoptimized |
First Optimization | Raspberry Pi | 4 | ~9% | 4× improvement |
Final Optimization | Raspberry Pi | 7 | ~16% | 7× improvement |
Real-World Test | Raspberry Pi | ~7 | ~16% | Field validated |

---

## Performance Targets

- Target platform: Raspberry Pi 4  
- Target FPS: 5+  
- Typical FPS achieved: ~7  
- Inference size: 320×320  
- Processing time: ~50–100 ms/frame  

Optimizations include:
- Frame downscaling  
- Small inference resolution  
- ONNX Runtime execution  
- CPU thread tuning  

---

## File Structure

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

## Installation

Install dependencies:

```
pip install opencv-python numpy onnxruntime
```

Optional GPU runtime:

```
pip install onnxruntime-gpu
```

---

## Usage

Run detection:

```
python runvid.py
```

Controls:
- Press q to quit
- Ctrl+C to stop safely

---

## Authors

Ahamed Faisal A  
Sanji Krishna M P  
Subiksha A

Project Status: Active Development  
Target Platform: Raspberry Pi 4
