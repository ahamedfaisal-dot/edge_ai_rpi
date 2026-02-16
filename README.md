# Edge AI Road Anomaly Detection System (Main Branch)

## Project Overview

This project implements a real-time road anomaly detection system designed for edge devices specially for Raspberry Pi 4. The pipeline combines deep learning–based detection with statistical fallback anomaly validation to achieve robust performance under real-world road conditions.

The main branch contains the final optimized ONNX-based inference pipeline with structured preprocessing, post-processing filters, temporal voting, and optional anomaly detection.

---

## Repository Branch Structure

This repository maintains multiple branches to show the evolution of the detection pipeline.

### main — Final Optimized Pipeline (This Branch)

- Final working and optimized implementation
- ONNX Runtime–based inference
- Structured preprocessing and post-processing
- Temporal voting and statistical fallback detection
- Recommended for demo, evaluation, and deployment

---

### First_approach — Baseline Pipeline

- Initial YOLO-based detection pipeline
- ROI filtering and basic false-positive reduction
- Minimal optimization
- See branch README for details

---

### Second_approach — Multi-Stage Verification Pipeline

- Enhanced pipeline with neural verification stage
- Strong false-positive reduction design
- Multi-stage filtering and validation
- See branch README for details

---

## Main Branch Processing Pipeline

Video Input  
→ Frame Preprocessing  
→ ONNX Model Inference  
→ Post-Processing Filters  
→ Non-Maximum Suppression  
→ Temporal Voting  
→ Optional Statistical Anomaly Detection  
→ Visualization

---

## Frame Preprocessing

Each frame is prepared before inference:

- Downscale frame for performance
- Resize to 320×320
- Convert BGR to RGB
- Normalize pixel values to [0–1]
- Convert to model input format

Purpose:
- Standardize inputs
- Improve inference stability
- Reduce compute cost

---

## Model Inference

- Model format: ONNX
- Detector: YOLO-based
- Input: 320×320 RGB image
- Backend: ONNX Runtime (CPU provider)
- Edge-device optimized

Output:
- Bounding boxes
- Confidence scores

---

## Post-Processing

Raw detections are filtered using:

- Confidence threshold filtering
- Bounding box scaling to original frame
- Bounding box size validation
- Removal of unrealistic boxes

Purpose:
- Remove weak predictions
- Reduce noise detections
- Improve reliability

---

## Non-Maximum Suppression (NMS)

Removes duplicate overlapping detections.

Steps:
- Sort by confidence
- Keep highest-confidence box
- Remove overlapping boxes above IoU threshold

Purpose:
- Avoid duplicate boxes
- Improve counting accuracy

---

## Temporal Voting

Multi-frame confirmation is applied.

Configuration:
- Window size: 3 frames
- Required votes: 2

Purpose:
- Reduce single-frame false positives
- Improve temporal stability

---

## Optional Statistical Anomaly Detection

Runs only when the main detector finds no valid detections.

Methods used:

Mahalanobis Detector:
- Statistical outlier detection using feature distance

Grayscale Detector:
- Brightness deviation detection using Z-score

Purpose:
- Detect unknown anomaly patterns
- Provide fallback safety check

---

## Visualization Output

Runtime display includes:

- Bounding boxes
- Confidence values
- FPS counter
- Frame counter
- Detection counter
- Processing time

---

## Performance Targets

Target hardware: Raspberry Pi class edge devices

Typical performance:

- Target FPS: 5
- Processing time: ~50–100 ms per frame
- Inference resolution: 320×320

Optimizations include:

- Frame downscaling
- ONNX Runtime execution
- CPU-friendly configuration

---

## File Structure

onnx/
- best.onnx
- runvid.py
- mahalanobis_detector.py
- fallback_anomaly.py
- sample video files

---

## Installation

Install dependencies:

pip install opencv-python numpy onnxruntime

Optional GPU runtime:

pip install onnxruntime-gpu

---

## Usage

Run the pipeline:

python runvid.py

Controls:

- Press q to exit
- Ctrl+C for termination

---

## Use Cases

- Road infrastructure monitoring
- Edge AI vision systems
- Smart transportation analytics
- Real-time anomaly detection research

---

## Notes

- Training code and dataset are not included
- Earlier designs are documented in other branches
- See branch-specific READMEs for approach details

---

## Authors

Ahamed Faisal A  
Sanji Krishna M P  
Subiksha A

Project Status: Active Development  
Target Platform: Edge Devices
