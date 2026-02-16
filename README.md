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

## System Architecture — First Approach

The first approach uses a single-stage detection pipeline with rule-based filtering, temporal validation, and an unsupervised fallback detector.

### Processing Pipeline Diagram

```
┌──────────────────────────────────────────────┐
│               DASHCAM VIDEO INPUT            │
│          (Live stream or video file)         │
└─────────────────────────┬────────────────────┘
                          │
                          ▼
┌──────────────────────────────────────────────┐
│            ROI CROPPING (ROAD ONLY)          │
│  • Keeps road region                         │
│  • Removes sky and background                │
│  • Reduces false positives                   │
└─────────────────────────┬────────────────────┘
                          │
                          ▼
┌──────────────────────────────────────────────┐
│              YOLOv5n DETECTION               │
│  • Detects known anomalies                   │
│  • Lightweight edge model                    │
│  • Generates candidate boxes                 │
└─────────────────────────┬────────────────────┘
                          │
                          ▼
┌──────────────────────────────────────────────┐
│        BOUNDING BOX RULE FILTERS             │
│  • Confidence threshold                      │
│  • Size filtering                            │
│  • Position filtering                        │
│  • Remove unrealistic detections             │
└─────────────────────────┬────────────────────┘
                          │
                          ▼
┌──────────────────────────────────────────────┐
│        TEMPORAL VOTING (MULTI-FRAME)         │
│  • Multi-frame confirmation                  │
│  • Reject single-frame noise                 │
│  • Improve stability                         │
└─────────────────────────┬────────────────────┘
                          │
                          ▼
┌──────────────────────────────────────────────┐
│   FALLBACK ANOMALY DETECTOR (UNSUPERVISED)   │
│  • Runs if YOLO finds nothing                │
│  • Detects unknown anomalies                 │
│  • Uses statistical deviation checks         │
└─────────────────────────┬────────────────────┘
                          │
                          ▼
┌──────────────────────────────────────────────┐
│        FINAL DECISION & VISUALIZATION        │
│  • Draw bounding boxes                       │
│  • Display FPS                               │ 
│  • Show detection summary                    │
└──────────────────────────────────────────────┘
```

---

## Detection Strategy

The system combines:

- Supervised detection using YOLOv5n for known anomaly classes  
- Unsupervised fallback detection for unknown anomaly patterns  

This allows the pipeline to detect anomalies beyond only trained categories.

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
