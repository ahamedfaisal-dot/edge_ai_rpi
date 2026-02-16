# Road Anomaly Detection on Edge — First Approach

## Project Overview

This branch contains the first implementation of a real-time road anomaly detection system optimized for Raspberry Pi 4. The pipeline detects both known anomalies (such as potholes and cracks) and unknown road irregularities using a hybrid supervised and unsupervised detection strategy.

This version represents the baseline architecture before later optimization and multi-stage verification improvements.

---

## Objectives

- Achieve at least 5 FPS on Raspberry Pi 4
- Minimize false positives
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

## System Architecture

The first approach uses a single-stage detection pipeline with rule-based filtering and temporal validation.

Processing flow:

Dashcam Video Input  
→ ROI Cropping (road-only region)  
→ YOLOv5n Detection  
→ Bounding Box Size and Position Filters  
→ Multi-Frame Temporal Voting  
→ Fallback Unsupervised Anomaly Detector  
→ Final Decision and Visualization

---

## Key Design Features

### Hybrid Detection Strategy
The system combines:
- Supervised detection using YOLOv5n for known anomaly types
- Unsupervised fallback detection for unknown anomalies

This allows detection beyond only trained classes.

---

## False Positive Reduction Techniques

The following controls are applied to reduce incorrect detections:

- Road-only ROI cropping to remove sky and roadside regions
- YOLO confidence threshold filtering
- Bounding box size filtering
- Bounding box position filtering
- Multi-frame temporal voting
- Fallback anomaly verification

---

## Repository Structure (First Approach)

```
yolov5/        YOLOv5 framework
inference/     Real-time detection pipeline
best.pt        Trained YOLOv5n weights
requirements.txt
README.html    Project documentation
```

---

## Setup Instructions

Clone the repository:

```bash
git clone <your-repo-link>
cd road_anomaly_project
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Place model weights:

Put `best.pt` inside:

```
yolov5/runs/train/road_anomaly_yolov5n_final/weights/
```

---

## Running Inference

Navigate to the inference folder:

```bash
cd inference
```

Run detection:

```bash
python detect_video.py
```

Controls:

- Press Q to exit the display window

---

## Output

The system provides:

- Real-time bounding boxes on detected anomalies
- YOLO-confirmed anomalies highlighted separately
- Fallback-detected anomalies highlighted separately
- FPS counter
- Detection summary printed in the terminal

---

## Notes

This branch represents the baseline version of the project pipeline.  
Later branches introduce additional verification stages and stronger false-positive reduction mechanisms.

Refer to the main branch and second approach branch for the optimized and multi-stage verification architectures.

---

## Project Context

Developed as part of an Edge AI road anomaly detection project targeting real-time deployment on Raspberry Pi–class devices.
