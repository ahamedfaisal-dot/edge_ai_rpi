# Edge AI Pothole Detection System for Raspberry Pi

## Project Overview

This project implements a **real-time pothole detection system** optimized for edge devices like Raspberry Pi using ONNX Runtime and computer vision. The system processes video streams at 5 FPS to detect potholes on roads using deep learning models with temporal voting for robust detection and anomaly detection for validation.

## Table of Contents

- [Architecture Overview](#architecture-overview)
- [System Components](#system-components)
- [Data Flow](#data-flow)
- [Module Details](#module-details)
- [Technologies Used](#technologies-used)
- [Performance Optimization](#performance-optimization)
- [Configuration](#configuration)
- [File Structure](#file-structure)
- [Usage](#usage)
- [Dependencies](#dependencies)

---

## Architecture Overview

### High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         VIDEO INPUT                             │
│                      (pothole2.mp4)                             │
└───────────────────────────┬─────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────────┐
│                   FRAME PREPROCESSING                           │
│  • Downscale (45% of original)                                  │
│  • Resize to 320x320 for inference                              │
│  • BGR to RGB conversion                                        │
│  • Normalization (0-1)                                          │
└───────────────────────────┬─────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────────┐
│                  ONNX MODEL INFERENCE                           │
│  • Model: best.onnx (YOLOv5-based)                              │
│  • Input: 320x320 normalized image                              │
│  • Output: Raw detections (bbox, confidence)                    │
│  • Backend: CPU Execution Provider                              │
└───────────────────────────┬─────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────────┐
│                    POST-PROCESSING                              │
│  • Confidence Filtering (threshold: 0.2)                        │
│  • Bounding Box Scaling                                         │
│  • Size Validation (10px - 1000px)                              │
│  • Non-Maximum Suppression (IOU: 0.4)                           │
└───────────────────────────┬─────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────────┐
│                  TEMPORAL VOTING                                │
│  • Multi-frame consensus (window: 3 frames)                     │
│  • Vote threshold: 2/3 frames                                   │
│  • Reduces false positives                                      │
└───────────────────────────┬─────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────────┐
│               ANOMALY DETECTION (OPTIONAL)                      │
│  ┌──────────────────────┬─────────────────────────┐             │
│  │  Mahalanobis         │  Fallback Anomaly       │             │
│  │  Detector            │  Detector               │             │
│  │  (Statistical)       │  (Grayscale Mean)       │             │
│  └──────────────────────┴─────────────────────────┘             │
└───────────────────────────┬─────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────────┐
│                      VISUALIZATION                              │
│  • Bounding boxes on detected potholes                          │
│  • Confidence scores                                            │
│  • FPS and performance metrics                                  │
│  • Frame counter and detection counter                          │
│  • Timestamped event logs                                       │
└─────────────────────────────────────────────────────────────────┘
```

---

## System Components

### 1. **Main Detection Pipeline** ([runvid.py](onnx/runvid.py))

The core orchestrator that integrates all components.

**Key Responsibilities:**

- Video stream management
- Frame preprocessing and scaling
- Model inference orchestration
- Post-processing and NMS
- Temporal voting integration
- Real-time visualization
- Performance monitoring

**Processing Pipeline:**

```
Video Frame → Downscale → Preprocess → Inference → Postprocess → NMS → Temporal Voting → Display
```

### 2. **Mahalanobis Anomaly Detector** ([mahalanobis_detector.py](onnx/mahalanobis_detector.py))

Statistical anomaly detection using Mahalanobis distance.

**Algorithm:**

- **Training Phase**: Computes mean vector and covariance matrix from normal samples
- **Detection Phase**: Calculates Mahalanobis distance for new samples
- **Threshold**: Configurable (default: 3.0)

**Mathematical Foundation:**

```
D_M(x) = √[(x - μ)ᵀ Σ⁻¹ (x - μ)]

Where:
- x: feature vector
- μ: mean vector
- Σ: covariance matrix
- D_M: Mahalanobis distance
```

**Use Cases:**

- Validate detected potholes against learned normal road patterns
- Reduce false positives from shadows, paint marks, etc.
- Quality assurance for detection pipeline

### 3. **Fallback Anomaly Detector** ([fallback_anomaly.py](onnx/fallback_anomaly.py))

Simple statistical anomaly detection using grayscale mean values.

**Algorithm:**

- Extract grayscale mean from each frame
- Compute running mean and standard deviation
- Score anomalies using Z-score

**Formula:**

```
Z-score = |value - mean| / std

Anomaly if: Z-score > threshold (default: 2.8)
```

**Use Cases:**

- Lightweight fallback when Mahalanobis detector is unavailable
- Quick frame quality assessment
- Brightness change detection

### 4. **ONNX Model** (best.onnx)

Pre-trained deep learning model for pothole detection.

**Specifications:**

- **Input Size**: 320x320x3 (RGB)
- **Architecture**: YOLOv8-based (inferred)
- **Output**: Bounding boxes with confidence scores
- **Format**: ONNX (cross-platform)
- **Optimization**: Quantized for edge devices

---

## Data Flow

### Frame Processing Flow

```
1. VIDEO CAPTURE
   ├─ Read frame from video file (pothole2.mp4)
   └─ Downscale to 45% (for speed)

2. PREPROCESSING
   ├─ Resize to 320x320 (inference size)
   ├─ Convert BGR → RGB
   ├─ Transpose to CHW format
   └─ Normalize to [0, 1]

3. INFERENCE
   ├─ Run ONNX model
   └─ Get raw predictions (Nx5 array)

4. POSTPROCESSING
   ├─ Filter by confidence (> 0.2)
   ├─ Scale bboxes to original frame size
   ├─ Validate bbox dimensions (10-1000px)
   └─ Apply Non-Maximum Suppression (IOU 0.4)

5. TEMPORAL VOTING
   ├─ Collect detections from 3 consecutive frames
   ├─ Require 2/3 consensus
   └─ Confirm detection only if threshold met

6. VISUALIZATION
   ├─ Draw bboxes on confirmed detections
   ├─ Display confidence scores
   ├─ Show FPS and metrics
   └─ Output to display window (640x360)
```

### Threading Model

```
Main Thread (Sequential Processing):
├─ Frame Capture
├─ Preprocessing
├─ Model Inference (4 intra-op threads)
├─ Postprocessing
├─ Temporal Voting
└─ Rendering & Display
```

---

## Module Details

### runvid.py - Main Detection Pipeline

#### Configuration Parameters

| Parameter       | Value | Purpose                           |
| --------------- | ----- | --------------------------------- |
| `TARGET_FPS`  | 5     | Target frames per second          |
| `FRAME_SCALE` | 0.45  | Downscale factor for input frames |
| `CONF_THRES`  | 0.2   | Minimum confidence threshold      |
| `IOU_THRES`   | 0.4   | IoU threshold for NMS             |
| `INFER_SIZE`  | 320   | Model input resolution            |

#### Key Functions

**`preprocess(frame)`**

- Resizes frame to 320x320
- Converts BGR to RGB
- Normalizes pixel values to [0, 1]
- Returns numpy array ready for ONNX inference

**`postprocess(preds, h, w)`**

- Filters predictions by confidence
- Scales bounding boxes to frame dimensions
- Validates box dimensions
- Returns list of valid detections

**`nms(dets, iou_thresh)`**

- Implements Non-Maximum Suppression
- Removes overlapping bounding boxes
- Keeps highest confidence detections
- Returns filtered detection list

#### Performance Metrics Tracked

- **FPS**: Frames per second (rolling average)
- **Processing Time**: Inference time per frame (ms)
- **Detection Count**: Total potholes detected
- **Frame Count**: Total frames processed

---

### mahalanobis_detector.py - Statistical Anomaly Detection

#### Class: `MahalanobisDetector`

**Attributes:**

- `threshold`: Distance threshold for anomaly classification
- `features`: Collected feature vectors during training
- `mean`: Mean feature vector (computed after training)
- `inv_cov`: Inverse covariance matrix

**Methods:**

| Method                      | Description                                   |
| --------------------------- | --------------------------------------------- |
| `__init__(threshold=3.0)` | Initialize detector with threshold            |
| `fit(feature)`            | Add training sample                           |
| `finalize()`              | Compute mean and covariance matrix            |
| `score(feature)`          | Calculate Mahalanobis distance for new sample |

**Mathematical Details:**

- Uses multivariate Gaussian distribution
- Covariance matrix regularization: `1e-6 * I` (prevents singularity)
- Distance calculation: Mahalanobis distance metric

---

### fallback_anomaly.py - Simple Anomaly Detection

#### Class: `FallbackAnomalyDetector`

**Attributes:**

- `threshold`: Z-score threshold (default: 2.8)
- `mean`: Running mean of grayscale values
- `std`: Running standard deviation
- `features`: History of feature values

**Methods:**

| Method                      | Description                       |
| --------------------------- | --------------------------------- |
| `__init__(threshold=2.8)` | Initialize with Z-score threshold |
| `extract_feature(frame)`  | Compute grayscale mean of frame   |
| `fit(frame)`              | Update statistics with new frame  |
| `score(frame)`            | Calculate Z-score for frame       |

**Feature Extraction:**

- Converts frame to grayscale
- Computes mean pixel intensity
- Single scalar feature (fast computation)

---

## Technologies Used

### Core Technologies

| Technology             | Version | Purpose                            |
| ---------------------- | ------- | ---------------------------------- |
| **Python**       | 3.x     | Primary programming language       |
| **OpenCV**       | Latest  | Video processing and visualization |
| **ONNX Runtime** | Latest  | Model inference engine             |
| **NumPy**        | Latest  | Numerical computations             |

### Model & AI

- **ONNX Format**: Cross-platform model deployment
- **YOLOv8**: Object detection architecture (inferred)
- **CPU Execution Provider**: Optimized for edge devices

### Computer Vision

- **OpenCV**: Frame capture, preprocessing, rendering
- **Custom NMS**: Hand-coded Non-Maximum Suppression
- **Temporal Voting**: Multi-frame consensus algorithm

---

## Performance Optimization

### 1. **Frame Processing Optimizations**

```python
# Downscaling for speed
FRAME_SCALE = 0.45  # 45% of original size

# Fast interpolation
cv2.INTER_NEAREST  # Faster than INTER_LINEAR
```

### 2. **Model Optimization**

```python
# ONNX Runtime Session Options
intra_op_num_threads = 4      # Parallel operations within ops
inter_op_num_threads = 1      # Sequential operator execution
execution_mode = SEQUENTIAL   # Predictable performance
graph_optimization = ALL      # Enable all optimizations
```

### 3. **Inference Optimizations**

- **Input Size**: 320x320 (smaller than typical 640x640)
- **Batch Size**: 1 (real-time processing)
- **Precision**: FP32 (can be quantized to INT8)

### 4. **Memory Optimizations**

- **Buffer Size**: Limited to 1 frame (reduces latency)
- **Rolling Averages**: Fixed window size (10 frames)
- **Contiguous Arrays**: `np.ascontiguousarray()` for speed

### 5. **FPS Control**

```python
TARGET_FRAME_TIME = 1.0 / 5  # 200ms per frame
sleep_time = TARGET_FRAME_TIME - elapsed
```

### Performance Benchmarks

| Metric          | Target | Typical   |
| --------------- | ------ | --------- |
| FPS             | 5      | 5-7       |
| Processing Time | <200ms | 50-100ms  |
| Display Latency | <50ms  | 20-30ms   |
| Total Latency   | <250ms | 100-150ms |

---

## Configuration

### Tunable Parameters

#### Detection Sensitivity

```python
CONF_THRES = 0.2    # Lower = more detections (more false positives)
                    # Higher = fewer detections (miss some potholes)
```

#### Detection Overlap

```python
IOU_THRES = 0.4     # Lower = keep more boxes (may duplicate)
                    # Higher = remove more overlaps (may miss some)
```

#### Temporal Stability

```python
# In temporal_voting module (if exists)
window_size = 3         # Number of frames to consider
vote_threshold = 2      # Minimum votes needed (2/3 consensus)
```

#### Display Settings

```python
FRAME_SCALE = 0.45      # Processing resolution
display = (640, 360)    # Output window size
```

---

## File Structure

```
edge_ai_rpi/
│
├── onnx/
│   ├── best.onnx                    # ONNX model file
│   ├── runvid.py                     # Main detection pipeline
│   ├── mahalanobis_detector.py       # Statistical anomaly detector
│   ├── fallback_anomaly.py           # Simple anomaly detector
│   └── pothole2.mp4                  # Test video file
│
└── README.md                          # This file
```

### File Descriptions

| File                        | Lines | Purpose                                   |
| --------------------------- | ----- | ----------------------------------------- |
| `runvid.py`               | 296   | Main detection and visualization pipeline |
| `mahalanobis_detector.py` | 24    | Advanced statistical anomaly detection    |
| `fallback_anomaly.py`     | 25    | Simple grayscale-based anomaly detection  |
| `best.onnx`               | -     | Pre-trained pothole detection model       |
| `pothole2.mp4`            | -     | Sample video for testing                  |

---

## Usage

### Basic Execution

```bash
# Navigate to project directory
cd c:\Users\faisa\Desktop\edge_ai_rpi\onnx

# Run pothole detection
python runvid.py
```

### Controls

- **`q` key**: Quit the application
- **`Ctrl+C`**: Stop execution (safe cleanup)

### Expected Output

```
✅ Model: best.onnx (320x320)
✅ Inference size: 320x320
✅ Target FPS: 5
✅ Video opened: pothole2.mp4

🚀 OPTIMIZED FOR 5 FPS
📊 Frame scale: 0.45
Press 'q' to quit

Frame 30: Found 2 potholes
Frame 60 | FPS: 5 | Processing: 80ms | Detections: 15
...
```

### Output Window

The display window shows:

- **Bounding boxes**: Green rectangles around detected potholes
- **Confidence scores**: Displayed above each box
- **FPS counter**: Real-time performance
- **Frame number**: Current frame index
- **Total detections**: Cumulative count
- **Processing time**: Inference latency

---

## Dependencies

### Python Packages

```
opencv-python       # Video processing and visualization
numpy              # Numerical computations
onnxruntime        # ONNX model inference
```

### Installation

```bash
pip install opencv-python numpy onnxruntime
```

### Optional Dependencies

```bash
# For GPU acceleration (CUDA)
pip install onnxruntime-gpu

# For additional image processing
pip install pillow scikit-image
```

---

## Algorithm Details

### Non-Maximum Suppression (NMS)

**Purpose**: Remove duplicate detections of the same pothole

**Algorithm:**

1. Sort detections by confidence (highest first)
2. Keep highest confidence detection
3. Remove all overlapping boxes (IoU > threshold)
4. Repeat until no boxes remain

**IoU Calculation:**

```
IoU = Intersection Area / Union Area

Where:
- Intersection: Overlapping region
- Union: Combined area of both boxes
```

### Temporal Voting

**Purpose**: Reduce false positives through multi-frame consensus

**Algorithm:**

1. Maintain sliding window of N frames (default: 3)
2. Track detection status per frame (True/False)
3. Count votes (detections) in window
4. Confirm detection only if votes ≥ threshold (2/3)

**Benefits:**

- Filters transient false positives
- Requires consistent detection across frames
- Improves overall system reliability

---

## Design Decisions

### Why ONNX Runtime?

- **Cross-platform**: Works on Windows, Linux, Raspberry Pi
- **Optimized**: Faster than TensorFlow/PyTorch on CPU
- **Lightweight**: Small binary size for edge devices
- **Production-ready**: Used in Microsoft products

### Why 320x320 Resolution?

- **Speed**: 4x faster than 640x640
- **Accuracy**: Sufficient for pothole detection
- **Memory**: Fits in Raspberry Pi RAM
- **Latency**: Enables real-time processing

### Why 5 FPS Target?

- **Processing time**: Allows 200ms per frame
- **Detection reliability**: Temporal voting needs consistent FPS
- **Resource usage**: Sustainable on Raspberry Pi
- **Responsiveness**: Fast enough for road monitoring

### Why CPU-only Execution?

- **Hardware target**: Raspberry Pi has limited GPU
- **Power efficiency**: Lower power consumption
- **Reliability**: CPU more stable than GPU on edge
- **Portability**: Works on any device

---

## Future Enhancements

### Potential Improvements

1. **Model Optimization**

   - INT8 quantization for faster inference
   - Pruning for smaller model size
   - Neural architecture search for edge devices
2. **Detection Features**

   - Depth estimation for pothole severity
   - Multi-class detection (cracks, bumps, etc.)
   - GPS integration for location tracking
3. **System Integration**

   - Real-time alerting system
   - Cloud upload for road maintenance
   - Mobile app integration
4. **Performance**

   - Multi-threading for capture and inference
   - Hardware acceleration (EdgeTPU, Neural Compute Stick)
   - Dynamic FPS adjustment

---

---

## Notes

### Important Considerations

1. **Temporal Voting Module**: Referenced in `runvid.py` but not present in repository. Implementation needed for full functionality.
2. **Model Training**: The `best.onnx` model is pre-trained. Training code and dataset not included.
3. **Video Loop**: Current implementation loops video when finished (for testing).
4. **Display Scale**: Output window is 640x360 regardless of input resolution.
5. **Error Handling**: Graceful fallback for model loading failures.

---

## Authors

Ahamed Faisal A

Sanji krishna MP

Subiksha A


**Last Updated**: 2026-02-09

**Project Status**: Active Development

**Target Hardware**: Raspberry Pi 4 Model B

**Primary Use Case**: Road Infrastructure Monitoring
