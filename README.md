# Edge AI Road Anomaly Detection — Multi-Stage Verification Pipeline (Second Approach)

## Overview

This branch contains the second-stage architecture of the Edge AI road anomaly detection system. It introduces a multi-stage verification pipeline designed specifically to reduce false positives while maintaining strong detection performance on edge devices.

The system combines a YOLO-based candidate detector, strict multi-stage bounding box filters, CNN-based appearance verification, temporal voting, and an unsupervised fallback anomaly detector.

This version represents a major improvement over the baseline pipeline in terms of detection reliability and false-positive control.

---

## High-Level Concept

The detection logic follows a staged validation strategy:

YOLO proposes candidates → rule-based filters refine → CNN verifies appearance → temporal voting confirms → fallback runs only if no valid detections exist.

The pipeline is designed so that early stages are permissive and later stages are strict.

---

## Architecture (Second Approach)

```
Dashcam Video
    ↓
ROI Cropping (road-only region)
    ↓
YOLOv5 Candidate Detection
    ↓
Multi-Stage Bounding Box Filters
    ↓
YOLO Temporal Voting
    ↓
CNN Appearance Verifier
    ↓
CNN Temporal Voting
    ↓
Fallback Anomaly Detector (only if no valid detections)
    ↓
Final Decision and Visualization
```

---

## Design Rationale (Conceptual Analogy)

The pipeline can be compared to a layered security inspection process:

- YOLO acts as a fast primary scanner that flags suspicious regions
- Bounding box filters apply strict rule checks
- CNN verifier performs detailed visual confirmation
- Temporal voting requires repeated agreement across frames
- Fallback anomaly detection runs only when primary detectors report nothing

The objective is to minimize false alarms while preserving true anomaly detections.

---

## ROI Cropping (Road Region Restriction)

Only the road region of the frame is processed.

Purpose:
- Remove sky, buildings, trees, and roadside objects
- Reduce irrelevant detections
- Lower false positives
- Reduce compute load

---

## YOLOv5 — Candidate Detector

YOLOv5 is used as a candidate generator, not a final decision maker.

Characteristics:
- Fast detection
- High recall
- Intentionally permissive
- Allows later stages to perform strict validation

---

## Multi-Stage Bounding Box Filters

All YOLO detections must pass every filter before CNN verification.

### Area (Size) Filter

Rejects:
- Extremely small boxes (noise)
- Extremely large boxes (road patches, vehicles)

Keeps:
- Medium-sized candidate regions

---

### Position Filter (Y-axis Constraint)

Rejects:
- Detections in upper frame regions

Keeps:
- Detections within the road zone

Purpose:
- Potholes and cracks occur on road surface only

---

### Minimum Width and Height Filter

Rejects:
- Very thin or line-like detections

Purpose:
- Prevent lane markings and hairline artifacts from passing

---

### Aspect Ratio Filter

Rejects:
- Long thin shapes
- Extremely tall shapes

Keeps:
- Blob-like regions typical of potholes

---

### Border Rejection Filter

Rejects:
- Boxes touching frame edges

Purpose:
- Avoid partially visible objects
- Improve classifier reliability

---

### Inter-Frame Stability Filter

Rejects:
- Sudden large size changes between frames

Purpose:
- Remove camera shake artifacts
- Remove unstable detector glitches

---

## YOLO Temporal Voting

YOLO detections must appear across multiple consecutive frames.

Configuration:
- Sliding frame window
- Detection confirmed only with multi-frame agreement

Purpose:
- Remove single-frame false positives
- Improve temporal consistency

---

## CNN Appearance Verifier

A CNN classifier validates the visual texture inside each filtered bounding box.

Role:
- Confirms whether the region visually resembles a pothole or anomaly
- Adds a second-stage semantic check beyond bounding box geometry

Safeguards:
- High confidence threshold
- CNN temporal voting across frames

---

## CNN Temporal Voting

CNN decisions are also validated across frames.

Purpose:
- Prevent single-frame CNN misclassifications
- Improve final decision stability

---

## Fallback Anomaly Detector (Unsupervised)

Fallback detection runs only when no valid YOLO + CNN detections remain.

Capabilities:
- Detect unknown anomaly types
- Use texture and intensity deviation features
- Apply temporal voting
- Use cooldown intervals between alerts

Purpose:
- Provide coverage for unseen anomaly patterns
- Avoid over-triggering

---

## Comparison with First Approach

## Comparison: First Approach vs Second Approach

| Aspect | First Approach | Second Approach |
|---------|----------------|----------------|
| YOLO role | Acts as final decision maker | Acts as candidate generator only |
| Filtering strategy | Basic rule filters | Multi-stage strict filters |
| Appearance verification | Not included | CNN-based verifier added |
| Temporal validation | Limited temporal check | YOLO + CNN temporal voting |
| Fallback usage | Triggered more often | Triggered rarely and under strict conditions |
| False positives | Relatively higher | Strongly reduced |
| Edge stability | Moderate | Improved and more stable |


---

## Key Improvements in This Branch

- Multi-stage bounding box validation
- CNN-based appearance verification
- Dual temporal voting (YOLO and CNN)
- Strict fallback activation rules
- Strong false-positive reduction design

---

## Notes

This branch focuses on pipeline robustness and false-positive control through staged verification. It is more computationally intensive than the baseline approach but provides significantly more stable detection behavior.

For the optimized ONNX edge deployment pipeline, refer to the main branch.
