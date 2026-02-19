# Edge AI Pothole Detection — Live Pi Camera Pipeline (`rpicam-vid`)

## Overview

This folder contains **`run_with_pi_cam.py`** — the live-camera variant of the
Edge AI Road Anomaly Detection system.

Instead of reading a pre-recorded video file, the detector streams frames directly
from a **Raspberry Pi Camera Module v2** using the built-in **`rpicam-vid`** tool
that ships with Raspberry Pi OS. No external Python camera library is needed —
`rpicam-vid` is launched as a subprocess and raw **YUV420 (I420)** frames are piped
directly into the Python process via `stdout`.

The detection logic extends the parent project's ONNX pipeline with:

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

## Hardware Requirements

| Component | Details                                      |
| --------- | -------------------------------------------- |
| SBC       | Raspberry Pi 4 Model B (4 GB RAM+)           |
| OS        | Raspberry Pi OS 64-bit (Bookworm / Bullseye) |
| Camera    | Pi Camera Module **v2** (IMX219)             |
| Interface | CSI-2 ribbon cable (included with camera)    |
| Storage   | 16 GB+ microSD (Class 10 / A1)               |
| Power     | Official 5 V / 3 A USB-C PSU                 |

> **Important:** `rpicam-vid` is a system-level tool bundled with Raspberry Pi OS.
> It talks to the camera hardware directly via **libcamera** — no Python camera
> library (`picamera2`, `picamera`) is required or used.

---

## Software Dependencies

```bash
# Python packages only
pip install opencv-python numpy onnxruntime
```

`rpicam-vid` is already installed on Raspberry Pi OS — no additional install needed.

```bash
# Verify rpicam-vid is available
rpicam-vid --version
```

> The script will exit immediately with a clear error if `rpicam-vid` fails to start.

---

## File Structure

```
with_pi_cam/
├── run_with_pi_cam.py   ← Main live-camera detection script
└── README.md            ← This document

# Parent project (model + shared modules):
../Code_in_PI4/
├── best.onnx            ← YOLOv5 pothole detector (ONNX, 320×320)
├── runvid.py            ← Original video-file pipeline
├── temporal_voting.py   ← MultiFrameVoter (imported by this script)
├── mahalanobis_detector.py
└── fallback_anomaly.py

# Auto-created at runtime:
../logs/
├── pothole_<SESSION>.log         ← Text log (per session)
├── detections_<SESSION>.csv      ← CSV of all confirmed detections
├── summary_<SESSION>.json        ← JSON session summary
├── potholes.db                   ← SQLite database (all sessions)
└── detection_images/
    └── <SESSION>_frame<N>.jpg    ← JPEG snapshots of confirmed frames
```

---

## How to Run

```bash
# Navigate to the folder
cd edge_ai_rpi/with_pi_cam

# Run the Pi Camera pipeline
python run_with_pi_cam.py
```

### Controls

| Key      | Action                                           |
| -------- | ------------------------------------------------ |
| `q`      | Quit (closes window, writes logs, prints stats)  |
| `Ctrl+C` | Safe stop (same cleanup as `q`)                  |

---

## Architecture — Full Detailed Pipeline

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
│  • FRAME_BYTES = 640 × 480 × 3 // 2  (YUV420 I420 layout)       │
│  • Uses readinto loop into a fixed bytearray buffer              │
│  • Guarantees exactly one full frame per call                    │
│  • Returns None if pipe closes (triggers clean shutdown)         │
│  • Converts: YUV420 → BGR via cv2.COLOR_YUV2BGR_I420            │
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
│  4. uint8 → float32,  divide by 255  → [0.0, 1.0]               │
│  5. Add batch dim: shape becomes (1, 3, 320, 320)                │
└───────────────────────────┬──────────────────────────────────────┘
                            │
                            ▼
┌──────────────────────────────────────────────────────────────────┐
│                   YOLOv5 ONNX INFERENCE                          │
│  • Model file  : best.onnx  (YOLOv5, exported to ONNX)          │
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
│    Convert centre-format → corners (x1, y1, x2, y2)             │
│    Clamp to frame boundary                                       │
│                                                                  │
│  Filter 3 — Size:                                                │
│    Discard if box width or height < 10 px or > 1000 px           │
│                                                                  │
│  Filter 4 — Area ratio (new):                                    │
│    Discard if box area > 30% of total frame area                 │
│    (prevents huge false boxes covering entire scene)             │
│                                                                  │
│  Filter 5 — Aspect ratio (new):                                  │
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
│  • Remove overlapping boxes where IoU > IOU_THRES (0.4)         │
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
│    VISUALISATION (draw)  │  │    DETECTION LOGGING (log_detection)│
│                          │  │                                    │
│ • Green boxes on frame   │  │ Triggered only when BOTH:           │
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
│  • TARGET_FRAME_TIME = 1.0 / 5 = 200 ms                         │
│  • Sleep remainder after each loop iteration                     │
│  • cv2.waitKey handles both sleep and key press check            │
│  • Console log every 2 seconds: frame/fps/proc/detections        │
└──────────────────────────────────────────────────────────────────┘
                            │
                 [q pressed / Ctrl+C / pipe closed]
                            ▼
┌──────────────────────────────────────────────────────────────────┐
│                    CLEAN SHUTDOWN (finally)                      │
│  1. cam_proc.terminate() + cam_proc.wait()  — kill rpicam-vid   │
│  2. cv2.destroyAllWindows()                                      │
│  3. Flush SQLite buffer, close CSV file                          │
│  4. Write summary_<SESSION>.json                                 │
│  5. Print final statistics to console                            │
└──────────────────────────────────────────────────────────────────┘
```

---

## Key Configuration Parameters

| Parameter            | Value  | Description                                     |
| -------------------- | ------ | ----------------------------------------------- |
| `TARGET_FPS`         | 5      | Frames per second cap (RPi CPU optimised)        |
| `FRAME_SCALE`        | 0.45   | Pre-inference downscale ratio                   |
| `CONF_THRES`         | **0.40** | Min confidence — raised from 0.20 to cut FPs  |
| `IOU_THRES`          | 0.4    | NMS IoU overlap threshold                       |
| `INFER_SIZE`         | 320    | ONNX model input size (px)                      |
| `CAM_WIDTH`          | 640    | Camera capture width (px)                       |
| `CAM_HEIGHT`         | 480    | Camera capture height (px)                      |
| `MAX_BOX_AREA_RATIO` | 0.30   | Reject boxes covering > 30% of frame *(new)*    |
| `MIN_ASPECT`         | 0.3    | Reject boxes narrower than this ratio *(new)*   |
| `MAX_ASPECT`         | 3.0    | Reject boxes wider than this ratio *(new)*      |
| `window_size`        | 3      | Temporal voter sliding window length            |
| `vote_threshold`     | 2      | Minimum votes to confirm a detection            |
| `intra_op_num_threads` | 4    | ONNX CPU threads (matches RPi4 cores)           |
| `_DB_FLUSH_EVERY`    | 10     | SQLite write-buffer size (rows before commit)   |

---

## rpicam-vid vs picamera2

| Aspect        | **rpicam-vid** (this script)        | `picamera2` (Python lib)         |
| ------------- | ----------------------------------- | -------------------------------- |
| Type          | System CLI tool (subprocess)        | Python library                   |
| Install       | Pre-installed on Raspberry Pi OS    | `pip install picamera2`          |
| Frame format  | YUV420 (I420) via stdout pipe       | BGR888 via Python API            |
| Latency       | Very low (direct kernel pipe)       | Low (DMA-backed)                 |
| Python import | Not needed (subprocess.Popen)       | `from picamera2 import Picamera2`|
| Availability  | Raspberry Pi OS only                | Raspberry Pi OS only             |

> This script uses **`rpicam-vid`** because it is built into Raspberry Pi OS 64-bit
> and requires no additional Python packages.

---

## YOLOv5 ONNX Model Details

| Property            | Value                                        |
| ------------------- | -------------------------------------------- |
| Architecture        | **YOLOv5**                                   |
| Model file          | `best.onnx`                                  |
| Input tensor        | `(1, 3, 320, 320)` float32 RGB               |
| Output tensor       | `(1, N, ≥5)` — (x_c, y_c, w, h, conf, …)   |
| Detector class      | Pothole / road anomaly                       |
| Export format       | ONNX opset 12+                               |
| Inference runtime   | ONNX Runtime CPU                             |
| Approximate RPi FPS | 5–7 FPS (with FRAME_SCALE=0.45)              |

---

## Component Responsibilities

| Section                  | Responsibility                                                     |
| ------------------------ | ------------------------------------------------------------------ |
| **rpicam-vid launch**    | Spawn subprocess, pipe YUV420 stdout, verify it started           |
| **`read_frame()`**       | Blocking readinto loop → YUV2BGR → return BGR frame               |
| **ONNX session setup**   | Load best.onnx, 4-thread CPU, full graph optimisation             |
| **`preprocess()`**       | Resize → RGB flip → CHW transpose → normalise → batch dim         |
| **`postprocess()`**      | Conf + size + area-ratio + aspect-ratio filters, box scaling      |
| **`nms()`**              | Greedy IoU-based duplicate suppression                            |
| **`MultiFrameVoter`**    | 3-frame sliding-window majority vote (from `temporal_voting.py`)  |
| **`log_detection()`**    | Write to CSV, buffer SQLite rows, save JPEG snapshot              |
| **`log_session_end()`**  | Flush DB, close CSV, write JSON summary                           |
| **Main loop**            | Capture → downscale → infer → filter → vote → draw → HUD → pace  |
| **`finally` block**      | Kill rpicam-vid, close windows, flush all logs, print stats       |

---

## Differences vs `runvid.py` (Video File Pipeline)

| Feature              | `runvid.py`                | `run_with_pi_cam.py`              |
| -------------------- | -------------------------- | --------------------------------- |
| Input source         | Pre-recorded `.mp4`        | Live Pi Camera via `rpicam-vid`   |
| Capture API          | `cv2.VideoCapture(path)`   | `subprocess.Popen` + pipe read    |
| Frame format         | Video codec (BGR)          | Raw YUV420 → BGR conversion       |
| Loop restart         | Rewinds video on EOF       | Continuous live stream            |
| MultiFrameVoter      | Imported                   | Imported (same module)            |
| Conf threshold       | 0.20                       | **0.40** (stricter)               |
| Extra box filters    | None                       | Area ratio + aspect ratio checks  |
| Logging              | Console only               | `.log` / `.csv` / `.db` / `.json` / `.jpg` |
| Camera fallback      | None                       | None (exits if rpicam-vid fails)  |

---

## Logging — Output Files

Every run creates a unique `SESSION_ID = YYYYMMDD_HHMMSS` and writes:

| File | Format | Contents |
|------|--------|----------|
| `pothole_<SESSION>.log` | Text | All INFO logs — session start/end + each detection event |
| `detections_<SESSION>.csv` | CSV | One row per confirmed detection: timestamp, frame, boxes, FPS, ms |
| `potholes.db` | SQLite (WAL) | `sessions` + `detections` tables, accumulates across all runs |
| `summary_<SESSION>.json` | JSON | Session totals: frames, detections, duration, avg FPS |
| `detection_images/<SESSION>_frame<N>.jpg` | JPEG 85% | Frame snapshot at the moment of each confirmed detection |

---

## Performance Optimisations

| Technique              | Detail                                                         |
| ---------------------- | -------------------------------------------------------------- |
| Frame downscale 0.45   | Reduces pixels ~80% before inference                          |
| INTER_NEAREST          | Fastest OpenCV resize interpolation                            |
| Input size 320         | ~4× fewer FLOPs vs default YOLOv5 640                         |
| 4 CPU threads          | Saturates all RPi4 cores via `intra_op_num_threads`           |
| Sequential exec mode   | Avoids thread-switching overhead                               |
| Full graph opt         | ONNX Runtime fuses and optimises compute graph                 |
| FPS cap (5)            | Prevents CPU spinning; paced via cv2.waitKey                   |
| Pipe buffer × 4        | Prevents rpicam-vid pipe stalls                                |
| SQLite WAL + buffer    | Batches DB writes every 10 rows — avoids per-frame I/O        |
| Rolling 10-frame avg   | FPS/proc-time stats without unbounded memory                   |

---

## Output — Console

```
✅ Model: best.onnx (320x320)
✅ Inference size : 320x320
✅ Target FPS     : 5
✅ Conf threshold : 0.4  ← raised to reduce false positives
🎥 Starting rpicam-vid …
✅ rpicam-vid started | 640x480 @ 5fps

🚀 Running | FPS=5 | scale=0.45 | conf>=0.4
Press Ctrl+C or 'q' to quit

Frame   110 | FPS: 5 | Proc: 138ms | Total detections: 2
Frame   120: 1 pothole(s) confirmed
Frame   160 | FPS: 5 | Proc: 135ms | Total detections: 3
...
✅ Stopped by user

📊 Final Statistics:
   Total frames    : 320
   Total time      : 64.0s
   Average FPS     : 5.0
   Total detections: 6
   Avg processing  : 137ms
   Model           : best.onnx (320x320)
   Logs saved to   : ../logs
✅ Done!
```

---

## Output — Display Window

Window title: **"Pothole Detection"**

- **Green rectangle** around each detection (drawn every frame with detections)
- **"POTHOLE X.XX"** confidence label above each box
- **Red "POTHOLE DETECTED!"** banner at top-left when confirmed
- **HUD** at bottom of frame:
  - FPS (cyan)  ·  Frame # (yellow)  ·  Total detections (magenta)  ·  Proc time ms (orange)
- Resized to **640×360** for stable output size

---

## Troubleshooting

| Problem                       | Likely Cause               | Fix                                                    |
| ----------------------------- | -------------------------- | ------------------------------------------------------ |
| `rpicam-vid failed` on start  | Camera not connected/enabled | Check CSI ribbon, enable camera in `raspi-config`    |
| `Error loading model`         | `best.onnx` not found      | Ensure `best.onnx` is in the same folder as the script |
| Black / green screen          | YUV format mismatch        | Confirm `--codec yuv420` is supported by your RPi OS   |
| Very low FPS (< 2)            | CPU overloaded / no cooling | Add heatsink/fan; lower `CAM_WIDTH`/`CAM_HEIGHT`     |
| `rpicam-vid: command not found` | Old Raspberry Pi OS version | `sudo apt update && sudo apt upgrade`               |
| `temporal_voting` not found   | Script run from wrong dir  | Make sure `temporal_voting.py` is on the Python path   |
| No display window             | Headless SSH session       | Use X11 forwarding (`ssh -X`) or run with VNC          |

---

## Quick Start (Raspberry Pi)

```bash
# 1. Enable camera (if not already)
sudo raspi-config
# → Interface Options → Camera → Enable

# 2. Verify rpicam-vid works
rpicam-vid --width 640 --height 480 --timeout 3000 --nopreview

# 3. Install Python dependencies
pip install opencv-python numpy onnxruntime

# 4. Navigate to folder (best.onnx must be here or update MODEL_PATH)
cd edge_ai_rpi/with_pi_cam

# 5. Run
python run_with_pi_cam.py
```

---

## Authors

Ahamed Faisal A
Sanji Krishna M P
Subiksha A

**Target Hardware:** Raspberry Pi 4
**Camera:** Raspberry Pi Camera Module v2 (IMX219, CSI)
**Camera Tool:** `rpicam-vid` (built-in to Raspberry Pi OS)
**Project Status:** Active Development
