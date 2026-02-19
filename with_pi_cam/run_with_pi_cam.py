import cv2
import numpy as np
import time
import onnxruntime as ort
import os
import csv
import json
import logging
import sqlite3
import subprocess
from datetime import datetime
from pathlib import Path
from temporal_voting import MultiFrameVoter

# =============================
# CONFIG
# =============================
TARGET_FPS   = 5
FRAME_SCALE  = 0.45
CONF_THRES   = 0.40        # FIX: raised from 0.20 → eliminates weak false positives
IOU_THRES    = 0.4
INFER_SIZE   = 320

MAX_BOX_AREA_RATIO = 0.30  # FIX: ignore boxes covering >30% of frame (e.g. laptop)
MIN_ASPECT   = 0.3         # FIX: ignore very tall/thin boxes
MAX_ASPECT   = 3.0         # FIX: ignore very wide boxes

CAM_WIDTH    = 640
CAM_HEIGHT   = 480
FRAME_BYTES  = CAM_WIDTH * CAM_HEIGHT * 3 // 2   # YUV420 (I420)

# =============================
# LOGGING
# =============================
LOG_DIR = Path(__file__).resolve().parent.parent / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)

IMG_DIR = LOG_DIR / "detection_images"
IMG_DIR.mkdir(parents=True, exist_ok=True)

SESSION_ID = datetime.now().strftime("%Y%m%d_%H%M%S")

log_file = LOG_DIR / f"pothole_{SESSION_ID}.log"
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.FileHandler(log_file), logging.StreamHandler()]
)
logger = logging.getLogger("pothole")

csv_file = LOG_DIR / f"detections_{SESSION_ID}.csv"
_csv_fh  = open(csv_file, "w", newline="", buffering=1)
csv_writer = csv.writer(_csv_fh)
csv_writer.writerow(["timestamp","frame_id","num_boxes","max_conf",
                     "boxes_x1y1x2y2_conf","fps","process_ms"])

db_file = LOG_DIR / "potholes.db"
_db = sqlite3.connect(str(db_file), check_same_thread=False)
_db.execute("PRAGMA journal_mode=WAL")
_db.execute("""CREATE TABLE IF NOT EXISTS sessions (
    session_id TEXT PRIMARY KEY, source TEXT, model_name TEXT, started_at TEXT)""")
_db.execute("""CREATE TABLE IF NOT EXISTS detections (
    id INTEGER PRIMARY KEY AUTOINCREMENT, session_id TEXT,
    timestamp TEXT, frame_id INTEGER, num_boxes INTEGER,
    max_conf REAL, avg_fps REAL, process_ms REAL, boxes_json TEXT)""")
_db.commit()

def _log_session_start(source, model_name):
    _db.execute("INSERT OR IGNORE INTO sessions VALUES (?,?,?,?)",
                (SESSION_ID, source, model_name, datetime.now().isoformat()))
    _db.commit()
    logger.info(f"Session started | id={SESSION_ID} | source={source} | model={model_name}")

_DB_FLUSH_EVERY = 10
_db_buffer = []

def log_detection(frame_id, detections, avg_fps, process_ms, frame=None):
    ts        = datetime.now().isoformat(timespec="milliseconds")
    num_boxes = len(detections)
    max_conf  = max(d[4] for d in detections) if detections else 0.0
    boxes     = [[int(x1),int(y1),int(x2),int(y2),round(float(c),3)]
                 for x1,y1,x2,y2,c in detections]
    boxes_json = json.dumps(boxes)
    csv_writer.writerow([ts, frame_id, num_boxes, round(max_conf,3),
                         boxes_json, round(avg_fps,1), round(process_ms,1)])
    logger.info(f"DETECTION | frame={frame_id} | boxes={num_boxes} | "
                f"max_conf={max_conf:.2f} | fps={avg_fps:.1f} | proc={process_ms:.0f}ms")
    _db_buffer.append((SESSION_ID, ts, frame_id, num_boxes,
                       round(max_conf,3), round(avg_fps,1), round(process_ms,1), boxes_json))
    if len(_db_buffer) >= _DB_FLUSH_EVERY:
        _flush_db()
    if frame is not None:
        img_path = IMG_DIR / f"{SESSION_ID}_frame{frame_id:06d}.jpg"
        cv2.imwrite(str(img_path), frame, [cv2.IMWRITE_JPEG_QUALITY, 85])

def _flush_db():
    if not _db_buffer: return
    _db.executemany("""INSERT INTO detections
        (session_id,timestamp,frame_id,num_boxes,max_conf,avg_fps,process_ms,boxes_json)
        VALUES (?,?,?,?,?,?,?,?)""", _db_buffer)
    _db.commit()
    _db_buffer.clear()

def log_session_end(frame_id, total_detections, total_time, actual_fps):
    _flush_db()
    _csv_fh.close()
    summary = {"session_id": SESSION_ID, "total_frames": frame_id,
               "total_detections": total_detections,
               "duration_s": round(total_time,1), "avg_fps": round(actual_fps,1)}
    (LOG_DIR / f"summary_{SESSION_ID}.json").write_text(json.dumps(summary, indent=2))
    logger.info(f"Session ended | frames={frame_id} | detections={total_detections} "
                f"| duration={total_time:.1f}s | fps={actual_fps:.1f}")
    _db.close()

# =============================
# MODEL LOADING
# =============================
session_options = ort.SessionOptions()
session_options.intra_op_num_threads     = 4
session_options.inter_op_num_threads     = 1
session_options.execution_mode           = ort.ExecutionMode.ORT_SEQUENTIAL
session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

try:
    ort_session = ort.InferenceSession(
        "best.onnx", sess_options=session_options,
        providers=["CPUExecutionProvider"])
    model_name = "best.onnx"
    print("✅ Model: best.onnx (320x320)")
except Exception as e:
    print(f"❌ Error loading model: {e}"); exit(1)

input_name = ort_session.get_inputs()[0].name
print(f"✅ Inference size : {INFER_SIZE}x{INFER_SIZE}")
print(f"✅ Target FPS     : {TARGET_FPS}")
print(f"✅ Conf threshold : {CONF_THRES}  ← raised to reduce false positives")

# FIX: simple voter, no sticky latch
voter = MultiFrameVoter(window_size=3, vote_threshold=2)

# =============================
# DETECTION FUNCTIONS
# =============================
def preprocess(frame):
    img = cv2.resize(frame, (INFER_SIZE, INFER_SIZE), interpolation=cv2.INTER_NEAREST)
    img = img[:, :, ::-1].transpose(2, 0, 1)
    img = np.ascontiguousarray(img, dtype=np.float32) / 255.0
    return np.expand_dims(img, axis=0)

def postprocess(preds, h, w):
    detections = []
    if len(preds) == 0:
        return detections
    frame_area = h * w
    for det in preds:
        if len(det) < 5:
            continue
        x, y, bw, bh, conf = det[:5]
        if conf < CONF_THRES:
            continue
        sx = w / INFER_SIZE
        sy = h / INFER_SIZE
        x1 = max(0, int((x - bw/2) * sx))
        y1 = max(0, int((y - bh/2) * sy))
        x2 = min(w, int((x + bw/2) * sx))
        y2 = min(h, int((y + bh/2) * sy))
        if x1 < x2 and y1 < y2:
            bw_px = x2 - x1
            bh_px = y2 - y1
            if not (10 < bw_px < 1000 and 10 < bh_px < 1000):
                continue
            # FIX: skip boxes that are too large (e.g. entire laptop/desk)
            box_area = bw_px * bh_px
            if box_area > MAX_BOX_AREA_RATIO * frame_area:
                continue
            # FIX: skip boxes with unrealistic aspect ratio for potholes
            aspect = bw_px / (bh_px + 1e-5)
            if aspect < MIN_ASPECT or aspect > MAX_ASPECT:
                continue
            detections.append((x1, y1, x2, y2, float(conf)))
    return detections

def nms(dets, iou_thresh=0.4):
    if not dets:
        return []
    dets = sorted(dets, key=lambda x: x[4], reverse=True)
    keep = []
    while dets:
        keep.append(dets[0])
        if len(dets) == 1:
            break
        dets = dets[1:]
        rem  = []
        x1k, y1k, x2k, y2k = keep[-1][:4]
        ak = (x2k - x1k) * (y2k - y1k)
        for d in dets:
            x1, y1, x2, y2 = d[:4]
            ix1, iy1 = max(x1k, x1), max(y1k, y1)
            ix2, iy2 = min(x2k, x2), min(y2k, y2)
            if ix2 > ix1 and iy2 > iy1:
                inter = (ix2 - ix1) * (iy2 - iy1)
                a     = (x2  - x1)  * (y2  - y1)
                iou   = inter / (ak + a - inter + 1e-5)
                if iou < iou_thresh:
                    rem.append(d)
            else:
                rem.append(d)
        dets = rem
    return keep

# =============================
# RPICAM-VID SETUP
# =============================
rpicam_cmd = [
    "rpicam-vid",
    "--width",     str(CAM_WIDTH),
    "--height",    str(CAM_HEIGHT),
    "--framerate", str(TARGET_FPS),
    "--codec",     "yuv420",
    "--nopreview",
    "--timeout",   "0",
    "--output",    "-"
]

print("🎥 Starting rpicam-vid …")
cam_proc = subprocess.Popen(
    rpicam_cmd,
    stdout=subprocess.PIPE,
    stderr=subprocess.PIPE,
    bufsize=FRAME_BYTES * 4
)

time.sleep(2.0)
if cam_proc.poll() is not None:
    err = cam_proc.stderr.read().decode(errors="replace")
    print(f"❌ rpicam-vid failed:\n{err}")
    exit(1)

print(f"✅ rpicam-vid started | {CAM_WIDTH}x{CAM_HEIGHT} @ {TARGET_FPS}fps")

_yuv_buf = bytearray(FRAME_BYTES)

def read_frame():
    """Guaranteed full-frame read using readinto loop."""
    total = 0
    view  = memoryview(_yuv_buf)
    while total < FRAME_BYTES:
        n = cam_proc.stdout.readinto(view[total:])
        if n == 0:
            return None
        total += n
    yuv = np.frombuffer(_yuv_buf, dtype=np.uint8).reshape((CAM_HEIGHT * 3 // 2, CAM_WIDTH))
    return cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR_I420)

_log_session_start("rpicam-vid", model_name)

# =============================
# MAIN LOOP
# =============================
TARGET_FRAME_TIME = 1.0 / TARGET_FPS
frame_id          = 0
total_detections  = 0
frame_times       = []
processing_times  = []

print(f"\n🚀 Running | FPS={TARGET_FPS} | scale={FRAME_SCALE} | conf>={CONF_THRES}")
print("Press Ctrl+C or 'q' to quit\n")

start_time      = time.time()
last_print_time = start_time

try:
    while True:
        loop_start = time.time()

        # ── 1. CAPTURE ───────────────────────────────────────────────────
        frame = read_frame()
        if frame is None:
            print("❌ Camera pipe closed — exiting")
            break

        # ── 2. DOWNSCALE ─────────────────────────────────────────────────
        h, w  = frame.shape[:2]
        frame = cv2.resize(frame,
                           (int(w * FRAME_SCALE), int(h * FRAME_SCALE)),
                           interpolation=cv2.INTER_NEAREST)
        frame_id += 1
        h, w = frame.shape[:2]

        # ── 3. INFERENCE ─────────────────────────────────────────────────
        process_start = time.time()
        inp = preprocess(frame)
        try:
            output = ort_session.run(None, {input_name: inp})
            preds  = output[0] if output else np.array([])
            if len(preds.shape) == 3:
                preds = preds[0]
        except Exception as e:
            print(f"Inference error: {e}")
            preds = np.array([])

        detections = postprocess(preds, h, w)
        detections = nms(detections, IOU_THRES)

        processing_time = (time.time() - process_start) * 1000
        processing_times.append(processing_time)
        if len(processing_times) > 10:
            processing_times.pop(0)

        # ── 4. TEMPORAL VOTER (simple, no sticky latch) ──────────────────
        # FIX: use clean voter result only — no _detection_active latch
        confirmed = voter.update(len(detections) > 0)

        # ── 5. FPS ───────────────────────────────────────────────────────
        frame_time = time.time() - loop_start
        frame_times.append(frame_time)
        if len(frame_times) > 10:
            frame_times.pop(0)
        avg_fps     = int(1.0 / (np.mean(frame_times) + 1e-6))
        avg_process = np.mean(processing_times)

        # ── 6. DRAW BOXES — every frame that has detections ───────────────
        if detections:
            for x1, y1, x2, y2, conf in detections:
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                label  = f"POTHOLE {conf:.2f}"
                label_y = y1 - 8 if y1 > 20 else y1 + 18
                cv2.putText(frame, label, (x1, label_y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)
            cv2.putText(frame, "POTHOLE DETECTED!",
                        (20, 40), cv2.FONT_HERSHEY_SIMPLEX,
                        1.0, (0, 0, 255), 3)

        # ── 7. LOGGING (voter-gated) + IMAGE SAVE ────────────────────────
        if confirmed and detections:
            total_detections += len(detections)
            log_detection(frame_id, detections, avg_fps, avg_process, frame)
            if frame_id % 10 == 0:
                print(f"Frame {frame_id}: {len(detections)} pothole(s) confirmed")

        # ── 8. HUD ───────────────────────────────────────────────────────
        cv2.putText(frame, f"FPS: {avg_fps}",
                    (20, h - 70), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,255), 2)
        cv2.putText(frame, f"Frame: {frame_id}",
                    (20, h - 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,0), 2)
        cv2.putText(frame, f"Total: {total_detections}",
                    (20, h - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,0,255), 2)
        cv2.putText(frame, f"Proc: {avg_process:.0f}ms",
                    (20, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,165,0), 1)

        # ── 9. DISPLAY ───────────────────────────────────────────────────
        display = cv2.resize(frame, (640, 360), interpolation=cv2.INTER_NEAREST)
        cv2.imshow("Pothole Detection", display)

        # ── 10. FPS THROTTLE ─────────────────────────────────────────────
        elapsed = time.time() - loop_start
        wait_ms = max(1, int((TARGET_FRAME_TIME - elapsed) * 1000))
        if cv2.waitKey(wait_ms) & 0xFF == ord('q'):
            break

        if time.time() - last_print_time > 2.0:
            print(f"Frame {frame_id} | FPS: {avg_fps} | "
                  f"Proc: {avg_process:.0f}ms | Total detections: {total_detections}")
            last_print_time = time.time()

except KeyboardInterrupt:
    print("\n✅ Stopped by user")

finally:
    cam_proc.terminate()
    cam_proc.wait()
    cv2.destroyAllWindows()

    total_time = time.time() - start_time
    actual_fps = frame_id / total_time if total_time > 0 else 0
    log_session_end(frame_id, total_detections, total_time, actual_fps)

    avg_proc_final = np.mean(processing_times) if processing_times else 0
    print(f"\n📊 Final Statistics:")
    print(f"   Total frames    : {frame_id}")
    print(f"   Total time      : {total_time:.1f}s")
    print(f"   Average FPS     : {actual_fps:.1f}")
    print(f"   Total detections: {total_detections}")
    print(f"   Avg processing  : {avg_proc_final:.0f}ms")
    print(f"   Model           : {model_name} (320x320)")
    print(f"   Logs saved to   : {LOG_DIR}")
    print("✅ Done!")