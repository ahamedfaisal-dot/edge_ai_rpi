import cv2
import numpy as np
import time
import onnxruntime as ort
import os
from temporal_voting import MultiFrameVoter

TARGET_FPS = 5
FRAME_SCALE = 0.45  
CONF_THRES = 0.2 
IOU_THRES = 0.4  
INFER_SIZE = 320

model_name = None

session_options = ort.SessionOptions()
session_options.intra_op_num_threads = 4
session_options.inter_op_num_threads = 1
session_options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

try:
    session = ort.InferenceSession(
        "best.onnx",
        sess_options=session_options, 
        providers=["CPUExecutionProvider"]
    )
    model_name = "best.onnx"
    print(" Model: best.onnx (320x320)")
except Exception as e:
    print(f" Error loading model: {e}")
    exit(1)

input_name = session.get_inputs()[0].name
print(f"✅ Inference size: {INFER_SIZE}x{INFER_SIZE}")
print(f"✅ Target FPS: {TARGET_FPS}")

voter = MultiFrameVoter(window_size=3, vote_threshold=2)

def preprocess(frame):
   
    img = cv2.resize(frame, (INFER_SIZE, INFER_SIZE), interpolation=cv2.INTER_NEAREST)
    img = img[:, :, ::-1].transpose(2, 0, 1)
    img = np.ascontiguousarray(img, dtype=np.float32) / 255.0
    return np.expand_dims(img, axis=0)


def postprocess(preds, h, w):
    
    detections = []
    
    if len(preds) == 0:
        return detections
    
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
            if 10 < bw_px < 1000 and 10 < bh_px < 1000:
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
        rem = []
        x1k, y1k, x2k, y2k = keep[-1][:4]
        ak = (x2k - x1k) * (y2k - y1k)
        
        for d in dets:
            x1, y1, x2, y2 = d[:4]
            ix1, iy1 = max(x1k, x1), max(y1k, y1)
            ix2, iy2 = min(x2k, x2), min(y2k, y2)
            
            if ix2 > ix1 and iy2 > iy1:
                inter = (ix2 - ix1) * (iy2 - iy1)
                a = (x2 - x1) * (y2 - y1)
                iou = inter / (ak + a - inter + 1e-5)
                if iou < iou_thresh:
                    rem.append(d)
            else:
                rem.append(d)
        
        dets = rem
    
    return keep



video_path = "pothole2.mp4"
if not os.path.exists(video_path):
    video_path = os.path.join("..", "pothole2.mp4")

if not os.path.exists(video_path):
    print(f" Video not found: {video_path}")
    exit(1)

cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    print(f" Failed to open video: {video_path}")
    exit(1)

cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
print(f" Video opened: {video_path}")


TARGET_FRAME_TIME = 1.0 / TARGET_FPS

frame_id = 0
total_detections = 0
frame_times = []
processing_times = []

print(f"\n OPTIMIZED FOR 5 FPS")
print(f" Frame scale: {FRAME_SCALE}")
print("Press 'q' to quit\n")

start_time = time.time()
last_print_time = start_time

try:
    while cap.isOpened():
        loop_start = time.time()
        
        ret, frame = cap.read()
        if not ret:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            continue
        
        
        h, w = frame.shape[:2]
        frame = cv2.resize(
            frame, 
            (int(w * FRAME_SCALE), int(h * FRAME_SCALE)),
            interpolation=cv2.INTER_NEAREST
        )
        
        frame_id += 1
        h, w = frame.shape[:2]
        
        
        process_start = time.time()
        
        inp = preprocess(frame)
        
        try:
            output = session.run(None, {input_name: inp})
            if output and len(output) > 0:
                preds = output[0]
                if len(preds.shape) == 3:
                    preds = preds[0]
            else:
                preds = np.array([])
        except Exception as e:
            print(f"Inference error: {e}")
            preds = np.array([])
        
       
        detections = postprocess(preds, h, w)
        detections = nms(detections, IOU_THRES)
        
        processing_time = (time.time() - process_start) * 1000
        processing_times.append(processing_time)
        if len(processing_times) > 10:
            processing_times.pop(0)
        
        if len(detections) > 0:
            total_detections += len(detections)
            
            if frame_id % 30 == 0:
                print(f"Frame {frame_id}: Found {len(detections)} potholes")
        
        detected = len(detections) > 0
        confirmed = voter.update(detected)
        
        
        if confirmed and detections:
            for x1, y1, x2, y2, conf in detections:
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(
                    frame, 
                    f"POTHOLE {conf:.2f}", 
                    (x1, y1-5), 
                    cv2.FONT_HERSHEY_SIMPLEX, 
                    0.5, 
                    (0, 255, 0), 
                    2
                )
            
            cv2.putText(
                frame, 
                "POTHOLE DETECTED!", 
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 
                1.2, 
                (0, 255, 0), 
                2
            )
        
        
        frame_time = time.time() - loop_start
        frame_times.append(frame_time)
        if len(frame_times) > 10:
            frame_times.pop(0)
        
        avg_fps = int(1.0 / (np.mean(frame_times) + 1e-6))
        avg_process = np.mean(processing_times)
        
        
        cv2.putText(frame, f"FPS: {avg_fps}", (20, 25),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv2.putText(frame, f"Frame: {frame_id}", (20, 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        cv2.putText(frame, f"Detections: {total_detections}", (20, 80),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)
        cv2.putText(frame, f"Process: {avg_process:.0f}ms", (20, 100),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 165, 0), 1)
        
        
        display = cv2.resize(frame, (640, 360), interpolation=cv2.INTER_NEAREST)
        cv2.imshow("Pothole Detection", display)
        
        
        elapsed = time.time() - loop_start
        sleep_time = TARGET_FRAME_TIME - elapsed
        
        wait_ms = max(1, int(sleep_time * 1000))
        
        if cv2.waitKey(wait_ms) & 0xFF == ord('q'):
            break
        
        if time.time() - last_print_time > 2.0:
            print(f"Frame {frame_id} | FPS: {avg_fps} | Processing: {avg_process:.0f}ms | Detections: {total_detections}")
            last_print_time = time.time()

except KeyboardInterrupt:
    print("\n Stopped by user")

finally:
    cap.release()
    cv2.destroyAllWindows()
    
    total_time = time.time() - start_time
    actual_fps = frame_id / total_time if total_time > 0 else 0
    
    print(f"\n Final Statistics:")
    print(f"   Total frames: {frame_id}")
    print(f"   Total time: {total_time:.1f}s")
    print(f"   Average FPS: {actual_fps:.1f}")
    print(f"   Total detections: {total_detections}")
    print(f"   Avg processing: {np.mean(processing_times):.0f}ms")
    print(f"   Model: {model_name} (320x320)")
    print(" Done!")