# Edge AI Performance Optimization Results

This document showcases the performance optimization journey of our edge AI pothole detection system, from initial laptop testing to final deployment on Raspberry Pi edge hardware.

## Performance Progression

### 1. Initial Development - Laptop Testing (25-45 FPS)

During the initial development phase, we tested the AI model on a laptop to establish baseline performance metrics. The system achieved impressive frame rates between **25-45 FPS**, demonstrating the model's capability under optimal hardware conditions.

**Demo Output:**

- 📹 [View Laptop Performance Video (Google Drive)](https://drive.google.com/file/d/1tuFs8f_KjfSUI5VHKQDSmJbHvGje8tMz/view?usp=sharing)

---

### 2. First Edge Deployment - Raspberry Pi (1 FPS)

When we first deployed the model to the Raspberry Pi edge device without optimization(converted pytorch model to onnx with image parameter 640), the performance dropped significantly to **1 FPS**. This highlighted the computational constraints of edge hardware and the need for optimization.

**Demo Output:**

- 📹 [View Initial Edge Performance Video (Google Drive)](https://drive.google.com/file/d/1jzODXXvMJjNPSZSYUZu8VvZMC2bm5ArQ/view?usp=drive_link)

---

### 3. First Optimization - Improved Edge Performance (4 FPS)

After implementing initial optimizations (model quantization(converted pytorch model to onnx with image parameter 480), inference optimizations, etc.), we achieved a **4x performance improvement** to **4 FPS**. This showed promising results but still required further enhancement for real-time applications.

**Demo Output:**

- 📹 [View Optimized Edge Performance Video (Google Drive)](https://drive.google.com/file/d/1KxWjq-N3JVdJNm04kwWShF8XRr2ab6DC/view?usp=sharing)

---

### 4. Final Optimization - Production Edge Deployment (7 FPS)

Through comprehensive optimization(converted pytorch model to onnx with image parameter 360) strategies including:

- Advanced model quantization
- Efficient preprocessing pipelines
- Optimized inference configurations
- Memory management improvements

We achieved **7 FPS** on the Raspberry Pi, representing a **7x improvement** over the initial edge deployment. This performance level enables near-real-time pothole detection on edge hardware.

**Demo Output:**

- 📹 [View Final Edge Performance Video (Google Drive)](https://drive.google.com/file/d/1aKS5fzINw_lEwPAybtC8g8YJhdgl4LF6/view?usp=sharing)

---

### 5. Real-World Field Testing - Bike Pothole Detection

To validate the system's real-world applicability, we conducted field testing by mounting the Raspberry Pi on a bike and riding through roads with potholes. The system successfully detected potholes in real-world conditions, demonstrating practical viability.

**Field Test Output:**

- 📹 [View Real-World Bike Testing Video (Google Drive)](https://drive.google.com/file/d/1fmIOCZe0yc1K5nB2kklY15GKl-EBwTE5/view?usp=sharing)

---

## Key Achievements

| Test Phase         | Hardware     | FPS   | Performance vs Laptop | Notes              |
| ------------------ | ------------ | ----- | --------------------- | ------------------ |
| Development        | Laptop       | 25-45 | Baseline (100%)       | Optimal conditions |
| Initial Edge       | Raspberry Pi | 1     | ~2% of laptop         | Unoptimized        |
| First Optimization | Raspberry Pi | 4     | ~9% of laptop         | 4x improvement     |
| Final Optimization | Raspberry Pi | 7     | ~16% of laptop        | 7x improvement     |
| Real-World Test    | Raspberry Pi | ~8    | ~16% of laptop        | Field validated    |

## Notes

- All edge device tests were conducted on Raspberry Pi
- The final 7 FPS performance enables practical real-time pothole detection
- Real-world testing confirms the system's viability for deployment on moving vehicles
- Video files are stored on Google Drive due to GitHub's file size limitations (>100MB)
