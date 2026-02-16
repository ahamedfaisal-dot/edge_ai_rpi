# fallback_anomaly.py
import numpy as np
import cv2

class FallbackAnomalyDetector:
    def __init__(self, threshold=2.8):
        self.threshold = threshold
        self.mean = None
        self.std = None
        self.features = []

    def extract_feature(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        return np.mean(gray)

    def fit(self, frame):
        val = self.extract_feature(frame)
        self.features.append(val)
        self.mean = np.mean(self.features)
        self.std = np.std(self.features) + 1e-6

    def score(self, frame):
        val = self.extract_feature(frame)
        return abs(val - self.mean) / self.std
