# fallback_anomaly.py
import numpy as np
import cv2

class FallbackAnomalyDetector:
    def __init__(self, threshold=5.0):
        """
        threshold: z-score threshold for anomaly (higher = stricter)
        Default raised from 2.8 to 5.0 for fewer false positives.
        """
        self.threshold = threshold
        self.mean = None
        self.std = None
        self.features = []

    def extract_feature(self, frame):
        """
        Extract combined feature: mean intensity + Laplacian texture variance.
        This makes the detector less sensitive to lighting changes.
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        mean_intensity = np.mean(gray)
        
        # Laplacian variance measures texture/edge sharpness
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        texture_var = laplacian.var()
        
        # Combine features (weighted)
        return mean_intensity + 0.01 * texture_var

    def fit(self, frame):
        val = self.extract_feature(frame)
        self.features.append(val)
        self.mean = np.mean(self.features)
        self.std = np.std(self.features) + 1e-6

    def score(self, frame):
        val = self.extract_feature(frame)
        return abs(val - self.mean) / self.std

