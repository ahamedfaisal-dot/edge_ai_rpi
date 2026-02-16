import numpy as np

class MahalanobisDetector:
    def __init__(self, threshold=3.0):
        self.threshold = threshold
        self.features = []
        self.mean = None
        self.inv_cov = None

    def fit(self, feature):
        self.features.append(feature)

    def finalize(self):
        X = np.array(self.features)
        self.mean = X.mean(axis=0)

        cov = np.cov(X, rowvar=False) + 1e-6 * np.eye(X.shape[1])
        self.inv_cov = np.linalg.inv(cov)

    def score(self, feature):
        diff = feature - self.mean
        dist = np.sqrt(diff.T @ self.inv_cov @ diff)
        return dist
