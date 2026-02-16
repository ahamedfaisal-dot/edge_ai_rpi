# verifier/cnn_verifier.py

import torch
import torch.nn as nn
import cv2
import numpy as np

class CNNVerifier(nn.Module):
    def __init__(self, model_path, device="cpu", threshold=0.5):
        super().__init__()
        self.device = device
        self.threshold = threshold

        # ✅ SAME SHAPES as training (grayscale 64x64)
        self.features = nn.Sequential(
            nn.Conv2d(1, 8, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(8, 16, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        # ✅ FIXED: Match saved model architecture
        # classifier.0 = Flatten, classifier.1 = Linear(4096,32), 
        # classifier.2 = ReLU, classifier.3 = Linear(32,1), classifier.4 = Sigmoid
        self.classifier = nn.Sequential(
            nn.Flatten(),           # Index 0
            nn.Linear(4096, 32),    # Index 1 (matches saved classifier.1)
            nn.ReLU(),              # Index 2
            nn.Linear(32, 1),       # Index 3 (matches saved classifier.3)
            nn.Sigmoid()            # Index 4
        )

        # 🔥 LOAD NON-STRICTLY (KEY FIX)
        state = torch.load(model_path, map_location=device)
        self.load_state_dict(state, strict=False)

        self.to(device)
        self.eval()

    def preprocess(self, img):
        """Preprocess patch exactly as training: BGR→Gray, resize 64x64, normalize."""
        if img is None or img.size == 0:
            return None

        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = cv2.resize(img, (64, 64))
        img = img.astype(np.float32) / 255.0
        img = torch.from_numpy(img).unsqueeze(0).unsqueeze(0)
        return img.to(self.device)

    def forward(self, x):
        x = self.features(x)
        # Flatten is now in classifier, no manual view needed
        return self.classifier(x)

    def verify_with_score(self, patch):
        """
        Returns (is_verified: bool, confidence_score: float).
        Use this for logging and temporal voting.
        """
        img = self.preprocess(patch)
        if img is None:
            return False, 0.0

        with torch.no_grad():
            score = self(img).item()

        return score > self.threshold, score

    def verify(self, patch):
        """Simple boolean verification (backward compatible)."""
        verified, _ = self.verify_with_score(patch)
        return verified



