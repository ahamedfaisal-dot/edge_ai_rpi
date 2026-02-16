import torch
import torchvision.models as models
import torchvision.transforms as T
import cv2
import numpy as np

class FeatureExtractor:
    def __init__(self, device="cuda"):
        self.device = device

        model = models.mobilenet_v3_small(weights="DEFAULT")
        self.feature_net = model.features.eval().to(device)

        self.pool = torch.nn.AdaptiveAvgPool2d((1, 1))

        self.transform = T.Compose([
            T.ToTensor(),
            T.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

    @torch.no_grad()
    def extract(self, frame):
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (224, 224))

        tensor = self.transform(img).unsqueeze(0).to(self.device)

        feat = self.feature_net(tensor)
        feat = self.pool(feat)

        return feat.view(-1).cpu().numpy()
