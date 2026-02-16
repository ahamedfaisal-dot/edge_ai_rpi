import torch
import torchvision.transforms as T
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from torch import nn, optim

# -------------------------
# Config
# -------------------------
DATA_DIR = "data"
BATCH_SIZE = 32
EPOCHS = 15
LR = 0.001

# -------------------------
# Dataset
# -------------------------
transform = T.Compose([
    T.Resize((64, 64)),
    T.Grayscale(),
    T.ToTensor()
])

dataset = ImageFolder(DATA_DIR, transform=transform)
loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

print("Classes:", dataset.classes)

# -------------------------
# CNN Model
# -------------------------
class CNNVerifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 8, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(8, 16, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(16 * 16 * 16, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)

model = CNNVerifier()
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=LR)

# -------------------------
# Training
# -------------------------
for epoch in range(EPOCHS):
    total_loss = 0
    for imgs, labels in loader:
        labels = labels.float().unsqueeze(1)
        preds = model(imgs)
        loss = criterion(preds, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch [{epoch+1}/{EPOCHS}] Loss: {total_loss:.4f}")

# -------------------------
# Save model
# -------------------------
torch.save(model.state_dict(), "cnn_model.pth")
print("✅ CNN model saved as cnn_model.pth")
