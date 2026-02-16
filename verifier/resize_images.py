import cv2
import os

BASE_DIR = "data"
SIZE = 64

for cls in ["pothole", "normal"]:
    folder = os.path.join(BASE_DIR, cls)
    for img in os.listdir(folder):
        path = os.path.join(folder, img)
        try:
            im = cv2.imread(path)
            if im is None:
                raise ValueError("Unreadable")
            im = cv2.resize(im, (SIZE, SIZE))
            cv2.imwrite(path, im)
        except:
            print("❌ Removing:", path)
            if os.path.exists(path):
                os.remove(path)

print("✅ Dataset cleaned & resized")