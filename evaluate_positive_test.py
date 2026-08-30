from pathlib import Path
from ultralytics import YOLO
import numpy as np

MODEL = "best.pt"
IMAGE_DIR = Path("datasets/processed_v3/images/test")

model = YOLO(MODEL)

positive_images = []

for img in IMAGE_DIR.iterdir():
    if img.suffix.lower() not in [".jpg", ".jpeg", ".png", ".bmp"]:
        continue

    label = IMAGE_DIR.parent.parent / "labels" / "test" / (img.stem + ".txt")

    if label.exists() and label.read_text().strip():
        positive_images.append(img)

print("=" * 60)
print("POSITIVE TEST EVALUATION")
print("=" * 60)

print(f"Positive test images: {len(positive_images)}")

missed = []
detected = []

for i, img in enumerate(positive_images, 1):

    result = model.predict(
        source=str(img),
        conf=0.25,
        imgsz=640,
        verbose=False,
    )[0]

    count = 0 if result.masks is None else len(result.masks.data)

    if count == 0:
        missed.append(img.name)
    else:
        detected.append(img.name)

    if i % 100 == 0:
        print(f"Processed {i}/{len(positive_images)}")

print()
print(f"Detected images: {len(detected)}")
print(f"Missed images:   {len(missed)}")

if positive_images:
    recall = len(detected) / len(positive_images)
    print(f"Image-level recall: {recall:.4f} ({recall*100:.2f}%)")

print()
print("First 50 missed images:")

for name in missed[:50]:
    print(" ", name)