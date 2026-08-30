from pathlib import Path
from ultralytics import YOLO
import numpy as np

MODEL = "best.pt"

IMAGE_DIR = Path("datasets/processed/images/test")

CONF_THRESHOLDS = [0.05, 0.10, 0.15, 0.20, 0.25]

model = YOLO(MODEL)

images = sorted([
    p for p in IMAGE_DIR.iterdir()
    if p.suffix.lower() in [".jpg", ".jpeg", ".png", ".bmp"]
])

print("=" * 70)
print("TESTING DETECTION RECALL AT DIFFERENT CONFIDENCE THRESHOLDS")
print("=" * 70)
print(f"Test images: {len(images)}")
print()

scores = []

for i, image_path in enumerate(images, 1):

    result = model.predict(
        source=str(image_path),
        imgsz=640,
        conf=0.001,
        verbose=False,
    )[0]

    if result.boxes is None or len(result.boxes) == 0:
        max_conf = 0.0
    else:
        max_conf = float(result.boxes.conf.max().cpu().item())

    scores.append((image_path.name, max_conf))

    if i % 100 == 0:
        print(f"Processed {i}/{len(images)}")


print()
print("=" * 70)
print("IMAGE-LEVEL RECALL")
print("=" * 70)

for threshold in CONF_THRESHOLDS:

    detected = sum(
        1 for _, score in scores
        if score >= threshold
    )

    recall = detected / len(scores)

    print(
        f"conf >= {threshold:.2f}: "
        f"{detected}/{len(scores)} "
        f"({recall * 100:.2f}%)"
    )


print()
print("=" * 70)
print("LOWEST CONFIDENCE DISTRIBUTION")
print("=" * 70)

values = np.array([s for _, s in scores])

for threshold in [
    0.001,
    0.005,
    0.01,
    0.02,
    0.05,
    0.10,
    0.15,
    0.20,
    0.25,
]:
    count = np.sum(values < threshold)
    print(
        f"max confidence < {threshold:.3f}: "
        f"{count}/{len(values)} "
        f"({count / len(values) * 100:.2f}%)"
    )


print()
print("=" * 70)
print("LOWEST CONFIDENCE IMAGES")
print("=" * 70)

for name, score in sorted(scores, key=lambda x: x[1])[:100]:
    print(f"{score:.6f}  {name}")