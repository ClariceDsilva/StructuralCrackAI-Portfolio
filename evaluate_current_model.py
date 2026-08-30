from pathlib import Path
from ultralytics import YOLO
import numpy as np

MODEL = Path("best.pt")
ROOT = Path("datasets/processed_v3/images/test")

THRESHOLDS = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50]

model = YOLO(str(MODEL))

images = sorted([
    p for p in ROOT.iterdir()
    if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}
])

negative_images = [
    p for p in images
    if p.name.lower().startswith("neg_")
]

positive_images = [
    p for p in images
    if not p.name.lower().startswith("neg_")
]

print("=" * 70)
print("CURRENT MODEL")
print("=" * 70)
print("Model:", MODEL.resolve())
print("Classes:", model.names)
print("Task:", model.task)
print()
print("Positive test images:", len(positive_images))
print("Negative test images:", len(negative_images))
print()

for conf in THRESHOLDS:

    print("=" * 70)
    print(f"CONFIDENCE THRESHOLD = {conf}")
    print("=" * 70)

    positive_detected = 0
    positive_missed = 0

    negative_detected = 0
    negative_clean = 0

    positive_confidences = []
    negative_confidences = []

    missed_positive_names = []
    false_positive_names = []

    for path in positive_images:

        result = model.predict(
            source=str(path),
            conf=conf,
            imgsz=640,
            verbose=False,
        )[0]

        if result.boxes is not None and len(result.boxes) > 0:

            positive_detected += 1

            confs = result.boxes.conf.detach().cpu().numpy()

            if len(confs):
                positive_confidences.append(float(np.max(confs)))

        else:

            positive_missed += 1
            missed_positive_names.append(path.name)

    for path in negative_images:

        result = model.predict(
            source=str(path),
            conf=conf,
            imgsz=640,
            verbose=False,
        )[0]

        if result.boxes is not None and len(result.boxes) > 0:

            negative_detected += 1

            confs = result.boxes.conf.detach().cpu().numpy()

            if len(confs):
                negative_confidences.append(float(np.max(confs)))

            false_positive_names.append(
                f"{path.name}: {float(np.max(confs)):.4f}"
            )

        else:

            negative_clean += 1

    positive_recall = (
        positive_detected / len(positive_images)
        if positive_images else 0
    )

    false_positive_rate = (
        negative_detected / len(negative_images)
        if negative_images else 0
    )

    print()
    print(f"Positive detected : {positive_detected}")
    print(f"Positive missed   : {positive_missed}")
    print(f"Image recall      : {positive_recall:.2%}")
    print()
    print(f"Negative clean    : {negative_clean}")
    print(f"False positives   : {negative_detected}")
    print(f"FP rate           : {false_positive_rate:.2%}")
    print()

    if positive_confidences:
        print(
            "Positive confidence:",
            f"min={min(positive_confidences):.4f}",
            f"median={np.median(positive_confidences):.4f}",
            f"max={max(positive_confidences):.4f}",
        )

    if negative_confidences:
        print(
            "Negative FP confidence:",
            f"min={min(negative_confidences):.4f}",
            f"median={np.median(negative_confidences):.4f}",
            f"max={max(negative_confidences):.4f}",
        )

    print()

    if false_positive_names:
        print("False positives:")
        for x in false_positive_names[:20]:
            print("  ", x)

    print()

    if missed_positive_names:
        print("First 20 missed positive images:")
        for x in missed_positive_names[:20]:
            print("  ", x)

    print()