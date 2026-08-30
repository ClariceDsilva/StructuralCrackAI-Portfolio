from pathlib import Path
import numpy as np

IMAGE_DIR = Path("datasets/processed_v3/images/test")
LABEL_DIR = Path("datasets/processed_v3/labels/test")

# Re-run the same evaluation logic used by evaluate_positive_test.py
from ultralytics import YOLO

MODEL_PATH = "best.pt"
CONF = 0.25
IMGSZ = 640

model = YOLO(MODEL_PATH)

image_files = sorted(
    p for p in IMAGE_DIR.iterdir()
    if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}
    and not p.name.startswith("neg_")
)

missed = []

print("=" * 60)
print("ANALYZING MISSED POSITIVE TEST IMAGES")
print("=" * 60)
print(f"Positive test images: {len(image_files)}")

for i, image_path in enumerate(image_files, 1):

    results = model.predict(
        source=str(image_path),
        conf=CONF,
        imgsz=IMGSZ,
        verbose=False,
    )

    result = results[0]

    detected = (
        result.masks is not None
        and len(result.masks.data) > 0
    )

    if not detected:
        missed.append(image_path)

    if i % 100 == 0:
        print(f"Processed {i}/{len(image_files)}")

print()
print(f"Missed images: {len(missed)}")
print()

# ------------------------------------------------------------
# Analyze ground-truth polygons
# ------------------------------------------------------------

all_widths = []
all_heights = []
all_areas = []
all_points = []

for image_path in missed:

    label_path = LABEL_DIR / (image_path.stem + ".txt")

    if not label_path.exists():
        print(f"WARNING: missing label: {image_path.name}")
        continue

    text = label_path.read_text().strip()

    if not text:
        print(f"WARNING: empty label: {image_path.name}")
        continue

    for line in text.splitlines():

        parts = line.split()

        if len(parts) < 7:
            continue

        coords = np.array(
            [float(x) for x in parts[1:]],
            dtype=float,
        )

        xs = coords[0::2]
        ys = coords[1::2]

        if len(xs) < 3:
            continue

        width = xs.max() - xs.min()
        height = ys.max() - ys.min()

        area = 0.5 * abs(
            np.dot(xs, np.roll(ys, 1))
            -
            np.dot(ys, np.roll(xs, 1))
        )

        all_widths.append(width)
        all_heights.append(height)
        all_areas.append(area)
        all_points.append(len(xs))


def report(name, values):

    if not values:
        print(f"{name}: no data")
        return

    a = np.asarray(values)

    print()
    print(name)
    print(f"  count  = {len(a)}")
    print(f"  min    = {a.min():.6f}")
    print(f"  p10    = {np.percentile(a, 10):.6f}")
    print(f"  median = {np.median(a):.6f}")
    print(f"  p90    = {np.percentile(a, 90):.6f}")
    print(f"  max    = {a.max():.6f}")


print("=" * 60)
print("GROUND-TRUTH CHARACTERISTICS OF MISSED CRACKS")
print("=" * 60)

report("width", all_widths)
report("height", all_heights)
report("polygon_area", all_areas)
report("polygon_points", all_points)

# ------------------------------------------------------------
# Threshold counts
# ------------------------------------------------------------

if all_widths:

    widths = np.asarray(all_widths)
    heights = np.asarray(all_heights)
    areas = np.asarray(all_areas)

    print()
    print("=" * 60)
    print("SMALL CRACKS AMONG MISSED IMAGES")
    print("=" * 60)

    for threshold in [
        0.0025,
        0.005,
        0.01,
        0.02,
        0.05,
        0.10,
    ]:

        count = np.sum(
            (widths < threshold)
            |
            (heights < threshold)
        )

        print(
            f"width OR height < {threshold:.4f}: "
            f"{count}/{len(widths)} "
            f"({100*count/len(widths):.2f}%)"
        )

    print()
    print("=" * 60)
    print("VERY SMALL AREAS AMONG MISSED IMAGES")
    print("=" * 60)

    for threshold in [
        0.00001,
        0.00005,
        0.00010,
        0.00050,
        0.00100,
        0.00500,
    ]:

        count = np.sum(areas < threshold)

        print(
            f"area < {threshold:.5f}: "
            f"{count}/{len(areas)} "
            f"({100*count/len(areas):.2f}%)"
        )

print()
print("=" * 60)
print("MISSED FILENAMES")
print("=" * 60)

for p in missed:
    print(p.name)