from pathlib import Path
import numpy as np

root = Path("datasets/processed/labels")

areas = []
widths = []
heights = []
points = []
instances = 0

for split in ["train", "val", "test"]:
    split_dir = root / split

    for f in split_dir.glob("*.txt"):
        text = f.read_text().strip()

        if not text:
            continue

        for line in text.splitlines():
            p = line.split()

            if len(p) < 7:
                continue

            coords = np.array([float(x) for x in p[1:]], dtype=float)

            xs = coords[0::2]
            ys = coords[1::2]

            if len(xs) < 3:
                continue

            w = xs.max() - xs.min()
            h = ys.max() - ys.min()

            area = 0.5 * abs(
                np.dot(xs, np.roll(ys, 1))
                - np.dot(ys, np.roll(xs, 1))
            )

            widths.append(w)
            heights.append(h)
            areas.append(area)
            points.append(len(xs))

            instances += 1

print("========== CRACK ANNOTATION ANALYSIS ==========")
print(f"Total crack instances: {instances}")

for name, data in [
    ("width", widths),
    ("height", heights),
    ("polygon_area", areas),
    ("polygon_points", points),
]:
    a = np.array(data)

    print(f"\n{name}")
    print(f"  min    = {a.min():.6f}")
    print(f"  p10    = {np.percentile(a, 10):.6f}")
    print(f"  median = {np.median(a):.6f}")
    print(f"  p90    = {np.percentile(a, 90):.6f}")
    print(f"  max    = {a.max():.6f}")
