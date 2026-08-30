from pathlib import Path
import numpy as np

root = Path("datasets/processed/labels")

areas = []
widths = []
heights = []

for split in ["train", "val", "test"]:
    for f in (root / split).glob("*.txt"):
        for line in f.read_text().splitlines():
            p = line.split()

            if len(p) < 7:
                continue

            c = np.array([float(x) for x in p[1:]])

            xs = c[0::2]
            ys = c[1::2]

            widths.append(xs.max() - xs.min())
            heights.append(ys.max() - ys.min())

            areas.append(
                0.5 * abs(
                    np.dot(xs, np.roll(ys, 1))
                    - np.dot(ys, np.roll(xs, 1))
                )
            )

widths = np.array(widths)
heights = np.array(heights)
areas = np.array(areas)

print("========== SMALL CRACK DISTRIBUTION ==========")

for threshold in [0.0025, 0.005, 0.01, 0.02, 0.05, 0.10]:

    count = np.sum(
        (widths < threshold) |
        (heights < threshold)
    )

    print(
        f"width OR height < {threshold:.4f}: "
        f"{count} / {len(widths)} "
        f"({100*count/len(widths):.2f}%)"
    )

print()
print("========== VERY SMALL POLYGON AREAS ==========")

for threshold in [0.00001, 0.00005, 0.0001, 0.0005, 0.001, 0.005]:

    count = np.sum(areas < threshold)

    print(
        f"area < {threshold:.5f}: "
        f"{count} / {len(areas)} "
        f"({100*count/len(areas):.2f}%)"
    )
