from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
import math
import random

ROOT = Path("datasets/processed_v3/images/test")
OUT = Path("negative_contact_sheets")
OUT.mkdir(exist_ok=True)

images = sorted(
    [
        p for p in ROOT.iterdir()
        if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}
        and p.name.lower().startswith("neg_")
    ]
)

print(f"Found {len(images)} negative test images.")

# Make results reproducible
random.seed(42)

# Put the two known false positives first
priority_names = {
    "neg_sdnet_7116-67.jpg",
    "neg_sdnet_7083-27.jpg",
}

priority = [p for p in images if p.name in priority_names]
remaining = [p for p in images if p.name not in priority_names]

random.shuffle(remaining)

ordered = priority + remaining

# 72 images per sheet = 9 columns x 8 rows
COLS = 9
ROWS = 8
THUMB_W = 180
THUMB_H = 150
LABEL_H = 35

PER_SHEET = COLS * ROWS

font = ImageFont.load_default()

for sheet_index in range(math.ceil(len(ordered) / PER_SHEET)):

    batch = ordered[
        sheet_index * PER_SHEET:
        (sheet_index + 1) * PER_SHEET
    ]

    canvas = Image.new(
        "RGB",
        (
            COLS * THUMB_W,
            ROWS * (THUMB_H + LABEL_H)
        ),
        "black"
    )

    draw = ImageDraw.Draw(canvas)

    for i, path in enumerate(batch):

        row = i // COLS
        col = i % COLS

        x = col * THUMB_W
        y = row * (THUMB_H + LABEL_H)

        try:
            img = Image.open(path).convert("RGB")
            img.thumbnail((THUMB_W - 8, THUMB_H - 8))

            px = x + (THUMB_W - img.width) // 2
            py = y + (THUMB_H - img.height) // 2

            canvas.paste(img, (px, py))

        except Exception as e:
            draw.rectangle(
                [x, y, x + THUMB_W, y + THUMB_H],
                outline="red"
            )
            draw.text(
                (x + 5, y + 5),
                "ERROR",
                fill="red",
                font=font
            )

        # Short filename
        name = path.stem

        # Highlight the two known false positives
        if path.name in priority_names:
            label = "FP: " + name
            fill = "red"
        else:
            label = name
            fill = "white"

        # Truncate if necessary
        if len(label) > 25:
            label = label[:22] + "..."

        draw.text(
            (x + 3, y + THUMB_H + 3),
            label,
            fill=fill,
            font=font
        )

    output = OUT / f"negative_sheet_{sheet_index + 1:02d}.jpg"
    canvas.save(output, quality=92)

    print(f"Created: {output}")

print()
print("Done.")
print(f"Contact sheets are in: {OUT.resolve()}")