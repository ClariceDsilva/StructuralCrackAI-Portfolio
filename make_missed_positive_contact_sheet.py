from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

ROOT = Path("datasets/processed_v3")
IMAGE_DIR = ROOT / "images" / "test"
LABEL_DIR = ROOT / "labels" / "test"

OUTPUT_DIR = Path("missed_positive_contact_sheets")
OUTPUT_DIR.mkdir(exist_ok=True)

missed_file = Path("missed_positive_images.txt")

# These are the 95 images identified by evaluate_positive_test.py.
# We regenerate the list so there is no manual copying.

from ultralytics import YOLO

MODEL = "best.pt"
model = YOLO(MODEL)

missed = []

positive_images = []

for img in IMAGE_DIR.iterdir():
    if img.suffix.lower() not in [".jpg", ".jpeg", ".png", ".bmp"]:
        continue

    label = LABEL_DIR / (img.stem + ".txt")

    if label.exists() and label.read_text().strip():
        positive_images.append(img)

print(f"Checking {len(positive_images)} positive test images...")

for i, img in enumerate(positive_images, 1):

    result = model.predict(
        source=str(img),
        conf=0.25,
        imgsz=640,
        verbose=False,
    )[0]

    count = 0 if result.masks is None else len(result.masks.data)

    if count == 0:
        missed.append(img)

    if i % 100 == 0:
        print(f"Processed {i}/{len(positive_images)}")

missed_file.write_text(
    "\n".join(str(x) for x in missed),
    encoding="utf-8",
)

print(f"Missed images: {len(missed)}")

# ------------------------------------------------------------
# Contact sheets
# ------------------------------------------------------------

THUMB_W = 320
THUMB_H = 240
COLS = 4
ROWS = 5

PER_SHEET = COLS * ROWS

font = ImageFont.load_default()

for sheet_index in range(0, len(missed), PER_SHEET):

    batch = missed[sheet_index:sheet_index + PER_SHEET]

    canvas = Image.new(
        "RGB",
        (COLS * THUMB_W, ROWS * THUMB_H),
        "black",
    )

    draw = ImageDraw.Draw(canvas)

    for j, img_path in enumerate(batch):

        try:
            img = Image.open(img_path).convert("RGB")
            img.thumbnail((THUMB_W - 10, THUMB_H - 35))

            x = (j % COLS) * THUMB_W
            y = (j // COLS) * THUMB_H

            px = x + (THUMB_W - img.width) // 2
            py = y + 5

            canvas.paste(img, (px, py))

            draw.text(
                (x + 5, y + THUMB_H - 25),
                img_path.name,
                fill="white",
                font=font,
            )

        except Exception as e:
            print(f"Could not load {img_path}: {e}")

    sheet_number = sheet_index // PER_SHEET + 1

    output = OUTPUT_DIR / f"missed_sheet_{sheet_number:02d}.jpg"

    canvas.save(output, quality=92)

    print(f"Created: {output}")

print()
print(f"Contact sheets: {OUTPUT_DIR.resolve()}")