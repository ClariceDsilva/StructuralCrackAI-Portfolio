from pathlib import Path
from ultralytics import YOLO

# ============================================================
# CONFIG
# ============================================================

ROOT = Path("datasets/processed_v3")

POSITIVE_DIR = ROOT / "images" / "test"
NEGATIVE_DIR = ROOT / "images" / "test"

OLD_MODEL = "best.pt"
NEW_MODEL = "best_v1.pt"

CONF_THRESHOLDS = [0.05, 0.10, 0.15, 0.20, 0.25]


# ============================================================
# IMAGE HELPERS
# ============================================================

def get_images(folder):
    return sorted([
        p for p in folder.iterdir()
        if p.is_file()
        and p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}
    ])


def is_negative(path):
    return path.name.lower().startswith("neg_")


def is_positive(path):
    return not is_negative(path)


# ============================================================
# EVALUATION
# ============================================================

def evaluate(model_path):

    print()
    print("=" * 70)
    print(f"EVALUATING: {model_path}")
    print("=" * 70)

    model = YOLO(model_path)

    all_images = get_images(POSITIVE_DIR)

    positive_images = [
        p for p in all_images
        if is_positive(p)
    ]

    negative_images = [
        p for p in all_images
        if is_negative(p)
    ]

    print(f"Positive images: {len(positive_images)}")
    print(f"Negative images: {len(negative_images)}")

    # --------------------------------------------------------
    # Positive confidence scores
    # --------------------------------------------------------

    positive_scores = []

    for i, image_path in enumerate(positive_images, 1):

        result = model.predict(
            source=str(image_path),
            conf=0.001,
            imgsz=640,
            verbose=False,
        )[0]

        if result.boxes is None or len(result.boxes) == 0:
            score = 0.0
        else:
            score = float(result.boxes.conf.max().item())

        positive_scores.append(score)

        if i % 100 == 0:
            print(f"Positive: {i}/{len(positive_images)}")

    # --------------------------------------------------------
    # Negative confidence scores
    # --------------------------------------------------------

    negative_scores = []

    for i, image_path in enumerate(negative_images, 1):

        result = model.predict(
            source=str(image_path),
            conf=0.001,
            imgsz=640,
            verbose=False,
        )[0]

        if result.boxes is None or len(result.boxes) == 0:
            score = 0.0
        else:
            score = float(result.boxes.conf.max().item())

        negative_scores.append(score)

        if i % 100 == 0:
            print(f"Negative: {i}/{len(negative_images)}")

    # --------------------------------------------------------
    # Results
    # --------------------------------------------------------

    print()
    print("-" * 70)
    print("IMAGE-LEVEL RESULTS")
    print("-" * 70)

    for threshold in CONF_THRESHOLDS:

        positive_detected = sum(
            s >= threshold
            for s in positive_scores
        )

        false_positives = sum(
            s >= threshold
            for s in negative_scores
        )

        recall = positive_detected / len(positive_images)

        fpr = false_positives / len(negative_images)

        print(
            f"conf >= {threshold:.2f} | "
            f"recall = {recall:.4f} ({recall * 100:.2f}%) | "
            f"FP = {false_positives}/{len(negative_images)} "
            f"({fpr * 100:.2f}%)"
        )

    # --------------------------------------------------------
    # Hard positive distribution
    # --------------------------------------------------------

    print()
    print("-" * 70)
    print("POSITIVE CONFIDENCE DISTRIBUTION")
    print("-" * 70)

    for threshold in CONF_THRESHOLDS:

        count = sum(
            s < threshold
            for s in positive_scores
        )

        print(
            f"max confidence < {threshold:.3f}: "
            f"{count}/{len(positive_images)} "
            f"({count / len(positive_images) * 100:.2f}%)"
        )

    # --------------------------------------------------------
    # Top false positives
    # --------------------------------------------------------

    print()
    print("-" * 70)
    print("TOP FALSE POSITIVES")
    print("-" * 70)

    ranked_negative = sorted(
        zip(negative_scores, negative_images),
        reverse=True
    )

    for score, path in ranked_negative[:20]:

        print(
            f"{score:.6f}  {path.name}"
        )

    # --------------------------------------------------------
    # Lowest positive confidence
    # --------------------------------------------------------

    print()
    print("-" * 70)
    print("LOWEST POSITIVE CONFIDENCE")
    print("-" * 70)

    ranked_positive = sorted(
        zip(positive_scores, positive_images)
    )

    for score, path in ranked_positive[:30]:

        print(
            f"{score:.6f}  {path.name}"
        )

    return {
        "positive_scores": positive_scores,
        "negative_scores": negative_scores,
        "positive_images": positive_images,
        "negative_images": negative_images,
    }


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    old = evaluate(OLD_MODEL)
    new = evaluate(NEW_MODEL)

    print()
    print("=" * 70)
    print("MODEL COMPARISON COMPLETE")
    print("=" * 70)