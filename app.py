import json
import time
from pathlib import Path
import textwrap

import cv2
import numpy as np
import streamlit as st
import torch

from PIL import Image
from skimage.morphology import skeletonize
from ultralytics import YOLO


# ============================================================
# CONFIG
# ============================================================

st.set_page_config(
    page_title="StructuralCrackAI",
    page_icon="S",
    layout="wide",
    initial_sidebar_state="collapsed",
)


BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "best.pt"
CALIBRATION_PATH = BASE_DIR / "calibration.json"

CONFIDENCE_THRESHOLD = 0.25
IMAGE_SIZE = 640


# ============================================================
# DARK INTERACTIVE FRONTEND
# ============================================================

st.markdown(
    """
    <style>

    /* ---------- MAIN ---------- */

    .stApp {
        background:
            radial-gradient(circle at 20% 10%, #102030 0%, transparent 30%),
            radial-gradient(circle at 80% 90%, #071a20 0%, transparent 35%),
            #03070b;
        color: #e8f0f5;
    }

    .block-container {
        max-width: 1400px;
        padding-top: 2rem;
        padding-bottom: 3rem;
    }

    /* ---------- HEADER ---------- */

    .hero {
        padding: 35px;
        border-radius: 22px;
        background:
            linear-gradient(
                135deg,
                rgba(12, 28, 42, 0.95),
                rgba(4, 10, 16, 0.98)
            );
        border: 1px solid rgba(90, 180, 255, 0.18);
        box-shadow: 0 20px 60px rgba(0,0,0,.35);
        margin-bottom: 25px;
    }

    .hero-title {
        font-size: 46px;
        font-weight: 800;
        letter-spacing: -1px;
        margin-bottom: 5px;
    }

    .hero-subtitle {
        color: #8da6b8;
        font-size: 16px;
        letter-spacing: 2px;
    }

    /* ---------- STATUS ---------- */

    .status-card {
        padding: 18px;
        border-radius: 16px;
        background: rgba(10,20,28,.85);
        border: 1px solid rgba(255,255,255,.07);
        text-align: center;
    }

    .status-value {
        font-size: 22px;
        font-weight: 700;
    }

    .status-label {
        color: #7890a0;
        font-size: 12px;
        letter-spacing: 1px;
        margin-bottom: 5px;
    }

    /* ---------- SCANNER ---------- */

    .scanner {
        height: 260px;
        border-radius: 22px;
        position: relative;
        overflow: hidden;
        background:
            linear-gradient(
                90deg,
                rgba(20,40,55,.5) 1px,
                transparent 1px
            ),
            linear-gradient(
                rgba(20,40,55,.5) 1px,
                transparent 1px
            ),
            #020609;
        background-size: 30px 30px;
        border: 1px solid rgba(70,170,230,.25);
        box-shadow: inset 0 0 60px rgba(0,120,180,.08);
    }

    .scanner-line {
        position: absolute;
        left: 0;
        right: 0;
        height: 3px;
        background: #27d9ff;
        box-shadow:
            0 0 10px #27d9ff,
            0 0 30px #27d9ff,
            0 0 60px rgba(39,217,255,.7);
        animation: scan 2.3s linear infinite;
    }

    @keyframes scan {
        0% { top: -5px; }
        100% { top: 100%; }
    }

    .scanner-center {
        position: absolute;
        inset: 0;
        display: flex;
        justify-content: center;
        align-items: center;
        flex-direction: column;
    }

    .scanner-icon {
        font-size: 60px;
        animation: pulse 1.4s infinite;
    }

    @keyframes pulse {
        0%,100% { transform: scale(1); opacity:.7; }
        50% { transform: scale(1.12); opacity:1; }
    }

    .scanner-text {
        margin-top: 10px;
        font-size: 17px;
        font-weight: 600;
        color: #b9eaff;
    }

    /* ---------- RESULT CARDS ---------- */

    .result-card {
        padding: 22px;
        border-radius: 18px;
        background: rgba(9,17,24,.95);
        border: 1px solid rgba(255,255,255,.07);
        margin-bottom: 20px;
    }

    .result-title {
        font-size: 22px;
        font-weight: 700;
        margin-bottom: 18px;
    }

    .metric-box {
        padding: 16px;
        border-radius: 13px;
        background: rgba(255,255,255,.035);
        border: 1px solid rgba(255,255,255,.06);
        margin-bottom: 10px;
    }

    .metric-name {
        color: #8096a6;
        font-size: 12px;
        text-transform: uppercase;
        letter-spacing: 1px;
    }

    .metric-value {
        font-size: 22px;
        font-weight: 700;
        margin-top: 4px;
    }

    .low {
        color: #46e6a1;
    }

    .moderate {
        color: #ffc857;
    }

    .high {
        color: #ff5f5f;
    }

    /* ---------- UPLOAD ---------- */

    [data-testid="stFileUploader"] {
        background: rgba(8,16,23,.8);
        border: 1px dashed rgba(55,190,255,.35);
        border-radius: 18px;
        padding: 15px;
    }

    /* ---------- BUTTON ---------- */

    .stButton > button {
        border-radius: 13px;
        height: 52px;
        font-weight: 700;
        letter-spacing: .5px;
    }

    /* ---------- FOOTER ---------- */

    .footer {
        text-align: center;
        color: #526674;
        padding: 25px;
        font-size: 13px;
    }

    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# HEADER
# ============================================================

st.markdown(
    """
    <div class="hero">
        <div class="hero-title">StructuralCrackAI</div>
        <div class="hero-subtitle">
            AI-POWERED STRUCTURAL CRACK INSPECTION SYSTEM
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# LOAD CALIBRATION
# ============================================================

@st.cache_data
def load_calibration():

    if not CALIBRATION_PATH.exists():
        return None

    with open(CALIBRATION_PATH, "r") as f:
        data = json.load(f)

    return data.get("pixels_per_mm")


pixels_per_mm = load_calibration()


# ============================================================
# LOAD YOLO
# ============================================================

@st.cache_resource
def load_yolo():

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model not found: {MODEL_PATH}"
        )

    return YOLO(str(MODEL_PATH))


# ============================================================
# HUGGING FACE PIPELINE LOADER
# ============================================================

def get_hf_pipeline():
    # Lazy import prevents Streamlit startup/import-order issues.
    from transformers import pipeline as hf_pipeline
    return hf_pipeline

# ============================================================
# INPUT SURFACE VALIDATION
# ============================================================

@st.cache_resource
def load_surface_classifier():

    device = 0 if torch.cuda.is_available() else -1

    hf_pipeline = get_hf_pipeline()

    return hf_pipeline(
        task="zero-shot-image-classification",
        model="openai/clip-vit-base-patch32",
        device=device,
    )


def validate_structural_surface(image):

    classifier = load_surface_classifier()

    labels = [
        "a concrete structural surface",
        "a concrete wall or building surface",
        "a bridge or structural concrete surface",
        "a road or pavement surface",
        "an unrelated object or scene",
        "a person",
        "an animal",
        "food",
        "a document or screenshot",
        "a computer or electronic device",
    ]

    results = classifier(
        image,
        candidate_labels=labels,
    )

    structural_labels = {
        "a concrete structural surface",
        "a concrete wall or building surface",
        "a bridge or structural concrete surface",
        "a road or pavement surface",
    }

    structural_score = sum(
        item["score"]
        for item in results
        if item["label"] in structural_labels
    )

    top_result = results[0]

    is_structural = structural_score >= 0.45

    return {
        "is_structural": is_structural,
        "structural_score": structural_score,
        "top_label": top_result["label"],
        "top_score": top_result["score"],
    }


# ============================================================
# LOAD DEPTH MODEL
# ============================================================

@st.cache_resource
def load_depth_model():

    device = 0 if torch.cuda.is_available() else -1

    hf_pipeline = get_hf_pipeline()

    depth_pipe = hf_pipeline(
        task="depth-estimation",
        model="depth-anything/Depth-Anything-V2-Small-hf",
        device=device,
    )

    return depth_pipe


# ============================================================
# DEPTH MAP
# ============================================================

def estimate_depth_map(image, depth_pipe):

    result = depth_pipe(image)

    depth_array = np.array(
        result["depth"]
    ).astype(np.float32)

    depth_resized = cv2.resize(
        depth_array,
        (image.width, image.height),
        interpolation=cv2.INTER_LINEAR,
    )

    return depth_resized


# ============================================================
# DEPTH CONTRAST
# ============================================================

def get_crack_depth_score(depth_map, mask):

    if depth_map.shape != mask.shape:

        depth_map = cv2.resize(
            depth_map,
            (mask.shape[1], mask.shape[0]),
            interpolation=cv2.INTER_LINEAR,
        )

    mask_uint8 = (
        (mask > 0.5).astype(np.uint8) * 255
    )

    kernel = np.ones(
        (15, 15),
        np.uint8,
    )

    dilated = cv2.dilate(
        mask_uint8,
        kernel,
        iterations=1,
    )

    ring = cv2.subtract(
        dilated,
        mask_uint8,
    )

    crack_pixels = depth_map[
        mask_uint8 > 0
    ]

    ring_pixels = depth_map[
        ring > 0
    ]

    if (
        len(crack_pixels) == 0
        or len(ring_pixels) == 0
    ):
        return None, None, None

    mean_crack_depth = float(
        np.mean(crack_pixels)
    )

    surrounding_depth = float(
        np.mean(ring_pixels)
    )

    depth_contrast = (
        surrounding_depth
        - mean_crack_depth
    )

    return (
        mean_crack_depth,
        surrounding_depth,
        depth_contrast,
    )


# ============================================================
# SEVERITY
# ============================================================

def classify_severity(
    area_mm2,
    length_mm,
    avg_width_mm,
    max_width_mm,
    confidence,
):

    score = 0

    if max_width_mm is not None:

        if max_width_mm >= 3:
            score += 3

        elif max_width_mm >= 1:
            score += 2

        else:
            score += 1

    if length_mm is not None:

        if length_mm >= 200:
            score += 2

        elif length_mm >= 50:
            score += 1

    if area_mm2 is not None:

        if area_mm2 >= 5000:
            score += 2

        elif area_mm2 >= 1000:
            score += 1

    if (
        confidence is not None
        and confidence < 0.5
    ):
        score -= 1

    if score >= 6:
        return "High"

    elif score >= 3:
        return "Moderate"

    return "Low"


# ============================================================
# STRUCTURAL INTELLIGENCE
# ============================================================

def structural_intelligence(
    confidence,
    area_mm2,
    length_mm,
    avg_width_mm,
    max_width_mm,
    relative_depth,
):

    score = 0

    # Confidence
    if confidence >= 0.80:
        score += 10

    elif confidence >= 0.60:
        score += 8

    elif confidence >= 0.40:
        score += 5

    # Area
    if area_mm2 > 300:
        score += 15

    elif area_mm2 > 150:
        score += 10

    elif area_mm2 > 50:
        score += 5

    # Length
    if length_mm > 100:
        score += 15

    elif length_mm > 60:
        score += 10

    elif length_mm > 30:
        score += 5

    # Average width
    if avg_width_mm > 4:
        score += 20

    elif avg_width_mm > 2:
        score += 12

    elif avg_width_mm > 1:
        score += 6

    # Maximum width
    if max_width_mm > 8:
        score += 15

    elif max_width_mm > 5:
        score += 10

    elif max_width_mm > 2:
        score += 5

    # Relative depth
    if relative_depth is not None:

        if relative_depth > 0.80:
            score += 20

        elif relative_depth > 0.60:
            score += 15

        elif relative_depth > 0.40:
            score += 10

        elif relative_depth > 0.20:
            score += 5

    integrity = max(
        0,
        100 - score
    )

    if score >= 70:

        level = "High"
        maintenance = "Immediate inspection"
        priority = "Level 1"

    elif score >= 40:

        level = "Moderate"
        maintenance = "Inspect within 30 days"
        priority = "Level 2"

    else:

        level = "Low"
        maintenance = "Routine monitoring"
        priority = "Level 3"

    return {
        "integrity": round(integrity, 1),
        "risk_score": score,
        "risk_level": level,
        "maintenance": maintenance,
        "priority": priority,
    }


# ============================================================
# ENGINEERING FEATURES
# ============================================================

def engineering_features(mask):

    mask = (
        mask > 0.5
    ).astype(np.uint8)

    contours, _ = cv2.findContours(
        mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )

    if len(contours) == 0:

        return {
            "aspect_ratio": 0,
            "compactness": 0,
            "skeleton_density": 0,
        }

    cnt = max(
        contours,
        key=cv2.contourArea,
    )

    x, y, w, h = cv2.boundingRect(cnt)

    aspect_ratio = (
        max(w, h)
        / (min(w, h) + 1e-6)
    )

    perimeter = cv2.arcLength(
        cnt,
        True,
    )

    area = cv2.contourArea(cnt)

    compactness = 0

    if perimeter > 0:

        compactness = (
            4 * np.pi * area
            / (perimeter ** 2)
        )

    skeleton = skeletonize(
        mask.astype(bool)
    )

    skeleton_pixels = np.sum(
        skeleton
    )

    mask_pixels = np.sum(mask)

    skeleton_density = 0

    if mask_pixels > 0:

        skeleton_density = (
            skeleton_pixels
            / mask_pixels
        )

    return {
        "aspect_ratio": round(
            float(aspect_ratio),
            4,
        ),
        "compactness": round(
            float(compactness),
            4,
        ),
        "skeleton_density": round(
            float(skeleton_density),
            4,
        ),
    }


# ============================================================
# ANALYZE ONE CRACK
# ============================================================

def analyze_single_crack(
    mask,
    confidence,
    depth_map,
    calibration,
):

    mask = (
        mask > 0.5
    ).astype(np.uint8)

    # Area
    area_pixels = int(
        np.sum(mask)
    )

    total_pixels = (
        mask.shape[0]
        * mask.shape[1]
    )

    area_percentage = (
        area_pixels
        / total_pixels
    ) * 100

    # Skeleton length
    skeleton = skeletonize(
        mask.astype(bool)
    )

    length_pixels = int(
        np.sum(skeleton)
    )

    # Width
    distance = cv2.distanceTransform(
        mask,
        cv2.DIST_L2,
        5,
    )

    valid_distances = (
        distance[distance > 0]
    )

    if len(valid_distances) > 0:

        avg_width_pixels = (
            2 * float(
                np.mean(
                    valid_distances
                )
            )
        )

        max_width_pixels = (
            2 * float(
                np.max(distance)
            )
        )

    else:

        avg_width_pixels = 0
        max_width_pixels = 0

    # Real-world conversion
    if calibration:

        area_mm2 = (
            area_pixels
            / calibration ** 2
        )

        length_mm = (
            length_pixels
            / calibration
        )

        avg_width_mm = (
            avg_width_pixels
            / calibration
        )

        max_width_mm = (
            max_width_pixels
            / calibration
        )

    else:

        area_mm2 = None
        length_mm = None
        avg_width_mm = None
        max_width_mm = None

    # Relative depth
    (
        mean_depth,
        surrounding_depth,
        depth_contrast,
    ) = get_crack_depth_score(
        depth_map,
        mask,
    )

    # Engineering features
    engineering = engineering_features(
        mask
    )

    # Severity
    severity = classify_severity(
        area_mm2,
        length_mm,
        avg_width_mm,
        max_width_mm,
        confidence,
    )

    # Structural intelligence
    intelligence = structural_intelligence(
        confidence,
        area_mm2,
        length_mm,
        avg_width_mm,
        max_width_mm,
        mean_depth,
    )

    return {

        "confidence": round(
            float(confidence),
            3,
        ),

        "area_pixels": area_pixels,

        "area_percentage": round(
            area_percentage,
            2,
        ),

        "length_pixels": length_pixels,

        "avg_width_pixels": round(
            avg_width_pixels,
            2,
        ),

        "max_width_pixels": round(
            max_width_pixels,
            2,
        ),

        "area_mm2": (
            round(area_mm2, 2)
            if area_mm2 is not None
            else None
        ),

        "length_mm": (
            round(length_mm, 2)
            if length_mm is not None
            else None
        ),

        "average_width_mm": (
            round(avg_width_mm, 3)
            if avg_width_mm is not None
            else None
        ),

        "maximum_width_mm": (
            round(max_width_mm, 3)
            if max_width_mm is not None
            else None
        ),

        "relative_depth": (
            round(mean_depth, 3)
            if mean_depth is not None
            else None
        ),

        "depth_contrast": (
            round(depth_contrast, 4)
            if depth_contrast is not None
            else None
        ),

        "aspect_ratio":
            engineering["aspect_ratio"],

        "skeleton_density":
            engineering["skeleton_density"],

        "compactness":
            engineering["compactness"],

        "integrity_index":
            intelligence["integrity"],

        "risk_score":
            intelligence["risk_score"],

        "risk_level":
            intelligence["risk_level"],

        "maintenance":
            intelligence["maintenance"],

        "priority":
            intelligence["priority"],

        "severity":
            severity,
    }


# ============================================================
# HERO STATUS
# ============================================================

c1, c2, c3, c4 = st.columns(4)

with c1:
    st.markdown(
        """
        <div class="status-card">
            <div class="status-label">AI MODEL</div>
            <div class="status-value">YOLO11n-Seg</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with c2:
    st.markdown(
        """
        <div class="status-card">
            <div class="status-label">DEPTH ENGINE</div>
            <div class="status-value">Depth Anything V2</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with c3:
    calibration_text = (
        f"{pixels_per_mm} px/mm"
        if pixels_per_mm
        else "Not calibrated"
    )

    st.markdown(
    textwrap.dedent(
        f"""
        <div class="status-card">
            <div class="status-label">CALIBRATION</div>
            <div class="status-value">{calibration_text}</div>
        </div>
        """
    ),
    unsafe_allow_html=True,
)

with c4:
    device_text = (
        "GPU"
        if torch.cuda.is_available()
        else "CPU"
    )

    st.markdown(
        f"""
        <div class="status-card">
            <div class="status-label">PROCESSOR</div>
            <div class="status-value">{device_text}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


st.write("")


# ============================================================
# UPLOAD
# ============================================================

st.markdown(
    "## Structural Inspection"
)

st.write(
    "Upload a structural surface image and start "
    "an AI-assisted crack inspection."
)

uploaded_file = st.file_uploader(
    "Drop your structural image here",
    type=[
        "jpg",
        "jpeg",
        "png",
        "bmp",
        "webp",
    ],
)


# ============================================================
# IMAGE UPLOADED
# ============================================================

if uploaded_file:

    image = Image.open(
        uploaded_file
    ).convert("RGB")

    # ========================================================
    # INPUT SURFACE VALIDATION
    # ========================================================

    surface_check = validate_structural_surface(image)

    if not surface_check["is_structural"]:

        st.error(
            "Invalid inspection image"
        )

        st.warning(
            "Please upload an appropriate structural/concrete "
            "surface image such as a concrete wall, bridge, "
            "building surface, pavement, tunnel, or similar "
            "structural surface."
        )

        st.caption(
            f"Input classified as: {surface_check['top_label']} "
            f"({surface_check['top_score'] * 100:.1f}%)"
        )

        st.stop()

    # ========================================================
    # VALID STRUCTURAL IMAGE - CONTINUE
    # ========================================================

    st.markdown(
        "### Inspection Image"
    )

    st.image(
        image,
        use_container_width=True,
    )

    st.write("")

    start = st.button(
        "START AI CRACK INSPECTION",
        type="primary",
        use_container_width=True,
    )

    # ========================================================
    # START INSPECTION
    # ========================================================

    if start:

        # Scanner
        scanner_placeholder = st.empty()

        scanner_placeholder.markdown(
    		textwrap.dedent(
        		"""
        		<div class="scanner">
            			<div class="scanner-line"></div>

            			<div class="scanner-center">
                			<div class="scanner-icon">AI</div>

                			<div class="scanner-text">
                    				AI SCANNING STRUCTURAL SURFACE
                			</div>
            			</div>
        		</div>
        		"""
    		),
    		unsafe_allow_html=True,
		)

        st.write("")

        progress = st.progress(
            0,
            text="Initializing StructuralCrackAI...",
        )

        status = st.empty()

        # ----------------------------------------------------
        # Stage 1
        # ----------------------------------------------------

        progress.progress(
            10,
            text="Initializing inspection system...",
        )

        status.info(
            "AI Initializing AI inspection system..."
        )

        time.sleep(0.7)

        # ----------------------------------------------------
        # Stage 2
        # ----------------------------------------------------

        progress.progress(
            25,
            text="Loading YOLO segmentation model...",
        )

        status.info(
            "Loading trained YOLO11 segmentation model..."
        )

        try:

            model = load_yolo()

        except Exception as e:

            scanner_placeholder.empty()
            st.error(
                f"Could not load YOLO model: {e}"
            )
            st.stop()

        # ----------------------------------------------------
        # Stage 3
        # ----------------------------------------------------

        progress.progress(
            40,
            text="Scanning structural surface...",
        )

        status.info(
            "Scanning structural surface for crack patterns..."
        )

        prediction = model.predict(
            source=np.array(image),
            conf=CONFIDENCE_THRESHOLD,
            imgsz=IMAGE_SIZE,
            verbose=False,
        )

        result = prediction[0]

        time.sleep(0.5)

        # ----------------------------------------------------
        # Stage 4
        # ----------------------------------------------------

        progress.progress(
            55,
            text="Running crack segmentation...",
        )

        status.info(
            "Segmenting detected crack regions..."
        )

        time.sleep(0.5)

        # ----------------------------------------------------
        # No crack
        # ----------------------------------------------------

        if (
            result.masks is None
            or len(result.masks.data) == 0
        ):

            progress.progress(
                100,
                text="Inspection complete.",
            )

            scanner_placeholder.empty()

            status.success(
                "Inspection complete - no structural cracks detected."
            )

            st.divider()

            st.success(
                "### NO CRACK DETECTED"
            )

            st.image(
                result.plot(),
                use_container_width=True,
            )

            st.stop()

        # ----------------------------------------------------
        # Stage 5 - Depth
        # ----------------------------------------------------

        progress.progress(
            70,
            text="Generating relative depth map...",
        )

        status.info(
            "Analyzing relative surface depth..."
        )

        try:

            depth_pipe = load_depth_model()

            depth_map = estimate_depth_map(
                image,
                depth_pipe,
            )

        except Exception as e:

            scanner_placeholder.empty()

            st.error(
                "Depth estimation failed."
            )

            st.exception(e)

            st.stop()

        # ----------------------------------------------------
        # Stage 6
        # ----------------------------------------------------

        progress.progress(
            82,
            text="Measuring crack geometry...",
        )

        status.info(
            "Measuring crack area, length and width..."
        )

        masks = (
            result.masks.data
            .cpu()
            .numpy()
        )

        confidences = (
            result.boxes.conf
            .cpu()
            .numpy()
        )

        reports = []

        for mask, confidence in zip(
            masks,
            confidences,
        ):

            report = analyze_single_crack(
                mask,
                float(confidence),
                depth_map,
                pixels_per_mm,
            )

            reports.append(report)

        # ----------------------------------------------------
        # Stage 7
        # ----------------------------------------------------

        progress.progress(
            94,
            text="Calculating structural risk...",
        )

        status.info(
            "Calculating severity and structural intelligence..."
        )

        time.sleep(0.5)

        # ----------------------------------------------------
        # COMPLETE
        # ----------------------------------------------------

        progress.progress(
            100,
            text="Inspection complete.",
        )

        scanner_placeholder.empty()

        status.success(
            "AI structural inspection completed."
        )

        time.sleep(0.4)

        # ====================================================
        # RESULT HEADER
        # ====================================================

        st.divider()

        st.header(
            "Inspection Results"
        )

        total_cracks = len(
            reports
        )

        st.success(
            f"CRACK DETECTED - {total_cracks} "
            f"region(s) identified."
        )

        # ====================================================
        # ORIGINAL / AI RESULT
        # ====================================================

        left, right = st.columns(2)

        with left:

            st.markdown(
                "### Original Image"
            )

            st.image(
                image,
                use_container_width=True,
            )

        with right:

            st.markdown(
                "### AI Segmentation"
            )

            annotated = result.plot()

            st.image(
                annotated,
                use_container_width=True,
            )

        # ====================================================
        # DEPTH MAP
        # ====================================================

        st.divider()

        st.markdown(
            "### Relative Surface Depth"
        )

        st.caption(
            "Relative depth is a unitless model-derived "
            "indicator. It is not physical crack depth in mm."
        )

        st.image(
            depth_map,
            use_container_width=True,
            clamp=True,
        )

        # ====================================================
        # SUMMARY
        # ====================================================

        st.divider()

        st.markdown(
            "## Inspection Summary"
        )

        summary1, summary2, summary3, summary4 = (
            st.columns(4)
        )

        max_confidence = max(
            r["confidence"]
            for r in reports
        )

        risk_levels = [
            r["risk_level"]
            for r in reports
        ]

        if "High" in risk_levels:
            overall_risk = "HIGH"

        elif "Moderate" in risk_levels:
            overall_risk = "MODERATE"

        else:
            overall_risk = "LOW"

        with summary1:
            st.metric(
                "Cracks Detected",
                total_cracks,
            )

        with summary2:
            st.metric(
                "Highest Confidence",
                f"{max_confidence * 100:.1f}%",
            )

        with summary3:
            st.metric(
                "Overall Risk",
                overall_risk,
            )

        with summary4:
            st.metric(
                "Calibration",
                f"{pixels_per_mm:.1f} px/mm"
                if pixels_per_mm
                else "N/A",
            )

        # ====================================================
        # INDIVIDUAL CRACKS
        # ====================================================

        st.divider()

        st.markdown(
            "## Crack-by-Crack Engineering Analysis"
        )

        for index, report in enumerate(
            reports,
            start=1,
        ):

            severity_class = (
                report["severity"]
                .lower()
                .replace(" ", "")
            )

            with st.expander(
                f"Crack #{index} - "
                f"{report['severity']} Severity",
                expanded=True,
            ):

                st.markdown(
    			textwrap.dedent(
        			f"""
        			<div class="result-card">
            				<div class="result-title">
                				Crack #{index}
            				</div>
        			</div>
        			"""
    			),
    			unsafe_allow_html=True,
		)

                # --------------------------------------------
                # PRIMARY METRICS
                # --------------------------------------------

                a, b, c, d = st.columns(4)

                with a:

                    st.metric(
                        "Confidence",
                        f"{report['confidence'] * 100:.1f}%",
                    )

                with b:

                    value = (
                        f"{report['area_mm2']:.2f} mm^2"
                        if report["area_mm2"]
                        is not None
                        else f"{report['area_pixels']} px^2"
                    )

                    st.metric(
                        "Crack Area",
                        value,
                    )

                with c:

                    value = (
                        f"{report['length_mm']:.2f} mm"
                        if report["length_mm"]
                        is not None
                        else f"{report['length_pixels']} px"
                    )

                    st.metric(
                        "Crack Length",
                        value,
                    )

                with d:

                    value = (
                        f"{report['maximum_width_mm']:.2f} mm"
                        if report[
                            "maximum_width_mm"
                        ] is not None
                        else f"{report['max_width_pixels']:.2f} px"
                    )

                    st.metric(
                        "Maximum Width",
                        value,
                    )

                # --------------------------------------------
                # GEOMETRY
                # --------------------------------------------

                st.markdown(
                    "#### Geometry"
                )

                g1, g2, g3, g4 = st.columns(4)

                with g1:

                    value = (
                        f"{report['average_width_mm']:.3f} mm"
                        if report[
                            "average_width_mm"
                        ] is not None
                        else f"{report['avg_width_pixels']:.2f} px"
                    )

                    st.metric(
                        "Average Width",
                        value,
                    )

                with g2:

                    st.metric(
                        "Area %",
                        f"{report['area_percentage']:.2f}%",
                    )

                with g3:

                    st.metric(
                        "Aspect Ratio",
                        report[
                            "aspect_ratio"
                        ],
                    )

                with g4:

                    st.metric(
                        "Compactness",
                        report[
                            "compactness"
                        ],
                    )

                # --------------------------------------------
                # DEPTH
                # --------------------------------------------

                st.markdown(
                    "#### Relative Depth"
                )

                d1, d2 = st.columns(2)

                with d1:

                    if (
                        report[
                            "relative_depth"
                        ] is not None
                    ):

                        st.metric(
                            "Mean Relative Depth",
                            f"{report['relative_depth']:.3f}",
                        )

                    else:

                        st.metric(
                            "Mean Relative Depth",
                            "N/A",
                        )

                with d2:

                    if (
                        report[
                            "depth_contrast"
                        ] is not None
                    ):

                        st.metric(
                            "Depth Contrast",
                            f"{report['depth_contrast']:.4f}",
                        )

                    else:

                        st.metric(
                            "Depth Contrast",
                            "N/A",
                        )

                # --------------------------------------------
                # STRUCTURAL INTELLIGENCE
                # --------------------------------------------

                st.markdown(
                    "#### Structural Intelligence"
                )

                s1, s2, s3, s4 = st.columns(4)

                with s1:

                    st.metric(
                        "Integrity Index",
                        f"{report['integrity_index']:.1f}/100",
                    )

                with s2:

                    st.metric(
                        "Risk Score",
                        report[
                            "risk_score"
                        ],
                    )

                with s3:

                    st.metric(
                        "Risk Level",
                        report[
                            "risk_level"
                        ],
                    )

                with s4:

                    st.metric(
                        "Priority",
                        report[
                            "priority"
                        ],
                    )

                # --------------------------------------------
                # MAINTENANCE
                # --------------------------------------------

                if report[
                    "risk_level"
                ] == "High":

                    st.error(
                        f"WARNING: {report['maintenance']}"
                    )

                elif report[
                    "risk_level"
                ] == "Moderate":

                    st.warning(
                        f"NOTICE: {report['maintenance']}"
                    )

                else:

                    st.success(
                        f"OK: {report['maintenance']}"
                    )

                st.info(
                    f"Severity classification: "
                    f"**{report['severity']}**"
                )

                # --------------------------------------------
                # ENGINEERING FEATURES
                # --------------------------------------------

                with st.expander(
                    "Advanced engineering features"
                ):

                    e1, e2, e3 = st.columns(3)

                    with e1:

                        st.metric(
                            "Skeleton Density",
                            report[
                                "skeleton_density"
                            ],
                        )

                    with e2:

                        st.metric(
                            "Aspect Ratio",
                            report[
                                "aspect_ratio"
                            ],
                        )

                    with e3:

                        st.metric(
                            "Compactness",
                            report[
                                "compactness"
                            ],
                        )

        # ====================================================
        # FOOTER
        # ====================================================

        st.divider()

        st.markdown(
    textwrap.dedent(
        """
            <div class="footer">
                StructuralCrackAI - YOLO11 Segmentation -
                Depth Anything V2 - Engineering Analysis
            </div>
            """
    ),
    unsafe_allow_html=True,
)

else:

    # ========================================================
    # IDLE STATE
    # ========================================================

    st.markdown("### READY FOR STRUCTURAL INSPECTION")

    st.caption(
        "Upload a structural/concrete surface image to begin AI analysis."
    )

    st.info(
        "Upload a JPG, PNG, BMP or WEBP structural image "
        "to begin the inspection."
    )
