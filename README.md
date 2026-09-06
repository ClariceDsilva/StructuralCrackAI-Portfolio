# StructuralCrackAI

AI-powered structural crack detection, segmentation, measurement, and risk-assessment system for concrete and structural surfaces.

StructuralCrackAI combines deep-learning-based crack segmentation with computer-vision geometry analysis, relative depth estimation, severity classification, and structural risk prioritization through an interactive Streamlit application.

### 🌐 Live Application

▶️ **[Open the Live StructuralCrackAI Application](https://structuralcrackai-i6rzijd9frukh4ydanrs3.streamlit.app/)**

The live application provides the complete AI-assisted structural crack inspection workflow, including structural-image validation, crack segmentation, geometry analysis, relative depth analysis, severity assessment, and structural risk prioritization.

---

## 🎥 Working Application Demo

Watch the recorded end-to-end StructuralCrackAI application demonstration.

The demo shows the working inspection pipeline, including:

- Structural surface image input
- AI-based crack detection and segmentation
- Crack geometry analysis
- Crack length and width estimation
- Relative depth analysis
- Severity assessment
- Crack-by-crack engineering analysis
- Structural risk prioritization

▶️ **[Watch the Working Application Demo on YouTube](https://youtu.be/y-S6DXDNoPI)**

> The production model weights, calibration files, datasets, and research notebooks are kept private because this project is currently undergoing patent-related protection.

---

## Overview

Structural inspections can involve large numbers of images and manual identification of cracks.

StructuralCrackAI was developed to investigate how computer vision and deep learning can assist this process by:

- Detecting cracks on structural/concrete surfaces
- Segmenting detected crack regions
- Measuring crack geometry
- Estimating real-world dimensions when calibration is available
- Generating relative surface-depth information
- Classifying crack severity
- Producing a rule-based structural risk assessment
- Providing an interactive visual inspection interface

The system is intended as an AI-assisted inspection and research tool rather than a replacement for professional structural engineering assessment.

---

## Key Features

### AI Crack Detection and Segmentation

The main detection system uses a YOLO11n segmentation model trained for structural crack detection.

The model produces:

- Crack detections
- Confidence scores
- Pixel-level segmentation masks

The segmentation mask is then used by the downstream engineering-analysis pipeline.

### Structural Surface Input Validation

The application includes an input-validation stage before running the crack detector.

The purpose of this stage is to restrict analysis to appropriate structural-surface imagery and reject clearly unrelated inputs such as:

- People
- Animals
- Food
- Documents/screenshots
- Electronic devices
- Other unrelated scenes

This prevents the crack-detection model from being blindly applied to arbitrary images.

---

## Engineering Analysis

After crack segmentation, the application extracts geometric and structural features from the detected crack mask.

### Crack Area

The segmented crack pixels are counted to estimate the crack area.

When calibration information is available, the pixel measurement is converted into an approximate physical area.

### Crack Length

The segmentation mask is skeletonized using image-processing techniques.

The number of skeleton pixels provides an estimate of crack length.

### Crack Width

Distance-transform analysis is used to estimate:

- Average crack width
- Maximum crack width

### Additional Geometry

The system also calculates:

- Area percentage
- Aspect ratio
- Compactness
- Skeleton density

These features provide additional information about the geometry and morphology of the detected crack.

---

## Calibration

The application supports pixel-to-real-world conversion through a calibration value stored in:

`calibration.json`

This allows measurements such as:

- Area → mm²
- Length → mm
- Average width → mm
- Maximum width → mm

Without calibration, the system reports measurements in pixels.

> The calibration file is intentionally not included in this public repository.

---

## Relative Depth Analysis

StructuralCrackAI integrates Depth Anything V2 to generate a relative depth map from the input image.

The depth component provides model-derived relative surface-depth information that can be compared around detected crack regions.

### Important

The depth output is a relative, unitless indicator.

It is **NOT** a direct physical measurement of crack depth in millimetres.

---

## Severity Classification

The application applies a rule-based severity classification using detected crack characteristics.

The classification considers factors including:

- Maximum crack width
- Crack length
- Crack area
- Detection confidence

The current categories are:

- Low
- Moderate
- High

This provides a consistent automated classification layer on top of the computer-vision measurements.

---

## Structural Intelligence

The system also calculates a rule-based structural intelligence score.

The analysis incorporates:

- Detection confidence
- Crack area
- Crack length
- Average width
- Maximum width
- Relative depth information

The resulting output includes:

- Integrity Index
- Risk Score
- Risk Level
- Maintenance recommendation
- Inspection priority

The risk assessment is an AI-assisted/rule-based prioritization mechanism and should not be interpreted as a certified structural-safety assessment.

---

## System Workflow

```text
                    INPUT IMAGE
                         |
                         v
             STRUCTURAL SURFACE
                 INPUT VALIDATION
                         |
                         v
                  YOLO11n-Seg
              Crack Detection +
                 Segmentation
                         |
                         v
                   CRACK MASK
                         |
          +--------------+--------------+
          |              |              |
          v              v              v
        AREA          LENGTH          WIDTH
          |              |              |
          +--------------+--------------+
                         |
                         v
               GEOMETRIC FEATURES
                         |
                         v
                DEPTH ANYTHING V2
                         |
                         v
                RELATIVE DEPTH
                         |
                         v
              SEVERITY CLASSIFICATION
                         |
                         v
             STRUCTURAL RISK ANALYSIS
                         |
                         v
               STREAMLIT INSPECTION
                    DASHBOARD
