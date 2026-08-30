# StructuralCrackAI - Project Roadmap

## Vision
Build an AI-powered industrial structural crack inspection software system capable of real-time crack detection, measurement, and risk assessment from images, videos, and live camera streams.

---

# Version 1 (Current Goal)

## Input
- Image
- Video
- Live webcam stream

## Output
- Detect crack(s).
- Display crack mask on screen.
- Display confidence score.
- Support real-time inference.

## V1 Scope
- YOLOv11-Seg model.
- Image inference.
- Video inference.
- Live webcam inference.
- Basic visualization.
- Use YOLOv11-Seg as the core segmentation model.

## Not Included in V1
- Robot hardware integration.
- Crack width measurement.
- Crack length measurement.
- Depth estimation.
- Severity prediction.
- Survival/risk prediction.
- Mobile application.

---

# Future Versions

## V2
- Crack width calculation.
- Crack length calculation.
- OpenCV geometry module.

## V3
- Severity classification.
- Structural risk scoring.

## V4
- Depth estimation.
- Robot hardware integration.
- Edge AI deployment.

---

## Long-Term Goal

Develop a deployable AI software platform for autonomous industrial structural inspection robots.
