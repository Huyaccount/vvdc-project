# 🚀 YOLOv11 — Custom Object Detection

![Python](https://img.shields.io/badge/Python-3.9%2B-blue)
![YOLOv11](https://img.shields.io/badge/YOLO-v11-green)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-orange)
![Status](https://img.shields.io/badge/Status-Production%20Ready-success)

> **Summary:**
> This project implements and fine-tunes **YOLOv11** for custom object detection tasks (e.g., motorbikes, cars, helmets) in real-world environments such as Vietnamese traffic scenes.
> The goal is to achieve a strong balance between **accuracy**, **speed**, and **deployment readiness**.

---

## 📌 Table of Contents

* [Introduction](#introduction)
* [Key Features](#key-features)
* [System Requirements](#system-requirements)
* [Quick Installation](#quick-installation)
* [Dataset Structure](#dataset-structure)
* [Training](#training)
* [Inference](#inference)
* [Evaluation & Metrics](#evaluation--metrics)
* [Sample Results](#sample-results)
* [Optimization Tips](#optimization-tips)
* [Resources](#resources)
* [License](#license)

---

## Introduction

This repository provides a complete pipeline for:

* Training YOLOv11 on custom datasets.
* Evaluating model performance using standard object detection metrics.
* Running inference on images and videos.
* Exporting trained models for deployment.

The project follows best practices for **reproducibility**, **scalability**, and **real-world deployment**.

---

## Key Features

* Multi-class object detection with custom labels.
* GPU-accelerated training with mixed precision (FP16).
* Checkpoint saving and resume training support.
* Image and video inference.
* Export to ONNX / TensorRT (optional).
* Clean and extensible project structure.

---

## System Requirements

* **OS:** Linux / Windows / macOS (WSL2 recommended on Windows)
* **Python:** 3.9+
* **GPU:** NVIDIA GPU recommended (≥ 8GB VRAM)
* **PyTorch:** 2.0+
* **CUDA:** Compatible with installed PyTorch version

---

## Quick Installation

```bash
git clone https://github.com/Huyaccount/vvdc-project.git
cd vvdc-project

python -m venv .venv
source .venv/bin/activate      # Linux / macOS
.venv\\Scripts\\activate         # Windows

pip install -r requirements.txt
pip install ultralytics
```

---

## Dataset Structure

The dataset follows the **YOLO format**:

```
dataset/
├── images/
│   ├── train/
│   ├── val/
│   └── test/
├── labels/
│   ├── train/
│   ├── val/
│   └── test/
└── data.yaml
```

### data.yaml example

```yaml
train: dataset/images/train
val: dataset/images/val
test: dataset/images/test

nc: 7
names:
  - car
  - coach
  - bus
  - truck
  - pickup
  - small truck
  - license plate
```

### Label format

```
<class_id> <x_center> <y_center> <width> <height>
```

All values are normalized to `[0, 1]`.

---

## Training

```bash
ultralytics train \
  model=yolov11.pt \
  data=data.yaml \
  epochs=130 \
  imgsz=640 \
  batch=16 \
  device=0
```

Resume training:

```bash
ultralytics train \
  model=runs/detect/exp/weights/last.pt \
  data=data.yaml \
  epochs=100 \
  imgsz=640 \
  device=0 \
  --resume
```

---

## Inference

```bash
ultralytics predict \
  model=runs/detect/exp/weights/best.pt \
  source=assets/image.jpg \
  save=true
```

Video inference:

```bash
ultralytics predict \
  model=runs/detect/exp/weights/best.pt \
  source=assets/video.mp4 \
  conf=0.25 \
  save=true
```

---

## Evaluation & Metrics

```bash
ultralytics val \
  model=runs/detect/exp/weights/best.pt \
  data=data.yaml \
  imgsz=640
```

Metrics reported:

* mAP@0.5
* mAP@0.5:0.95
* Precision
* Recall
* FPS

---

## Sample Results

| Metric                | Value |
| --------------------- | ----- |
| mAP@0.5               | 0.87  |
| mAP@0.5:0.95          | 0.62  |
| Precision             | 0.85  |
| Recall                | 0.78  |
| FPS (RTX 3060, 640px) | ~25   |

---

## Optimization Tips

* Use data augmentation (Mosaic, MixUp, HSV jitter).
* Tune image size for speed vs accuracy trade-off.
* Enable FP16 mixed precision.
* Address class imbalance using weighted sampling or focal loss.
* Export to ONNX / TensorRT for deployment.

---

## Resources

* Dataset: [https://drive.google.com/drive/folders/1TSF8fh3Pum9QR7wzaII1DmhgaJNo4Tkt](https://drive.google.com/drive/folders/1TSF8fh3Pum9QR7wzaII1DmhgaJNo4Tkt)
* Google Colab: [https://colab.research.google.com/drive/1x4GpRugw0A0o7CIeW3ZILrIDjKD4PpkS](https://colab.research.google.com/drive/1x4GpRugw0A0o7CIeW3ZILrIDjKD4PpkS)

---

## License

Released under the **MIT License**.

Please credit YOLO / Ultralytics and dataset authors if applicable.

---

⭐ If this repository helps you, consider giving it a star!
