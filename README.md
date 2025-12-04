# Cow & Buffalo Detector — YOLOv8

An advanced **computer vision project** for detecting **cows** and **buffaloes** using YOLOv8 — trained on a custom Kaggle dataset in Google Colab.

> Achieved **mAP50 = 0.98** and **mAP50-95 = 0.84** over 30 epochs on a dataset of ~2000 labeled images.

---

## Overview
This project demonstrates an end-to-end deep learning pipeline for object detection:
- Download dataset from Kaggle  
- Preprocess and split into `train` / `valid` sets  
- Train YOLOv8 model (`yolov8n.pt` pretrained weights)  
- Evaluate metrics (Precision, Recall, mAP, Confusion Matrix)  
- Export and deploy trained model (`best.pt`)

---

## Results
| Metric | Score |
|--------|--------|
| Precision | **0.97** |
| Recall | **0.94** |
| mAP@50 | **0.989** |
| mAP@50–95 | **0.840** |

### Visual Results:
| Detection Example | Confusion Matrix |
|--------------------|------------------|
| ![Detection Result](runs/detect/cow_buffalo_detector/results.png) | ![Confusion Matrix](runs/detect/cow_buffalo_detector/confusion_matrix_normalized.png) |

---

## Tech Stack
- **YOLOv8 (Ultralytics)**
- **PyTorch**
- **Google Colab**
- **Kaggle API**
- **OpenCV / Matplotlib**

---

## How to Run
### Install dependencies
```bash
pip install ultralytics opencv-python kaggle
