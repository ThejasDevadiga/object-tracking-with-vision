# Object Tracking with Vision on YOLOv7

Repository template and guide for building an **object detection + multi-object tracking (MOT)** system that uses **YOLOv7** for detection and a separate tracker (DeepSORT / BYTE / SORT) for consistent IDs across frames. This README assumes a repository named `object-tracking-with-vision-yolov7` that combines YOLOv7 detection with a tracker and utilities for dataset preparation, training, inference and evaluation.

---


## High-level approach

1. **Detect** objects in each frame using YOLOv7 (fast, high-quality detection).
2. **Extract features** (appearance embeddings) for detections (DeepSORT / ReID model) or use motion-only association.
3. **Associate** detections across frames with a tracker (DeepSORT, ByteTrack, SORT) to produce stable track IDs.
4. **Evaluate** with MOT metrics (MOTA, IDF1, IDs, FP, FN).

---

## Features

* YOLOv7 detection integration (training & inference)
* Tracker options: **DeepSORT**, **ByteTrack**, **SORT**
* Support for MOT challenge format and COCO-style datasets
* Visualization and video output (annotated bounding boxes + track IDs)
* Scripts for dataset conversion and evaluation using standard MOT metrics
* Optional feature-extractor (ReID) for stronger identity matching

---

## Requirements

Recommended: Linux + NVIDIA GPU

Install dependencies:

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Example `requirements.txt` (non-exhaustive):

```
torch>=1.12
torchvision
opencv-python
numpy
matplotlib
scipy
lap         # linear assignment
filterpy    # for Kalman filter-based trackers
scikit-image
cython
motmetrics
tqdm
```



## Inference / Tracking (examples)



Webcam (real-time):

```bash
python track.py --weights weights/yolov7.pt --source 0 --img 640 --tracker bytetrack
```

Image folder:

```bash
python track.py --weights weights/yolov7.pt --source datasets/mot/seq1/img1/ --tracker deep_sort --save-txt
```

Outputs: annotated video/images in `runs/track/exp1`, plus `.txt` files with per-frame detection + track ID in MOT TXT format.

---

## Tracking script behavior

`yolov7/track.py` in this repo wraps:

* Run detection (YOLOv7) per frame.
* Optionally crop & extract appearance embeddings from some ReID model (for DeepSORT).
* Pass detections (x1,y1,x2,y2,score,embedding) to tracker.
* Tracker returns track objects with unique IDs.
* Visualization: draw bbox + ID + trail, and optionally save detection/tracking results in MOT TXT format.



## Tips & Best Practices

* Use a strong detector (higher recall) if your tracker expects high recall (ByteTrack).
* For identity preservation under occlusion, use DeepSORT with a trained ReID model.
* Tune association thresholds (cosine distance threshold, gating thresholds) per dataset.
* Precompute embeddings for speed if CPU is a bottleneck, or use a lightweight ReID model on GPU.
* Use appropriate input resolution — higher resolution improves accuracy but lowers FPS.

---

## Conversion to ONNX / TensorRT (optional)

* Export YOLOv7 to ONNX for inference speed or to convert to TensorRT.
* Test exported model extensively — small ops or dynamic shapes may need fixes.





