# Object Detection and Count in polygon zone

This repository contains multiple Python files for object detection and object counting in different regions of a frame, utilizing various versions of YOLO (You Only Look Once) algorithms.

## Installation

```bash
git clone https://github.com/sankalpvarshney/Detect-and-count-objects-in-polygon-zone.git
cd Detect-and-count-objects-in-polygon-zone
conda create --prefix ./env python=3.8 -y
pip install -r requirements.txt
```

## Usage

### Linux

For Single polygon region uisng YOLO v8
```bash
python yolov8SinglePolygon.py -i <input video path> -o <output video path>
```