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

For single polygon region uisng YOLO v8
```bash
python yolov8SinglePolygon.py -i <input video path> -o <output video path>
```
For multiple polygon regions uisng YOLO v5 
```bash
python yolov5MultiplePolygon.py -i <input video path> -o <output video path>
```
For multiple polygon regions uisng YOLO v8 (Recommended for multiple region)
```bash
python yolov8MultiplePolygon.py -i <input video path> -o <output video path>
```

### Python
For single polygon region uisng YOLO v8
```python
from yolov8SinglePolygon import CountObject
obj = CountObject(<input_file_path>,<output_file_path>)
obj.process_video()
```

For multiple polygon regions uisng YOLO v5
```python
from yolov5MultiplePolygon import CountObject
obj = CountObject(<input_file_path>,<output_file_path>)
obj.process_video()
```

For multiple polygon regions uisng YOLO v8 (Recommended for multiple region)
```python
from yolov8MultiplePolygon import CountObject
obj = CountObject(<input_file_path>,<output_file_path>)
obj.process_video()
```

https://github.com/sankalpvarshney/Detect-and-count-objects-in-polygon-zone/assets/41926323/0ccf1c8f-d1a1-4935-ba19-f349dca88a82

## Contributing

Pull requests are welcome. For major changes, please open an issue first
to discuss what you would like to change.

Please make sure to update tests as appropriate.
