import argparse
from ultralytics import YOLO
import numpy as np
import supervision as sv


parser = argparse.ArgumentParser(
                    prog='yolov8',
                    description='This program help to detect and count the person in the polygon region',
                    epilog='Text at the bottom of help')

parser.add_argument('-i', '--input',required=True)      # option that takes a value
parser.add_argument('-o', '--output',required=True)

args = parser.parse_args()

class CountObject():

    def __init__(self,input_video_path,output_video_path) -> None:
        
        self.model = YOLO('yolov8s.pt')
        # initiate polygon zone
        self.polygon = np.array([
            [200, 3840],
            [1300, 600],
            [1325, 600],
            [550, 3840]
        ])

        self.input_video_path = input_video_path
        self.output_video_path = output_video_path

        self.video_info = sv.VideoInfo.from_video_path(input_video_path)
        self.zone = sv.PolygonZone(polygon=self.polygon, frame_resolution_wh=self.video_info.resolution_wh)

        # initiate annotators
        self.box_annotator = sv.BoxAnnotator(thickness=4, text_thickness=4, text_scale=2)
        self.zone_annotator = sv.PolygonZoneAnnotator(zone=self.zone, color=sv.Color.white(), thickness=6, text_thickness=6, text_scale=4)

    def process_frame(self,frame: np.ndarray, _) -> np.ndarray:
        # detect
        results = self.model(frame, imgsz=1280)[0]
        detections = sv.Detections.from_yolov8(results)
        detections = detections[detections.class_id == 0]
        self.zone.trigger(detections=detections)

        # annotate
        box_annotator = sv.BoxAnnotator(thickness=4, text_thickness=4, text_scale=2)
        labels = [f"{self.model.names[class_id]} {confidence:0.2f}" for _, confidence, class_id, _ in detections]
        frame = box_annotator.annotate(scene=frame, detections=detections, labels=labels)
        frame = self.zone_annotator.annotate(scene=frame)

        return frame
    
    def process_video(self):

        sv.process_video(source_path=self.input_video_path, target_path=self.output_video_path, callback=self.process_frame)


if __name__ == "__main__":

    obj = CountObject(args.input,args.output)
    obj.process_video()