import cv2
from openpifpaf.predictor import Predictor
import numpy as np
from pybboxes import BoundingBox

class openpifpaf_voc(object):
    def __init__(self):
        
        self.mode = "camera"
        
        
        if(self.mode == "camera"):
            self.cap = cv2.VideoCapture(0)
        elif(self.mode == "video"):
            self.cap = cv2.VideoCapture('person.mp4')
    
    def voc_pub(self,frame):
        
        predictor = Predictor()
        pred, _, meta = predictor.numpy_image(frame)
        
        poses = []
        
        for p in pred:
            pose = p.json_data()
            poses.append(pose)
            
        bbox =[]
        score =[]
        label = []
        for p in poses:
            my_coco_box = p['bbox']
            scores = p['score']
            labels = p['category_id']
        
            coco_bbox = BoundingBox.from_coco(*my_coco_box)
            voc_bbox = coco_bbox.to_voc(return_values=True)
            bbox.append(voc_bbox)
            score.append(scores)
            label.append(labels)
        
        return bbox,score,label
        