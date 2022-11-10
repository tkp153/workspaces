import cv2
from openpifpaf.predictor import Predictor
import numpy as np
from pybboxes import BoundingBox

class openpifpaf_voc(object):
    def __init__(self):
        
        self.mode = "camera"
    
    def voc_pub(self,frame):
        
        predictor = Predictor()
        pred, _, meta = predictor.numpy_image(frame)
        
        poses = []
        
        for p in pred:
            pose = p.json_data()
            poses.append(pose)
            
        pt =[]
        bbox =[]
        score =[]
        label = []
        datas = []
        for p in poses:
            my_coco_box = p['bbox']
            coco_box = np.array(my_coco_box)
            scores = p['score']
            labels = p['category_id'] = 1
            keypoints = p['keypoints']
            
            data = np.array([xywh_to_xyxy(coco_box[0],coco_box[1],coco_box[2],coco_box[3]),scores, labels])
            bbox.append(xywh_to_xyxy(coco_box[0],coco_box[1],coco_box[2],coco_box[3]))
            score.append(scores)
            label.append(labels)
            pt.append(keypoints)
            datas.append(data)
            
            #data = np.append(bbox,score)
            #data = np.append(data,label)
            
        #print(f"bbox :{bbox} \n keypoints:{pt} \n ")
        return datas
        
        
    def normal_pub(self, frame):
        
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
                coco_box = xywh_to_xyxy(my_coco_box)
                scores = p['score']
                labels = p['category_id']
        
                bbox.append(coco_box)
                score.append(scores)
                label.append(labels)
        
            return bbox,score,label
        
def xywh_to_xyxy(data):
    """convert xywh format to xyxy format"""
    x1 = data[0]
    y1 = data[1]
    w  = data[2]
    h  = data[3]
        
    x2 = x1 + w
    y2 = y1 + h
    voc = np.array([x1,y1,x2,y2])
    return  voc         