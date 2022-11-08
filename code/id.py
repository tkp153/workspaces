import cv2
from openpifpaf.predictor import Predictor
from motpy import Detection,MultiObjectTracker

import numpy as np

def main():
    cap = cv2.VideoCapture(0)

    predictor = Predictor()
    while True:
        ret,frame = cap.read()
        
        pred, _, meta = predictor.numpy_image(frame)
        
        poses = []
        
        for p in pred:
            pose = p.json_data()
            bbox = pose['bbox']
            score = pose['score']
            #print(bbox,score)
            sort_bbox = transform_list(bbox)
            sort_score = score
            #sort_data = np.append(sort_bbox,sort_score)
            poses.append(sort_bbox)
            
        input = np.array(poses,dtype =float)
        motpy_engine(input)
        
def transform_list(bbox):
    """
    x1 = bbox[0]
    y1 = bbox[1]
    x2 = bbox[0] + bbox[2]
    y2 = bbox[1] - bbox[3]
    """
    xHigh = bbox[0] + bbox[2]
    yHigh = bbox[1]
    xLow = bbox[0]
    yLow = bbox[1] -bbox[3]
    
    data = np.array([xLow,yLow,xHigh,yHigh])
    
    return data
    
def motpy_engine(data):
    # 物体検出のデータを次のフォーマットで用意：[xmin, ymin, xmax, ymax]
    
    object_box = data
    
    tracker = MultiObjectTracker(dt= 0.1)
    
    for step in range(len(object_box)):
        # フレームのbboxを更新する
        # ここではobject_boxは1フレームの[xmin, ymin, xmax, ymax]のデータが格納されたlistとする
        
        tracker.step(detections=[Detection(box = object_box)])
        
        # アクティブな追跡物体を取得する
        tracks = tracker.active_tracks()
        
        # trackはID、bbox、検出スコアの情報を持つ
        print('MOT tracker tracks %d objects' % len(tracks))
        print('first track box: %s' % str(tracks[0].box))
            
        
    
if __name__ == "__main__":
    main()        
            
        
        