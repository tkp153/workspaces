import copy
import openpifpaf_voc as pif

import cv2
from motpy import Detection,MultiObjectTracker
from motpy.testing_viz import draw_track

import numpy as np 

class Motpy:
    def __init__(self):
        self.tracker = MultiObjectTracker(dt = 0.1)
        
    def track(self,a,b,c):
        
        boxes = a
        scores = b
        labels = c
        #outputs = [Detection(box = box[:2],score = box[3],class_id=box[4]) for box in outputs]
        outputs = [Detection(box = b,score = s, class_id = l)
        for b,s,l in zip(boxes, scores,labels)]
        
        self.tracker.step(detections=outputs)
        
        tracks = self.tracker.active_tracks()
        return tracks
    
def main():
    videodata = "atc_motpy_1.mp4"
    fmt = cv2.VideoWriter_fourcc('m','p','4','v')
    fps = 30.0
    size = (1280,720)
    writer = cv2.VideoWriter(videodata,fmt,fps,size)
    
    mot = Motpy()
    cap = cv2.VideoCapture("atc1.webm")
    while True:
        ret_val,frame = cap.read()
        
        if ret_val:
            bbox,score,label = pif.openpifpaf_voc.normal_pub(ret_val, frame)
            result_frame = frame
            tracks = mot.track(bbox,score,label)
            
            for trc in tracks:
                draw_track(result_frame,trc,thickness=1)
                #print("tracksid: " + trc.id)
            #cv2.imshow('hoge',result_frame)
            writer.write(frame)
            ch = cv2.waitKey(1)
            if ch == 27 or ch == ord("q") or ch == ord("Q"):
                break
        else:
            break
    writer.release()
    cv2.destroyAllWindows()
        
if __name__ == '__main__':
    main()       