import argparse
import os
import cv2
import numpy as np
import pandas as pd
import onnxruntime
from tqdm import tqdm

from yolox.data.data_augment import preproc as preprocess
from yolox.data.datasets import COCO_CLASSES
from yolox.utils import mkdir,multiclass_nms,demo_postprocess, vis

model = '/root/workspaces/YOLOX/yolox_s.onnx'

cap = cv2.VideoCapture("/root/atc.mp4")
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
frame_video = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))


with tqdm(range(frame_video)):
    while(cap.isOpened()):
        ret_val,frame = cap.read()
    
        if ret_val:
    
            origin_img = frame
            input_shape =(640, 640)
            img,ratio = preprocess(origin_img,input_shape)
            provider = ['CUDAExecutionProvider','CPUExecutionProvider']
            session = onnxruntime.InferenceSession('/root/workspaces/YOLOX/yolox_s.onnx', providers=provider)
            ort_inputs = {session.get_inputs()[0].name: img[None,:,:,:]}
            output = session.run(None,ort_inputs)
            predictions = demo_postprocess(output[0], input_shape)[0]
            boxes = predictions[:, :4]
            scores = predictions[:, 4:5] * predictions[:, 5:]
            boxes_xyxy = np.ones_like(boxes)
            boxes_xyxy[:, 0] = boxes[:, 0] - boxes[:, 2]/2.
            boxes_xyxy[:, 1] = boxes[:, 1] - boxes[:, 3]/2.
            boxes_xyxy[:, 2] = boxes[:, 0] + boxes[:, 2]/2.
            boxes_xyxy[:, 3] = boxes[:, 1] + boxes[:, 3]/2.
            boxes_xyxy /= ratio
            dets = multiclass_nms(boxes_xyxy, scores, nms_thr=0.45, score_thr=0.5)
        
            if dets is not None:
                final_boxes, final_scores, final_cls_inds = dets[:, :4], dets[:, 4], dets[:, 5]
                #origin_img = vis(origin_img, final_boxes,       final_scores, final_cls_inds,
                #0.3, class_names=COCO_CLASSES)
                #print(final_boxes)
                #行数カウント
                count_Y = final_boxes.shape[0]
                count_X = final_boxes.shape[1]
                #print(count_X)
            
                data_x_min = []
                data_y_min = []
                data_x_max = []
                data_y_max = []
                i = 0
                for i in range(count_Y):
                    data_x_min.append(final_boxes[i][0])
                    data_y_min.append(final_boxes[i][1])
                    data_x_max.append(final_boxes[i][2])
                    data_y_max.append(final_boxes[i][3])
                # データ分析
                Xmin = min(data_x_min)
                Ymin = min(data_y_min)
                Xmax = max(data_x_max)
                Ymax = max(data_y_max)
            
                if(Xmin - 20 > 0):
                    Xmin -= 20
                else:
                    Xmin = 0
                if(Ymin - 20 > 0):
                    Ymin -= 20
                else:
                    Ymin = 0
                if(Xmax + 20 < width):
                    Xmax += 20
                else:
                    Xmax = width
                if(Ymax + 20 < height):
                    Ymax += 20
                else:
                    Ymax = height
                
                #print(Xmin,Ymin,Xmax,Ymax)
                cut_image = frame[int(Ymin):int(Ymax),int(Xmin):int(Xmax)]
                #cv2.imshow('data',cut_image)
                #if cv2.waitKey(1) & 0xFF == ord('q'):
                    #break
        else:
            break
                    