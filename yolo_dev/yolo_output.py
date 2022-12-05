import argparse
import os
import cv2
import numpy as np
import pandas as pd
import onnxruntime

from yolox.data.data_augment import preproc as preprocess
from yolox.data.datasets import COCO_CLASSES
from yolox.utils import mkdir,multiclass_nms,demo_postprocess,vis

output_dir ='onnx_out'
image_path = '/root/workspaces/2_bus_101_top-thumb-720x503-10340.jpg'
model = '/root/workspaces/YOLOX/yolox_x.onnx'

input_shape =(640,640)
origin_img = cv2.imread(image_path)
img,ratio = preprocess(origin_img,input_shape)
session = onnxruntime.InferenceSession(model)
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
    origin_img = vis(origin_img, final_boxes,       final_scores, final_cls_inds,
    0.3, class_names=COCO_CLASSES)

mkdir(output_dir)
output_path = os.path.join(output_dir, os.path.basename(image_path))

cv2.imwrite(output_path, origin_img)
cv2.imshow("hoge",origin_img)
result = []
[result.extend((final_cls_inds[x],COCO_CLASSES[int(final_cls_inds[x])],final_scores[x],final_boxes[x][0],final_boxes[x][1],final_boxes[x][2],final_boxes[x][3]) for x in range(len(final_scores)))]
df = pd.DataFrame(result, columns = ['class-id','class','score','x-min','y-min','x-max','y-max'])
print(df)