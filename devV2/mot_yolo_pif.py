import argparse
import onnxruntime as ort
import cv2
import time
import numpy as np
from tqdm import tqdm
from yolox.utils import  multiclass_nms, demo_postprocess
from yolox.data.data_augment import preproc as preprocess
from yolox.utils import vis
from yolox.data.datasets import COCO_CLASSES
from motpy import Detection,MultiObjectTracker
from motpy.testing_viz import draw_track
from openpifpaf.predictor import Predictor


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

def load_session(args):
    print(args.model)

    
    provider = ['CUDAExecutionProvider','CPUExecutionProvider']
    session = ort.InferenceSession('/root/YOLOX/yolox_s.onnx', providers=provider)
    print(session.get_providers())

    if args.model in ["yolox_tiny", "yolox_nano"]:
        input_size = (416, 416)
    else:
        input_size = (640, 640)
        print(type(input_size))
    return session, input_size

def main(args):
    session, input_size = load_session(args)
    cap = cv2.VideoCapture("/root/atc1.mp4")
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_video = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    yolo_count = 0
    videodata = "atc_motpy_1_mix.mp4"
    ct = 0
    cou = 0
    output_folder ="/root/workspaces/output_pic/frame%d.jpg"
    mot = Motpy()
    predictor = Predictor()

    start_time = time.time()

    while(cap.isOpened()):
        success, frame = cap.read()
            #print(type(frame.dtype))
        if not success:
            print("Error reading or finish")
            break
        img, ratio = preprocess(frame, input_size)
        #print(img.dtype)
            

        ort_inputs = {session.get_inputs()[0].name: img[None, :, :, :]}

        output = session.run(None, ort_inputs)
        predictions = demo_postprocess(output[0], input_size, p6=False)[0]
        #print(type(predictions))
        boxes = predictions[:, :4]
        scores = predictions[:, 4:5] * predictions[:, 5:]

        boxes_xyxy = np.ones_like(boxes)
        boxes_xyxy[:, 0] = boxes[:, 0] - boxes[:, 2]/2.
        boxes_xyxy[:, 1] = boxes[:, 1] - boxes[:, 3]/2.
        boxes_xyxy[:, 2] = boxes[:, 0] + boxes[:, 2]/2.
        boxes_xyxy[:, 3] = boxes[:, 1] + boxes[:, 3]/2.
        boxes_xyxy /= ratio
        dets = multiclass_nms(boxes_xyxy, scores, nms_thr=0.45, score_thr=0.1)
        yolo_count += 1
        '''
        トリミング処理
        '''
            
        if dets is not None:
            final_boxes, final_scores, final_cls_inds = dets[:, :4], dets[:, 4], dets[:, 5]
                
            #行数カウント
            Count_XAxis = final_boxes.shape[1]
            Count_YAxis = final_boxes.shape[0]
                
            #データキャッシュ
            data_x_min = []
            data_y_min = []
            data_x_max = []
            data_y_max = []
            i = 0
            people_count = 0
            for i in range(Count_YAxis):
                if final_cls_inds[i] == 0.0:
                    data_x_min.append(final_boxes[i][0])
                    data_y_min.append(final_boxes[i][1])
                    data_x_max.append(final_boxes[i][2])
                    data_y_max.append(final_boxes[i][3])
                    people_count += 1
                
            if(people_count > 0):
                #データ分析
                Xmin = min(data_x_min)
                Ymin = min(data_y_min)
                Xmax = max(data_x_max)
                Ymax = max(data_y_max)
                

            #　画像拡大処理
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
            #画像トリミング処理
            cut_image = frame[int(Ymin):int(Ymax),int(Xmin):int(Xmax)]
            
            '''
            Motpy Engine
            '''
            if (ct % 4 == 0):
                bbox,score,label = normal_pub(cut_image,Xmin,Ymin)
                det,motpy_frame = cap.read()
                tracks = mot.track(bbox,score,label)

                for trc in tracks:
                    print(trc.id[:5])
                    #draw_track(motpy_frame,trc,thickness=1)
                #cv2.imwrite(output_folder % cou , motpy_frame)
                cou += 1
            ct += 1


    elapsed = time.time() - start_time
    print(elapsed)
    cap.release()

def normal_pub( frame,xmin,ymin):
        
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
                coco_box = xywh_to_xyxy(my_coco_box,xmin,ymin)
                scores = p['score']
                labels = p['category_id']
        
                bbox.append(coco_box)
                score.append(scores)
                label.append(labels)
        
            return bbox,score,label
        
def xywh_to_xyxy(data,xmin,ymin):
    """convert xywh format to xyxy format"""
    x1 = data[0] + xmin
    y1 = data[1] + ymin
    w  = data[2]
    h  = data[3]
        
    x2 = x1 + w
    y2 = y1 + h
    voc = np.array([x1,y1,x2,y2])
    return  voc             

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="")
    args = parser.parse_args()

    main(args)
