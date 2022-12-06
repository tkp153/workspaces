import argparse
import onnxruntime as ort
import cv2
import time
import numpy as np
from tqdm import tqdm
from yolox_utils import preproc, multiclass_nms, demo_postprocess
from yolox.utils import vis
from yolox.data.datasets import COCO_CLASSES

def load_session(args):
    print(args.model)

    try:
        provider = ['CUDAExecutionProvider','CPUExecutionProvider']
        session = ort.InferenceSession('/root/workspaces/YOLOX/yolox_x.onnx', providers=provider)
    except:
        provider = ['CPUExecutionProvider']
        session = ort.InferenceSession(f'/root/workspaces/YOLOX/{args.model}.onnx', providers=provider)
    print(session.get_providers())

    if args.model in ["yolox_tiny", "yolox_nano"]:
        input_size = (416, 416)
    else:
        input_size = (640, 640)
    return session, input_size

def main(args):
    session, input_size = load_session(args)
    cap = cv2.VideoCapture("/root/atc.mp4")
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_video = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    start_time = time.time()

    with tqdm(range(frame_video)) as pbar:
        while(cap.isOpened()):
            success, frame = cap.read()
            if not success:
                print("Error reading or finish")
                break
            img, ratio = preproc(frame, input_size)

            ort_inputs = {session.get_inputs()[0].name: img[None, :, :, :]}
            output = session.run(None, ort_inputs)
            predictions = demo_postprocess(output[0], input_size, p6=False)[0]

            boxes = predictions[:, :4]
            scores = predictions[:, 4:5] * predictions[:, 5:]

            boxes_xyxy = np.ones_like(boxes)
            boxes_xyxy[:, 0] = boxes[:, 0] - boxes[:, 2]/2.
            boxes_xyxy[:, 1] = boxes[:, 1] - boxes[:, 3]/2.
            boxes_xyxy[:, 2] = boxes[:, 0] + boxes[:, 2]/2.
            boxes_xyxy[:, 3] = boxes[:, 1] + boxes[:, 3]/2.
            boxes_xyxy /= ratio
            dets = multiclass_nms(boxes_xyxy, scores, nms_thr=0.45, score_thr=0.1)
            
            if dets is not None:
                final_boxes, final_scores, final_cls_inds = dets[:, :4], dets[:, 4], dets[:, 5]
                origin_img = vis(frame, final_boxes,       final_scores, final_cls_inds,0.3,class_names=COCO_CLASSES)
                
                #行数カウント
                Count_XAxis = final_boxes.shape[1]
                Count_YAxis = final_boxes.shape[0]
                
                #データキャッシュ
                data_x_min = []
                data_y_min = []
                data_x_max = []
                data_y_max = []
                i = 0
                for i in range(Count_YAxis):
                    data_x_min.append(final_boxes[i][0])
                    data_y_min.append(final_boxes[i][1])
                    data_x_max.append(final_boxes[i][2])
                    data_y_max.append(final_boxes[i][3])
                    
                # データ分析
                Xmin = min(data_x_min)
                Ymin = min(data_y_min)
                Xmax = max(data_x_max)
                Ymax = max(data_y_max)
            
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
                pbar.update(1)

    elapsed = time.time() - start_time
    print(elapsed)

    cap.release()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="")
    args = parser.parse_args()

    main(args)
