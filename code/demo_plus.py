import copy
from pybboxes import BoundingBox

import cv2
from motpy import Detection,MultiObjectTracker

from openpifpaf.predictor import Predictor
import numpy as np

def main():
    mode = "camera"
    if(mode == "camera"):
        cap = cv2.VideoCapture(0)
    elif(mode == "video"):
        cap = cv2.VideoCapture('person.mp4')
    
    predictor = Predictor()
    
    
    while True:
        ret, frame = cap.read()
        if not ret: break
        debug_image = copy.deepcopy(frame)
        
        pred, _, meta = predictor.numpy_image(frame)
        
        poses = []
        
        
        for p in pred:
            pose = p.json_data()
            poses.append(pose)
        bboxs =[]
        score =[]
        label = []
        for p in poses:
            my_coco_box = p['bbox']
            scores = p['score']
            labels = p['category_id']
        
            coco_bbox = BoundingBox.from_coco(*my_coco_box)
            voc_bbox = coco_bbox.to_voc(return_values=True)
            bboxs.append(voc_bbox)
            score.append(scores)
            label.append(labels)
        print(bboxs)
        print(len(bboxs[0]))
        
        
        
        '''
            bbox = pose['bbox']
            score = pose['score']
            label = pose['category_id']
            c_bbox = BoundingBox.from_coco(*bbox)
            cc_box = c_bbox.to_voc()
            bbox_s.append(cc_box)
            scores.append(score)
            labels.append(label)
            print(bbox_s,scores,labels)
        '''
            
            
        #motpy_engine(debug_image,bbox_s,scores,labels)
        #print(bbox_s,scores,labels)
        #Prepare motpy
        fps = 60
        tracker = MultiObjectTracker(
        dt=(1/fps),
        tracker_kwargs={'max_staleness':5},
        model_spec={
            'order_pos':1,
            'dim_pos': 2,
            'order_size': 0,
            'dim_size': 2,
            'q_var_pos':5000.0,
            'r_var_pos':0.1
        },
        matching_fn_kwargs={
            'min_iou':0.25,
            'multi_match_min_iou':0.03
        },
    )
    
        track_id_dict ={}
    
        # motpy入力用のDetectionクラスにデータを設定する
        detection =[
        Detection(box = b ,score = s, class_id = l)
        for b,s,l in zip(bboxs,score,label)
        ]
    
        _ = tracker.step(detections=detection)
        track_results = tracker.active_tracks(min_steps_alive= 2)
    
        for track_results in track_results:
            if track_results.id not in track_id_dict:
                new_id = len(track_id_dict)
                track_id_dict[track_results.id] = new_id
                print(new_id)
            
        debug_image = draw_debug(
            debug_image,
            track_results,
            track_id_dict
            )
    
    
        cv2.imshow('OpenPifPaf_tracker',debug_image)
            
        
        # キー処理(ESC：終了)
        key = cv2.waitKey(1)
        if key == 27:  # ESC
            break
        
    cap.release()
    cv2.destroyAllWindows()

def motpy_engine(pic,d_box , d_score,d_label = 'person'):
    
    debug_image = pic 
    #Prepare motpy
    fps = 30
    tracker = MultiObjectTracker(
        dt=(1/fps),
        tracker_kwargs={'max_staleness':5},
        model_spec={
            'order_pos':1,
            'dim_pos': 2,
            'order_size': 0,
            'dim_size': 2,
            'q_var_pos':5000.0,
            'r_var_pos':0.1
        },
        matching_fn_kwargs={
            'min_iou':0.25,
            'multi_match_min_iou':0.03
        },
    )
    
    track_id_dict ={}
    boxes = d_box
    scores = d_score
    labels = d_label
    
    # motpy入力用のDetectionクラスにデータを設定する
    detections =[
        Detection(box = b ,score = s, class_id = l)
        for b,s,l in zip(boxes,scores,labels)
        ]
    
    _ = tracker.step(detections= detections)
    track_results = tracker.active_tracks(min_steps_alive= 3)
    for track_results in track_results:
        if track_results.id not in track_id_dict:
            new_id = len(track_id_dict)
            track_id_dict[track_results.id] = new_id
            
    debug_image = draw_debug(
            debug_image,
            track_results,
            track_id_dict
        )
    
    
    cv2.imshow('OpenPifPaf_tracker',debug_image)
            


def draw_debug(
    image,
    track_results,
    track_id_dict,
):
    debug_image = copy.deepcopy(image)

    for track_result in track_results:
        tracker_id = track_id_dict[track_result.id]
        bbox = track_result.box
        class_id = int(track_result.class_id)
        score = track_result.score

        x1, y1, x2, y2 = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])

        # トラッキングIDに応じた色を取得
        color = get_id_color(tracker_id)

        # バウンディングボックス描画
        debug_image = cv2.rectangle(
            debug_image,
            (x1, y1),
            (x2, y2),
            color,
            thickness=2,
        )

        # スコア、ラベル名描画
        score = '%.2f' % score
        text = '%s' % (score)
        debug_image = cv2.putText(
            debug_image,
            text,
            (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            color,
            thickness=2,
        )

    return debug_image

def get_id_color(index):
    temp_index = (index + 1) * 5
    color = (
        (37 * temp_index) % 255,
        (17 * temp_index) % 255,
        (29 * temp_index) % 255,
    )
    return color

def data_analysis(poses):
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
        
        
if __name__ == '__main__':
    main()
    