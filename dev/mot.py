import copy
import openpifpaf_voc as openpifpaf

import cv2
from motpy import Detection,MultiObjectTracker
import numpy as np

def get_id_color(index):
    temp_index = (index + 1) * 5
    color = (
        (37 * temp_index) % 255,
        (17 * temp_index) % 255,
        (29 * temp_index) % 255,
    )
    return color

def draw_debug(image,track_results,track_id_dict):
    
    debug_image = copy.deepcopy(image)
    
    for track_result in track_results:
        
        tracker_id = track_id_dict[track_result.id]
        bbox = track_result.box
        class_id = int(track_result.class_id)
        score = track_result.score
        
        x1,x2,y1,y2 = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
        
        # Color data is got by tracking id
        color = get_id_color(tracker_id)
        
        # Writing the bounding box
        debug_image = cv2.rectangle(
            debug_image,
            (x1,y1),
            (x2,y2),
            color,
            thickness = 2
        )
        
        # Writing the score and text
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

# Video capture (Webcam)
cap = cv2.VideoCapture('re2.mp4')

# Prepare Motpy
fps = cap.get(cv2.CAP_PROP_FPS)
tracker = MultiObjectTracker(
    dt=(1 / fps),
    tracker_kwargs={'max_staleness': 10},
    model_spec={
        'order_pos': 1,
        'dim_pos': 2,
        'order_size': 0,
        'dim_size': 2,
        'q_var_pos': 2500.0,
        'r_var_pos': 0.1
    },
    matching_fn_kwargs={
        'min_iou': 0.25,
        'multi_match_min_iou': 0.93
    },
)

# The Value which saving the tracking ID

track_id_dict = {}

while True:
    
    ret, frame = cap.read()
    
    debug_image = copy.deepcopy(frame)
    
    # Object Detection function execution
    
    boxes, scores, labels = openpifpaf.openpifpaf_voc.voc_pub(ret, frame)
    
    
    
    detections = [Detection(box = b,score = s, class_id = l)
    for b,s,l in zip(boxes, scores,labels)]
    
    # execution the tracking by motpy
    _ = tracker.step(detections = detections)
    track_results = tracker.active_tracks(min_steps_alive= 4)
    
    # connection with serial number and trackingID
    for track_result in track_results:
        if track_result.id not in track_id_dict:
            new_id = len(track_id_dict)
            track_id_dict[track_result.id] = new_id
            
    debug_image = draw_debug(
        debug_image,
        track_results,
        track_id_dict,
    )
    
    #print(type(debug_image))
    
    key = cv2.waitKey(1)
    if key == 27:
        break
    
    cv2.imshow('OpenPifPaf_MOT',debug_image)
    
cv2.release()
cv2.destroyAllWindows()