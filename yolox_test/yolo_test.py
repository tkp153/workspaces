import copy

import cv2
from motpy import Detection, MultiObjectTracker


from yolox.yolox_onnx import YoloxONNX


def get_id_color(index):
    temp_index = (index + 1) * 5
    color = (
        (37 * temp_index) % 255,
        (17 * temp_index) % 255,
        (29 * temp_index) % 255,
    )
    return color


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


# OpenCV動画読み込み準備
cap = cv2.VideoCapture('fish.mp4')

# 物体検出(YOLOX-Nano)準備
yolox = YoloxONNX(
    model_path='yolox_nano.onnx',
    input_shape=(416, 416),
    class_score_th=0.3,
    nms_th=0.45,
    nms_score_th=0.1,
    with_p6=False,
)

# motpy準備
fps = 30
tracker = MultiObjectTracker(
    dt=(1 / fps),
    tracker_kwargs={'max_staleness': 5},
    model_spec={
        'order_pos': 1,
        'dim_pos': 2,
        'order_size': 0,
        'dim_size': 2,
        'q_var_pos': 5000.0,
        'r_var_pos': 0.1
    },
    matching_fn_kwargs={
        'min_iou': 0.25,
        'multi_match_min_iou': 0.93
    },
)

# トラッキングID保持用変数
track_id_dict = {}

while True:
    # 動画からフレームを読み込む
    ret, frame = cap.read()
    if not ret:
        break
    debug_image = copy.deepcopy(frame)

    # 物体検出実行
    boxes, scores, labels = yolox.inference(frame)

    # motpy入力用のDetectionクラスにデータを設定する
    detections = [
        Detection(box=b, score=s, class_id=l)
        for b, s, l in zip(boxes, scores, labels)
    ]

    # motpyを用いてトラッキングを実行する
    _ = tracker.step(detections=detections)
    track_results = tracker.active_tracks(min_steps_alive=3)

    # トラッキングIDと連番の紐付け
    for track_result in track_results:
        if track_result.id not in track_id_dict:
            new_id = len(track_id_dict)
            track_id_dict[track_result.id] = new_id

    # 結果を描画する
    debug_image = draw_debug(
        debug_image,
        track_results,
        track_id_dict,
    )

    # キー処理(ESC：終了)
    key = cv2.waitKey(1)
    if key == 27:  # ESC
        break

    # 画面反映
    cv2.imshow('YOLOX MOT', debug_image)

cap.release()
cv2.destroyAllWindows()
