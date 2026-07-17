"""F1: YOLO poseによる姿勢認識（起床判定）。

判定条件（REQUIREMENTS.md 6章 F1）:
  1. 手首（左右どちらか、または両方）が鼻よりy座標で一定以上高い位置にある
  2. 肩中心と腰中心がほぼ垂直に並んでいる（直立）、かつ上半身がフレーム内
  3. 上記で使うキーポイント（鼻・肩・腰・手首）のconfidenceが全て閾値以上
"""

import os

from ultralytics import YOLO

import config

# COCO 17キーポイントのインデックス
NOSE = 0
LEFT_SHOULDER, RIGHT_SHOULDER = 5, 6
LEFT_WRIST, RIGHT_WRIST = 9, 10
LEFT_HIP, RIGHT_HIP = 11, 12

_BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_model = None


def _get_model() -> YOLO:
    global _model
    if _model is None:
        model_path = os.path.join(_BASE_DIR, config.POSE_MODEL_PATH)
        if not os.path.exists(model_path):
            # ultralytics標準モデル名なら未配置でも自動ダウンロードされる
            model_path = os.path.basename(config.POSE_MODEL_PATH)
        _model = YOLO(model_path)
    return _model


def is_person_present(frame) -> bool:
    """フレーム内に人物が検出されるかを返す（F3: 布団監視のキャリブレーション条件用）。"""
    results = _get_model()(frame, verbose=False)
    boxes = results[0].boxes
    return boxes is not None and boxes.shape[0] > 0


def check_posture(frame) -> bool:
    """規定姿勢（手首>頭・直立・confidence閾値クリア）を満たすか判定する。

    Returns:
        bool: 姿勢OKならTrue
    """
    return check_posture_debug(frame)[0]


def check_posture_debug(frame):
    """check_posture()の判定内訳（各条件の可否・角度等）付きバージョン。閾値調整用。

    Returns:
        tuple[bool, dict]: (姿勢OKか, デバッグ情報。人物未検出時は空dict)
    """
    results = _get_model()(frame, verbose=False)
    keypoints = results[0].keypoints
    if keypoints is None or keypoints.shape[0] == 0:
        return False, {}

    # 複数人検出時は最も面積の大きい（＝手前にいる）人物を採用
    boxes = results[0].boxes
    if boxes is not None and boxes.shape[0] > 0:
        areas = (boxes.xywh[:, 2] * boxes.xywh[:, 3]).tolist()
        person_idx = areas.index(max(areas))
    else:
        person_idx = 0

    xy = keypoints.xy[person_idx]
    conf = keypoints.conf[person_idx] if keypoints.conf is not None else None
    if conf is None:
        return False, {}

    required = [NOSE, LEFT_SHOULDER, RIGHT_SHOULDER, LEFT_WRIST, RIGHT_WRIST, LEFT_HIP, RIGHT_HIP]
    confs = {i: float(conf[i]) for i in required}
    conf_ok = all(c >= config.POSTURE_KEYPOINT_CONF_THRESHOLD for c in confs.values())

    nose_y = float(xy[NOSE][1])
    left_wrist_y = float(xy[LEFT_WRIST][1])
    right_wrist_y = float(xy[RIGHT_WRIST][1])
    wrist_above_nose = (
        left_wrist_y < nose_y - config.POSTURE_WRIST_ABOVE_NOSE_MARGIN_PX
        or right_wrist_y < nose_y - config.POSTURE_WRIST_ABOVE_NOSE_MARGIN_PX
    )

    shoulder_center_x = (float(xy[LEFT_SHOULDER][0]) + float(xy[RIGHT_SHOULDER][0])) / 2
    hip_center_x = (float(xy[LEFT_HIP][0]) + float(xy[RIGHT_HIP][0])) / 2
    shoulder_width = abs(float(xy[LEFT_SHOULDER][0]) - float(xy[RIGHT_SHOULDER][0]))
    offset_ratio = abs(shoulder_center_x - hip_center_x) / shoulder_width if shoulder_width else float("inf")
    is_upright = offset_ratio <= config.POSTURE_VERTICAL_ALIGN_MAX_RATIO

    info = {
        "confs": confs,
        "conf_ok": conf_ok,
        "wrist_above_nose": wrist_above_nose,
        "offset_ratio": offset_ratio,
        "is_upright": is_upright,
        "person_idx": person_idx,
    }
    return bool(conf_ok and wrist_above_nose and is_upright), info
