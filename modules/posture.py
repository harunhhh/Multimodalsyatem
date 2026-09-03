"""F1: YOLO poseによる姿勢認識（起床判定）。

判定条件（REQUIREMENTS.md 6章 F1をベースに、上半身のみで判定できるよう調整）:
  1. 手首（左右どちらか）が鼻よりy座標で一定以上高い位置にある
     ※ 上げている側の手首だけconfidenceを要求する
  2. 起き上がっている（直立）
     - 腰が見えている場合: 肩中心と腰中心がほぼ垂直に並んでいる（従来通り）
     - 腰が見えない場合  : 肩のラインの傾きが小さい（横になると縦向きに近づくため）
  3. 鼻・両肩のconfidenceが閾値以上（腰は必須ではない）

全身がフレームに入らない状況（椅子に座る・ベッド上で上体を起こす）でも判定できる。
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

    threshold = config.POSTURE_KEYPOINT_CONF_THRESHOLD
    watched = [NOSE, LEFT_SHOULDER, RIGHT_SHOULDER, LEFT_WRIST, RIGHT_WRIST, LEFT_HIP, RIGHT_HIP]
    confs = {i: float(conf[i]) for i in watched}

    # 上半身だけで判定できるよう、必須は鼻・両肩のみ。腰は見えていれば使う。
    core = [NOSE, LEFT_SHOULDER, RIGHT_SHOULDER]
    conf_ok = all(confs[i] >= threshold for i in core)

    # 手首は「上げている側」だけconfidenceを要求する（下ろした手が写っていなくてもよい）
    nose_y = float(xy[NOSE][1])
    margin = config.POSTURE_WRIST_ABOVE_NOSE_MARGIN_PX
    wrist_above_nose = any(
        confs[w] >= threshold and float(xy[w][1]) < nose_y - margin
        for w in (LEFT_WRIST, RIGHT_WRIST)
    )

    shoulder_center_x = (float(xy[LEFT_SHOULDER][0]) + float(xy[RIGHT_SHOULDER][0])) / 2
    shoulder_width = abs(float(xy[LEFT_SHOULDER][0]) - float(xy[RIGHT_SHOULDER][0]))
    hips_visible = confs[LEFT_HIP] >= threshold and confs[RIGHT_HIP] >= threshold

    # 直立判定は肩の傾きのみで行う。
    # 横になると肩のラインが縦向きに近づき、y差が大きく・x差(肩幅)が小さくなるため比が跳ね上がる。
    # 腰は使わない: 布団の上で上体を起こすと腰が体の前方にずれ、肩中心とのxズレが
    # 大きく出てしまい「直立していない」と誤判定されるため（実測 ratio=0.63〜40）。
    shoulder_tilt = abs(float(xy[LEFT_SHOULDER][1]) - float(xy[RIGHT_SHOULDER][1]))
    offset_ratio = shoulder_tilt / shoulder_width if shoulder_width else float("inf")
    is_upright = offset_ratio <= config.POSTURE_SHOULDER_TILT_MAX_RATIO
    upright_basis = "shoulder"

    # 参考値として腰基準の比も残す（判定には使わない）
    if hips_visible:
        hip_center_x = (float(xy[LEFT_HIP][0]) + float(xy[RIGHT_HIP][0])) / 2
        hip_offset_ratio = abs(shoulder_center_x - hip_center_x) / shoulder_width if shoulder_width else float("inf")
    else:
        hip_offset_ratio = None

    info = {
        "points": {i: (float(xy[i][0]), float(xy[i][1])) for i in watched},
        "shoulder_width": shoulder_width,
        "confs": confs,
        "conf_ok": conf_ok,
        "wrist_above_nose": wrist_above_nose,
        "offset_ratio": offset_ratio,
        "hip_offset_ratio": hip_offset_ratio,
        "is_upright": is_upright,
        "hips_visible": hips_visible,
        "upright_basis": upright_basis,
        "person_idx": person_idx,
    }
    return bool(conf_ok and wrist_above_nose and is_upright), info
