"""F3: 背景差分によるベッド（布団）監視。

単眼RGBカメラでは奥行き情報がないため、「高さ・幅」は物理量ではなく
ROI内の変化ピクセル割合という面積ベースの近似指標として扱う（REQUIREMENTS.md 6章F3）。
"""

import cv2
import numpy as np


def set_roi(frame):
    """起動時に手動でベッド領域を矩形選択する。

    ドラッグしてEnter/SPACEで確定。確定後は選択範囲を表示し、
    'y'で決定、それ以外のキー（'r'等）で選び直せる。

    Returns:
        tuple[int, int, int, int]: (x, y, w, h)
    """
    window_name = "ROIを選択してEnter"
    while True:
        x, y, w, h = cv2.selectROI(window_name, frame, showCrosshair=True)
        if w == 0 or h == 0:
            print("ROIが選択されていません。もう一度ドラッグして選択してください。")
            continue

        preview = frame.copy()
        cv2.rectangle(preview, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(
            preview, "y:決定  r/その他キー:選び直し",
            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2,
        )
        cv2.imshow(window_name, preview)
        key = cv2.waitKey(0) & 0xFF
        if key == ord("y"):
            cv2.destroyWindow(window_name)
            return (int(x), int(y), int(w), int(h))
        print("選び直します。")


def _crop_gray(frame, roi):
    x, y, w, h = roi
    cropped = frame[y:y + h, x:x + w]
    return cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)


def calibrate(frame, roi):
    """人物がフレームから消えたタイミングの基準フレーム（グレースケール）を記録する。"""
    return _crop_gray(frame, roi)


def check_change(frame, roi, baseline) -> float:
    """基準フレームとの差分から、ROI内の変化ピクセル割合（0.0〜1.0）を返す。"""
    current = _crop_gray(frame, roi)
    diff = cv2.absdiff(baseline, current)
    _, binary = cv2.threshold(diff, 30, 255, cv2.THRESH_BINARY)
    changed = int(np.count_nonzero(binary))
    total = binary.size
    return changed / total if total else 0.0
