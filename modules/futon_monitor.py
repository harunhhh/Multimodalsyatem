"""F3: 背景差分によるベッド（布団）監視。

単眼RGBカメラでは奥行き情報がないため、「高さ・幅」は物理量ではなく
ROI内の変化ピクセル割合という面積ベースの近似指標として扱う（REQUIREMENTS.md 6章F3）。
"""

import cv2
import numpy as np

# cv2.putTextは日本語を描画できないため、画面上の案内は英字で表示する
_ROI_WINDOW_NAME = "Select futon area"


def _rect_from_points(start, end):
    """ドラッグの始点・終点から (x, y, w, h) を作る。範囲が無い場合はNone。"""
    if start is None or end is None:
        return None
    x = min(start[0], end[0])
    y = min(start[1], end[1])
    w = abs(end[0] - start[0])
    h = abs(end[1] - start[1])
    if w == 0 or h == 0:
        return None
    return (int(x), int(y), int(w), int(h))


def select_roi_live(cap):
    """ライブ映像の上で直接ベッド領域をドラッグ選択する。

    静止画を撮る手順は無く、映像を見ながらドラッグ → Enter で確定できる。
    r で選び直し、q で中止（KeyboardInterrupt）。

    Returns:
        tuple[int, int, int, int]: (x, y, w, h)
    """
    drag = {"start": None, "end": None, "active": False}

    def on_mouse(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            drag["start"] = (x, y)
            drag["end"] = (x, y)
            drag["active"] = True
        elif event == cv2.EVENT_MOUSEMOVE and drag["active"]:
            drag["end"] = (x, y)
        elif event == cv2.EVENT_LBUTTONUP:
            drag["end"] = (x, y)
            drag["active"] = False

    cv2.namedWindow(_ROI_WINDOW_NAME)
    cv2.setMouseCallback(_ROI_WINDOW_NAME, on_mouse)
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                raise RuntimeError("ROI設定用のフレーム取得に失敗しました")

            display = frame.copy()
            roi = _rect_from_points(drag["start"], drag["end"])
            if roi is not None:
                x, y, w, h = roi
                cv2.rectangle(display, (x, y), (x + w, y + h), (0, 255, 0), 2)
                guide = "Enter: confirm   r: redo   q: quit"
            else:
                guide = "Drag to select the futon area"
            cv2.putText(display, guide, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.imshow(_ROI_WINDOW_NAME, display)

            key = cv2.waitKey(30) & 0xFF
            if key in (13, 10) and roi is not None:  # Enter
                return roi
            if key == ord("r"):
                drag["start"] = drag["end"] = None
                drag["active"] = False
            elif key == ord("q"):
                raise KeyboardInterrupt
    finally:
        cv2.destroyWindow(_ROI_WINDOW_NAME)
        cv2.waitKey(1)  # Windowsではイベントを回さないとウィンドウが残る


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
