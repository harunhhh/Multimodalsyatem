"""GUI: 映像表示・状態表示・停止ボタン（cv2.imshowベース）。"""

import cv2

_WINDOW_NAME = "nidone_boushi_system"
_BUTTON_SIZE = (100, 40)
_BUTTON_MARGIN = 10

_initialized = False
_button_rect = None  # (x, y, w, h)
_stop_clicked = False


def _mouse_callback(event, x, y, flags, param):
    global _stop_clicked
    if event == cv2.EVENT_LBUTTONDOWN and _button_rect is not None:
        bx, by, bw, bh = _button_rect
        if bx <= x <= bx + bw and by <= y <= by + bh:
            _stop_clicked = True


def _ensure_window():
    global _initialized
    if not _initialized:
        cv2.namedWindow(_WINDOW_NAME)
        cv2.setMouseCallback(_WINDOW_NAME, _mouse_callback)
        _initialized = True


def draw_frame(frame, state: str, hold_time: float, extra_lines=None):
    """現在の状態・累積保持時間をフレームに描画して表示する。

    extra_lines: [(text, is_ok_bool_or_None), ...] を追加行として表示する。
    """
    global _button_rect
    _ensure_window()

    display = frame.copy()
    frame_w = display.shape[1]
    bw, bh = _BUTTON_SIZE
    bx = frame_w - bw - _BUTTON_MARGIN
    by = _BUTTON_MARGIN
    _button_rect = (bx, by, bw, bh)

    cv2.rectangle(display, (bx, by), (bx + bw, by + bh), (0, 0, 200), -1)
    cv2.putText(display, "STOP", (bx + 18, by + 27), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    y = 30
    cv2.putText(display, f"state: {state}", (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    y += 30
    cv2.putText(display, f"hold: {hold_time:.1f}s", (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    y += 30

    for text, is_ok in extra_lines or []:
        color = (0, 255, 0) if is_ok is None else ((0, 200, 0) if is_ok else (0, 0, 220))
        cv2.putText(display, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        y += 30

    cv2.imshow(_WINDOW_NAME, display)
    cv2.waitKey(1)


def is_stop_button_clicked() -> bool:
    """停止ボタンがクリックされたかを返す（呼び出し後はフラグをリセット）。"""
    global _stop_clicked
    clicked = _stop_clicked
    _stop_clicked = False
    return clicked
