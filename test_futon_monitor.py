"""F3: modules/futon_monitor.pyの実機テスト（calibrate/check_change）。

1. ライブプレビューを見ながら's'キーでROI選択用のフレームを撮影
2. ROIを選択
3. 's'キーで現在のフレームを基準（calibrate）として記録
4. 以降、変化ピクセル割合(check_change)をリアルタイム表示
5. 'q'キーで終了
"""

import cv2

import config
from modules import futon_monitor


def main():
    cap = cv2.VideoCapture(config.CAMERA_INDEX)
    if not cap.isOpened():
        print("カメラを開けませんでした")
        return

    print("ライブ映像上でROIをドラッグ選択 → Enterで確定 / 'r':やり直し / 'q':終了")
    try:
        roi = futon_monitor.select_roi_live(cap)
    except KeyboardInterrupt:
        cap.release()
        cv2.destroyAllWindows()
        return
    print(f"ROI: {roi}")

    baseline = None
    change_start = None

    print("'s'キー: 現在のフレームを基準として記録 / 'r'キー: 基準をリセット / 'q'キー: 終了")
    while True:
        ok, frame = cap.read()
        if not ok:
            print("フレーム取得に失敗しました")
            break

        x, y, w, h = roi
        display = frame.copy()
        cv2.rectangle(display, (x, y), (x + w, y + h), (0, 255, 0), 2)

        if baseline is None:
            cv2.putText(display, "press 's' to take baseline", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        else:
            ratio = futon_monitor.check_change(frame, roi, baseline)
            is_change = ratio >= config.FUTON_CHANGE_RATIO_THRESHOLD
            color = (0, 0, 220) if is_change else (0, 200, 0)

            now = cv2.getTickCount() / cv2.getTickFrequency()
            if is_change:
                if change_start is None:
                    change_start = now
                held = now - change_start
            else:
                change_start = None
                held = 0.0

            cv2.putText(display, f"change_ratio={ratio:.2f} (threshold={config.FUTON_CHANGE_RATIO_THRESHOLD})",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            cv2.putText(display, f"held={held:.1f}s / confirm={config.FUTON_CHANGE_CONFIRM_SEC}s",
                        (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            if held >= config.FUTON_CHANGE_CONFIRM_SEC:
                cv2.putText(display, ">>> RELAPSE DETECTED <<<", (10, 90),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        cv2.imshow("futon monitor test (s:calibrate, q:quit)", display)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("s"):
            baseline = futon_monitor.calibrate(frame, roi)
            change_start = None
            print("基準フレームを記録しました")
        elif key == ord("r"):
            baseline = None
            change_start = None
            print("基準フレームをリセットしました")
        elif key == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
