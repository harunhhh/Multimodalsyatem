"""F1: modules/posture.pyのWebカメラ実機テスト（閾値調整用）。

Webカメラ映像にキーポイント・各判定条件の内訳・OK/NGを重ねて表示する。
'q'キーで終了。
"""

import cv2

import config
from modules import posture


def main():
    cap = cv2.VideoCapture(config.CAMERA_INDEX)
    if not cap.isOpened():
        print("カメラを開けませんでした")
        return

    hold_total = 0.0
    hold_active = False
    prev_tick = cv2.getTickCount()

    while True:
        ok, frame = cap.read()
        if not ok:
            print("フレーム取得に失敗しました")
            break

        now_tick = cv2.getTickCount()
        dt = (now_tick - prev_tick) / cv2.getTickFrequency()
        prev_tick = now_tick

        result = posture._get_model()(frame, verbose=False)
        annotated = result[0].plot()

        posture_ok, info = posture.check_posture_debug(frame)

        if posture_ok:
            hold_total += dt
            hold_active = True
        else:
            hold_active = False

        y = 30
        color_ok = (0, 200, 0)
        color_ng = (0, 0, 220)

        def put(text, ok=None):
            nonlocal y
            color = (255, 255, 255) if ok is None else (color_ok if ok else color_ng)
            cv2.putText(annotated, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            y += 25

        put(f"WOKEN: {posture_ok}  hold={hold_total:.1f}s", ok=posture_ok)
        if info:
            put(f"wrist_above_nose: {info['wrist_above_nose']}", ok=info["wrist_above_nose"])
            put(f"is_upright(offset_ratio={info['offset_ratio']:.2f}): {info['is_upright']}", ok=info["is_upright"])
            put(f"conf_ok: {info['conf_ok']}", ok=info["conf_ok"])
            for i, c in info["confs"].items():
                put(f"  kp{i} conf={c:.2f}", ok=c >= config.POSTURE_KEYPOINT_CONF_THRESHOLD)
        else:
            put("person not detected")

        if hold_total >= config.POSTURE_HOLD_REQUIRED_SEC:
            put(">>> WOKEN (3s reached) <<<", ok=True)

        cv2.imshow("posture test (q to quit)", annotated)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
