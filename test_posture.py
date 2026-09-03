"""F1: modules/posture.pyのWebカメラ実機テスト（閾値調整用）。

Webカメラ映像にキーポイント・各判定条件の内訳・OK/NGを重ねて表示する。

サンプル収集:
    'u' キー → カウントダウン後に「起きている姿勢」を自動記録
    'd' キー → カウントダウン後に「寝ている姿勢」を自動記録
    キーを押してから姿勢をとる時間があるので、手を上げたまま操作する必要はない。
    'q' キーで終了し、posture_samples.log に保存する。
"""

import cv2

import config
from modules import posture

SAMPLE_LOG = "posture_samples.log"
COUNTDOWN_SEC = 5.0      # キーを押してから記録開始までの猶予
CAPTURE_SEC = 2.0        # 記録し続ける時間
CAPTURE_INTERVAL_SEC = 0.4

# 記録したサンプルの見出しとキーポイント名（閾値設計用）
_KP_NAMES = {0: "nose", 5: "l_shoulder", 6: "r_shoulder",
             9: "l_wrist", 10: "r_wrist", 11: "l_hip", 12: "r_hip"}


def _write_samples(samples):
    """ラベル付きサンプルを解析用にテキスト出力する。"""
    with open(SAMPLE_LOG, "w", encoding="utf-8") as f:
        for n, (label, info) in enumerate(samples, 1):
            f.write(f"--- sample {n}: {label} ---\n")
            f.write(f"shoulder_width={info['shoulder_width']:.1f} "
                    f"upright_basis={info['upright_basis']} "
                    f"offset_ratio={info['offset_ratio']:.3f} "
                    f"hips_visible={info['hips_visible']} "
                    f"wrist_above_nose={info['wrist_above_nose']}\n")
            for i, name in _KP_NAMES.items():
                x, y = info["points"][i]
                f.write(f"  {name:<11} x={x:7.1f} y={y:7.1f} conf={info['confs'][i]:.2f}\n")
    print(f"{len(samples)}件を {SAMPLE_LOG} に保存しました")


def main():
    cap = cv2.VideoCapture(config.CAMERA_INDEX)
    if not cap.isOpened():
        print("カメラを開けませんでした")
        return

    hold_total = 0.0
    prev_tick = cv2.getTickCount()
    now = 0.0

    samples = []
    pending_label = None     # 記録予約中のラベル
    countdown_until = 0.0
    capture_until = 0.0
    next_capture_at = 0.0
    notice = ""
    notice_until = 0.0

    print("'u': 起きている姿勢を記録 / 'd': 寝ている姿勢を記録 / 'q': 終了して保存")

    while True:
        ok, frame = cap.read()
        if not ok:
            print("フレーム取得に失敗しました")
            break

        now_tick = cv2.getTickCount()
        dt = (now_tick - prev_tick) / cv2.getTickFrequency()
        prev_tick = now_tick
        now += dt

        result = posture._get_model()(frame, verbose=False)
        annotated = result[0].plot()

        posture_ok, info = posture.check_posture_debug(frame)

        if posture_ok:
            hold_total += dt
        # 姿勢が崩れたら表示をリセットする。
        # （main.pyは仕様上「累積」で判定するが、この調整ツールでは今の状態を見たいため）
        else:
            hold_total = 0.0

        # --- カウントダウン → 自動記録 ---
        if pending_label and now >= countdown_until:
            if capture_until == 0.0:
                capture_until = now + CAPTURE_SEC
                next_capture_at = now
            if now >= next_capture_at:
                next_capture_at = now + CAPTURE_INTERVAL_SEC
                if info:
                    samples.append((pending_label, info))
                    print(f"[{len(samples)}] {pending_label} を記録しました")
                else:
                    notice = "NOT RECORDED: person not detected"
                    notice_until = now + 2.0
            if now >= capture_until:
                pending_label = None
                capture_until = 0.0

        y = 30
        color_ok = (0, 200, 0)
        color_ng = (0, 0, 220)

        def put(text, ok=None, scale=0.6):
            nonlocal y
            color = (255, 255, 255) if ok is None else (color_ok if ok else color_ng)
            cv2.putText(annotated, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, 2)
            y += 25

        put(f"WOKEN: {posture_ok}  hold={hold_total:.1f}s", ok=posture_ok, scale=0.9)
        if posture_ok and hold_total >= config.POSTURE_HOLD_REQUIRED_SEC:
            put(">>> WOKEN (3s reached) <<<", ok=True, scale=0.9)
        if info:
            put(f"wrist_above_nose: {info['wrist_above_nose']}", ok=info["wrist_above_nose"])
            put(f"is_upright by {info['upright_basis']}(ratio={info['offset_ratio']:.2f}): {info['is_upright']}",
                ok=info["is_upright"])
            put(f"conf_ok(nose/shoulders): {info['conf_ok']}", ok=info["conf_ok"])
            put(f"hips_visible: {info['hips_visible']}")
        else:
            put("person not detected", ok=False)

        put(f"u: upright sample   d: lying sample   q: quit & save")
        put(f"samples: {len(samples)}")
        if notice and now < notice_until:
            put(notice, ok=False)

        # カウントダウン／記録中は大きく表示する（離れていても見えるように）
        if pending_label:
            if now < countdown_until:
                put(f">>> {pending_label.upper()} in {countdown_until - now:.1f}s - GET IN POSITION <<<",
                    ok=None, scale=0.9)
            else:
                put(f">>> RECORDING {pending_label.upper()} <<<", ok=True, scale=0.9)

        cv2.imshow("posture test (u:upright, d:lying, q:quit)", annotated)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        elif key in (ord("u"), ord("d")) and not pending_label:
            pending_label = "upright" if key == ord("u") else "lying"
            countdown_until = now + COUNTDOWN_SEC
            capture_until = 0.0
            print(f"{COUNTDOWN_SEC:.0f}秒後に {pending_label} を記録します。姿勢をとってください。")

    cap.release()
    cv2.destroyAllWindows()

    if samples:
        _write_samples(samples)
    else:
        print("サンプルが記録されませんでした。")


if __name__ == "__main__":
    main()
