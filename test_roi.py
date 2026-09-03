"""assets/snapshots/ の画像でROI選択(select_roi_live)をテストする。

使い方:
    python test_roi.py            # 一覧表示 + 最新の1枚で実行
    python test_roi.py <番号>      # 一覧の番号を指定して実行
    python test_roi.py <ファイル名> # ファイル名を直接指定
"""

import os
import sys

import cv2

from modules import futon_monitor

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SNAPSHOT_DIR = os.path.join(BASE_DIR, "assets", "snapshots")


def main():
    files = sorted(
        (f for f in os.listdir(SNAPSHOT_DIR) if f.lower().endswith((".png", ".jpg", ".jpeg"))),
        key=lambda f: os.path.getmtime(os.path.join(SNAPSHOT_DIR, f)),
    )
    if not files:
        print(f"{SNAPSHOT_DIR} に画像がありません。先に capture_snapshot.py で撮影してください。")
        return

    print("画像一覧:")
    for i, f in enumerate(files):
        print(f"  [{i}] {f}")

    arg = sys.argv[1] if len(sys.argv) > 1 else None
    if arg is None:
        target = files[-1]  # 最新
    elif arg.isdigit() and 0 <= int(arg) < len(files):
        target = files[int(arg)]
    elif arg in files:
        target = arg
    else:
        print(f"'{arg}' に一致する画像が見つかりません。")
        return

    print(f"使用する画像: {target}")
    frame = cv2.imread(os.path.join(SNAPSHOT_DIR, target))
    if frame is None:
        print("画像を読み込めませんでした。")
        return

    # select_roi_live はcapオブジェクト（.read()を持つもの）を受け取るため、
    # 静止画を毎回返すだけのアダプタでラップする
    class _StillFrame:
        def read(self):
            return True, frame

    try:
        roi = futon_monitor.select_roi_live(_StillFrame())
    except KeyboardInterrupt:
        print("中止しました。")
        return
    print(f"ROI選択結果: {roi}")


if __name__ == "__main__":
    main()
