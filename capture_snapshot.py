"""Webカメラのライブプレビューを見ながら、任意のタイミングでフレームを保存する。

's'キー: 現在のフレームを assets/snapshots/ に保存
'q'キー: 終了
"""

import os
from datetime import datetime

import cv2

import config

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SNAPSHOT_DIR = os.path.join(BASE_DIR, "assets", "snapshots")


def main():
    os.makedirs(SNAPSHOT_DIR, exist_ok=True)

    cap = cv2.VideoCapture(config.CAMERA_INDEX)
    if not cap.isOpened():
        print("カメラを開けませんでした")
        return

    print("'s'キー: 保存 / 'q'キー: 終了")
    while True:
        ok, frame = cap.read()
        if not ok:
            print("フレーム取得に失敗しました")
            break

        cv2.imshow("snapshot (s:save, q:quit)", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("s"):
            filename = f"snapshot_{datetime.now():%Y%m%d_%H%M%S}.png"
            path = os.path.join(SNAPSHOT_DIR, filename)
            cv2.imwrite(path, frame)
            print(f"保存しました: {path}")
        elif key == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
