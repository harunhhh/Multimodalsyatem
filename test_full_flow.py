"""main.pyの一連の状態遷移を、STANDBYの時刻待ちをスキップして通しでテストする。

ROI選択(_setup_futon_roi)後、時刻チェックを介さず直接ALARM_RINGINGから開始する。
"""

import time

import cv2

import config
import main as m


def main():
    system = m.NidoneBoushiSystem()

    # _setup_futon_roi()はライブ映像上でROIを選ぶため、先にカメラを開いておく必要がある
    system.cap = cv2.VideoCapture(config.CAMERA_INDEX)
    if not system.cap.isOpened():
        print("カメラを開けませんでした")
        return
    system.camera_last_ok_time = time.monotonic()
    system._setup_futon_roi()

    # STANDBY→ALARM_RINGING遷移時と同じ初期化を、時刻チェック無しで行う
    system.last_alarm_volume_up = time.monotonic()
    system.futon_baseline = None
    system.futon_change_start = None
    system.camera_error_notified = False
    system.alarm_sound_path = config.ALARM_NORMAL_SOUND_PATH
    system.alarm_volume = config.ALARM_BASE_VOLUME
    system.state = m.State.ALARM_RINGING

    print("ALARM_RINGINGから開始します")
    try:
        while True:
            if system.state == m.State.STANDBY:
                print("STANDBYに戻りました。テスト終了。")
                break
            elif system.state == m.State.ALARM_RINGING:
                system._run_alarm_ringing()
            elif system.state == m.State.POSTURE_HOLDING:
                system._run_posture_holding()
            elif system.state == m.State.WOKEN:
                system._run_woken()
            elif system.state == m.State.GREETING:
                system._run_greeting()
            elif system.state == m.State.MONITORING:
                system._run_monitoring()
            elif system.state == m.State.RELAPSE_ALARM:
                system._run_relapse_alarm()
    except KeyboardInterrupt:
        pass
    finally:
        system._shutdown()


if __name__ == "__main__":
    main()
