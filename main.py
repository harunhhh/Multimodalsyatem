"""エントリーポイント・状態遷移管理（REQUIREMENTS.md 7章のステートマシン）"""

import logging
import time
import winsound
from datetime import datetime
from enum import Enum, auto

import cv2

import config
from modules import alarm, posture, futon_monitor, llm_client, ui, voice

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.FileHandler("nidone.log", encoding="utf-8"), logging.StreamHandler()],
)


class State(Enum):
    STANDBY = auto()
    ALARM_RINGING = auto()
    POSTURE_HOLDING = auto()
    WOKEN = auto()
    GREETING = auto()
    MONITORING = auto()
    RELAPSE_ALARM = auto()


class NidoneBoushiSystem:
    def __init__(self):
        self.state = State.STANDBY
        self.cap = None
        self.posture_hold_start = None
        self.posture_hold_total = 0.0
        self.last_alarm_volume_up = None
        self.futon_roi = None
        self.futon_baseline = None
        self.futon_change_start = None
        self.monitoring_start = None
        self.camera_last_ok_time = None
        self.camera_error_notified = False
        self.alarm_sound_path = config.ALARM_NORMAL_SOUND_PATH
        self.alarm_volume = config.ALARM_BASE_VOLUME
        self.is_relapse_wake = False

    def run(self):
        self._setup_futon_roi()
        try:
            while True:
                if self.state == State.STANDBY:
                    self._run_standby()
                elif self.state == State.ALARM_RINGING:
                    self._run_alarm_ringing()
                elif self.state == State.POSTURE_HOLDING:
                    self._run_posture_holding()
                elif self.state == State.WOKEN:
                    self._run_woken()
                elif self.state == State.GREETING:
                    self._run_greeting()
                elif self.state == State.MONITORING:
                    self._run_monitoring()
                elif self.state == State.RELAPSE_ALARM:
                    self._run_relapse_alarm()
        except KeyboardInterrupt:
            self._shutdown()

    def _transition(self, new_state: State):
        logging.info("状態遷移: %s -> %s", self.state.name, new_state.name)
        self.state = new_state

    def _setup_futon_roi(self):
        temp_cap = cv2.VideoCapture(config.CAMERA_INDEX)
        window_name = "撮影プレビュー (s:撮影)"
        roi_frame = None
        while roi_frame is None:
            ok, frame = temp_cap.read()
            if not ok:
                temp_cap.release()
                raise RuntimeError("ROI設定用のフレーム取得に失敗しました")
            cv2.imshow(window_name, frame)
            if cv2.waitKey(1) & 0xFF == ord("s"):
                roi_frame = frame
        temp_cap.release()
        cv2.destroyAllWindows()
        self.futon_roi = futon_monitor.set_roi(roi_frame)

    def _run_standby(self):
        now = datetime.now()
        if now.hour == config.ALARM_HOUR and now.minute == config.ALARM_MINUTE:
            self.cap = cv2.VideoCapture(config.CAMERA_INDEX)
            self.last_alarm_volume_up = time.monotonic()
            self.futon_baseline = None
            self.futon_change_start = None
            self.camera_last_ok_time = time.monotonic()
            self.camera_error_notified = False
            self.alarm_sound_path = config.ALARM_NORMAL_SOUND_PATH
            self.alarm_volume = config.ALARM_BASE_VOLUME
            self.is_relapse_wake = False
            self._transition(State.ALARM_RINGING)
        else:
            time.sleep(1.0)

    def _run_alarm_ringing(self):
        alarm.play_loop(self.alarm_sound_path, self.alarm_volume)
        if self._check_stop_button():
            alarm.stop()
            self._transition(State.STANDBY)
            return
        self._transition(State.POSTURE_HOLDING)

    def _run_posture_holding(self):
        frame = self._read_camera_frame()
        if frame is None:
            time.sleep(config.POSTURE_CHECK_INTERVAL_SEC)
            return

        ui.draw_frame(frame, self.state.name, self.posture_hold_total)
        if self._check_stop_button():
            alarm.stop()
            self._transition(State.STANDBY)
            return

        now = time.monotonic()
        if now - self.last_alarm_volume_up >= config.ALARM_VOLUME_UP_INTERVAL_SEC:
            self.alarm_volume = min(self.alarm_volume + config.ALARM_VOLUME_STEP, config.ALARM_MAX_VOLUME)
            alarm.play_loop(self.alarm_sound_path, self.alarm_volume)
            self.last_alarm_volume_up = now

        posture_ok = posture.check_posture(frame)
        if posture_ok:
            if self.posture_hold_start is None:
                self.posture_hold_start = now
            self.posture_hold_total += now - self.posture_hold_start
            self.posture_hold_start = now
        else:
            self.posture_hold_start = None

        if self.posture_hold_total >= config.POSTURE_HOLD_REQUIRED_SEC:
            alarm.stop()
            self._transition(State.WOKEN)
        else:
            time.sleep(config.POSTURE_CHECK_INTERVAL_SEC)

    def _run_woken(self):
        self._transition(State.GREETING)

    def _run_greeting(self):
        logging.info("挨拶文を生成中...")
        text = llm_client.generate_greeting(is_relapse=self.is_relapse_wake)
        logging.info("挨拶文: %s / 音声合成・再生中...(初回はVOICEVOX初期化のため数秒かかります)", text)
        voice.speak(text)
        logging.info("音声再生完了")
        self._transition(State.MONITORING)

    def _run_monitoring(self):
        frame = self._read_camera_frame()
        if frame is None:
            time.sleep(config.FUTON_CHECK_INTERVAL_SEC)
            return

        hold_time = 0.0 if self.monitoring_start is None else time.monotonic() - self.monitoring_start

        if self.futon_baseline is None:
            ui.draw_frame(frame, self.state.name, hold_time, [("waiting: 布団基準フレーム記録待ち", None)])
            if self._check_stop_button():
                self._transition(State.STANDBY)
                return
            # WOKEN後、人物がフレームから消えたタイミングでベッド状態を基準として記録
            if not posture.is_person_present(frame):
                self.futon_baseline = futon_monitor.calibrate(frame, self.futon_roi)
                self.monitoring_start = time.monotonic()
                self.futon_change_start = None
                logging.info("布団の基準フレームを記録しました。監視を開始します。")
            time.sleep(config.FUTON_CHECK_INTERVAL_SEC)
            return

        change_ratio = futon_monitor.check_change(frame, self.futon_roi, self.futon_baseline)
        now = time.monotonic()
        is_changed = change_ratio >= config.FUTON_CHANGE_RATIO_THRESHOLD
        held = now - self.futon_change_start if is_changed and self.futon_change_start else 0.0
        logging.info("change_ratio=%.3f (threshold=%.2f) held=%.1fs", change_ratio, config.FUTON_CHANGE_RATIO_THRESHOLD, held)
        ui.draw_frame(frame, self.state.name, hold_time, [
            (f"change_ratio={change_ratio:.2f} (threshold={config.FUTON_CHANGE_RATIO_THRESHOLD})", not is_changed),
            (f"relapse held={held:.1f}s / confirm={config.FUTON_CHANGE_CONFIRM_SEC}s", not is_changed),
        ])

        if self._check_stop_button():
            self.futon_baseline = None
            self._transition(State.STANDBY)
            return

        if is_changed:
            if self.futon_change_start is None:
                self.futon_change_start = now
            elif now - self.futon_change_start >= config.FUTON_CHANGE_CONFIRM_SEC:
                self.futon_baseline = None
                self.futon_change_start = None
                self._transition(State.RELAPSE_ALARM)
                return
        else:
            self.futon_change_start = None

        if now - self.monitoring_start >= config.MONITORING_TIMEOUT_SEC:
            self.futon_baseline = None
            self._transition(State.STANDBY)
            return

        time.sleep(config.FUTON_CHECK_INTERVAL_SEC)

    def _run_relapse_alarm(self):
        self.posture_hold_total = 0.0
        self.posture_hold_start = None
        self.last_alarm_volume_up = time.monotonic()
        self.alarm_sound_path = config.ALARM_RELAPSE_SOUND_PATH
        self.alarm_volume = config.ALARM_BASE_VOLUME
        self.is_relapse_wake = True
        self._transition(State.ALARM_RINGING)

    def _check_stop_button(self) -> bool:
        return ui.is_stop_button_clicked()

    def _read_camera_frame(self):
        """カメラからフレームを取得する。取得できない状態が続いたら警告音・ログで通知する。"""
        ok, frame = self.cap.read()
        if ok:
            self.camera_last_ok_time = time.monotonic()
            self.camera_error_notified = False
            return frame

        stale_for = time.monotonic() - self.camera_last_ok_time if self.camera_last_ok_time is not None else 0.0
        if stale_for >= config.CAMERA_STALE_TIMEOUT_SEC and not self.camera_error_notified:
            logging.warning("カメラからフレームを取得できません（切断の可能性、%.1f秒間更新なし）", stale_for)
            winsound.Beep(1000, 300)
            self.camera_error_notified = True
        return None

    def _shutdown(self):
        if self.cap is not None:
            self.cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    NidoneBoushiSystem().run()
