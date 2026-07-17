"""F4/ALARM_RINGING: アラーム音のループ再生。"""

import os
import wave

import numpy as np
import sounddevice as sd

_BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _load_wav(path):
    with wave.open(path, "rb") as wf:
        n_frames = wf.getnframes()
        raw = wf.readframes(n_frames)
        dtype = {1: np.int8, 2: np.int16, 4: np.int32}[wf.getsampwidth()]
        samples = np.frombuffer(raw, dtype=dtype).reshape(-1, wf.getnchannels()).astype(np.float32)
        samples /= np.iinfo(dtype).max
        return samples, wf.getframerate()


def play_loop(relative_path: str, volume: float):
    """指定した音声ファイルを指定音量でループ再生する（既存の再生は停止して差し替え）。"""
    samples, sample_rate = _load_wav(os.path.join(_BASE_DIR, relative_path))
    sd.stop()
    sd.play(samples * volume, sample_rate, loop=True)


def stop():
    sd.stop()
