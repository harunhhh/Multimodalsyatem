r"""
F2: VOICEVOX Coreによる音声合成 (Windows / CUDA版)

実行環境: 専用の仮想環境 (Python 3.10, CUDA 12.8)。

以下がプロジェクト直下に配置されていること（いずれもGit管理外。入手方法はREADME.md参照）:
  dict/open_jtalk_dic_utf_8-1.11/  日本語の読み方辞書
  onnxruntime/                     推論エンジン
  models/vvms/0.vvm                音声モデル

このファイル単体で実行すると output.wav / output_2.wav を生成し、動作確認ができる。

使用モデル: ずんだもん（ノーマル）スタイルID: 5 (0.vvm)
"""

import io
import os
import wave

import config

# ---- Windows: onnxruntime の DLL を明示的に登録 ----
_onnx_dll_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "onnxruntime", "lib")
if os.path.isdir(_onnx_dll_dir):
    os.add_dll_directory(_onnx_dll_dir)

from voicevox_core.blocking import (
    Onnxruntime,
    OpenJtalk,
    Synthesizer,
    VoiceModelFile,
)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

_synthesizer = None


def _get_synthesizer() -> Synthesizer:
    global _synthesizer
    if _synthesizer is not None:
        return _synthesizer

    onnxruntime_path = os.path.join(BASE_DIR, "onnxruntime", "lib", Onnxruntime.LIB_VERSIONED_FILENAME)
    open_jtalk_dict_dir = os.path.join(BASE_DIR, "dict", "open_jtalk_dic_utf_8-1.11")

    synthesizer = Synthesizer(
        Onnxruntime.load_once(filename=onnxruntime_path),
        OpenJtalk(open_jtalk_dict_dir),
        acceleration_mode="AUTO",
    )
    vvm_path = os.path.join(BASE_DIR, "models", "vvms", "0.vvm")
    with VoiceModelFile.open(vvm_path) as model:
        synthesizer.load_voice_model(model)

    _synthesizer = synthesizer
    return _synthesizer


def synthesize(text: str, style_id: int = None) -> bytes:
    """テキストをWAVバイト列に変換する。"""
    style_id = config.VOICEVOX_STYLE_ID if style_id is None else style_id
    return _get_synthesizer().tts(text, style_id)


def speak(text: str, style_id: int = None):
    """テキストを読み上げ、スピーカーから再生する（再生完了までブロック）。"""
    import sounddevice as sd

    wav_bytes = synthesize(text, style_id)
    with wave.open(io.BytesIO(wav_bytes)) as wf:
        import numpy as np

        n_frames = wf.getnframes()
        raw = wf.readframes(n_frames)
        dtype = {1: np.int8, 2: np.int16, 4: np.int32}[wf.getsampwidth()]
        samples = np.frombuffer(raw, dtype=dtype).reshape(-1, wf.getnchannels())
        sample_rate = wf.getframerate()

    sd.play(samples, sample_rate)
    sd.wait()


if __name__ == "__main__":
    text = "きょうふのみそしる。ここではきものをおぬぎください。"
    wav = synthesize(text)
    output_path = os.path.join(BASE_DIR, "output.wav")
    with open(output_path, "wb") as f:
        f.write(wav)
    print(f"{output_path} に保存しました。再生します...")
    speak(text)
