r"""
画像・音声認識 第11回 - VOICEVOX音声合成 (Windows / CUDA版)
スライド p.1 ~ p.10 に対応するコード

【実行環境】
    仮想環境  : GPTSoVits (Python 3.10, CUDA 12.8)
    実行方法  : C:\Users\nk2400\anaconda3\envs\GPTSoVits\python.exe 1_10.py
    作業フォルダ: 11kai/ (dict/, models/, onnxruntime/ が同フォルダにあること)

【事前準備】（voicevox_setup_windows.md 参照）
    1. download-windows-x64.exe を 11kai/ に置いて CUDA版モデルをダウンロード
       > .\download-windows-x64.exe -o . --exclude c-api --devices cuda
    2. CUDA版 wheel をインストール
       > C:\...\GPTSoVits\python.exe -m pip install
           https://github.com/VOICEVOX/voicevox_core/releases/download/0.16.4/
           voicevox_core-0.16.4+cuda-cp310-abi3-win_amd64.whl

【使用モデル】
    ずんだもん（ノーマル）スタイルID: 5 (0.vvm)
"""

import os
import sys

# ---- Windows: onnxruntime の DLL を明示的に登録 ----
# ImportError: DLL load failed が出る場合の対策
_onnx_dll_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "onnxruntime", "lib")
if os.path.isdir(_onnx_dll_dir):
    os.add_dll_directory(_onnx_dll_dir)

from voicevox_core.blocking import (
    Onnxruntime,
    OpenJtalk,
    Synthesizer,
    VoiceModelFile,
)

# --------
# パス設定（プロジェクトルート直下に dict/, models/, onnxruntime/ がある想定。
# 本ファイルは modules/ 配下にあるため、1階層上をルートとする）
# --------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

voicevox_onnxruntime_path = os.path.join(
    BASE_DIR, "onnxruntime", "lib", Onnxruntime.LIB_VERSIONED_FILENAME
)
open_jtalk_dict_dir = os.path.join(BASE_DIR, "dict", "open_jtalk_dic_utf_8-1.11")

# --------
# 初期化 (CUDA を優先、使えない場合は自動でCPUにフォールバック)
# --------
print("VOICEVOX Core を初期化中...")
synthesizer = Synthesizer(
    Onnxruntime.load_once(filename=voicevox_onnxruntime_path),
    OpenJtalk(open_jtalk_dict_dir),
    acceleration_mode="AUTO",  # AUTO: GPU/CPU を自動選択
)
print("  初期化完了。")

# --------
# モデルの読み込み
# ずんだもん（ノーマル）（スタイルID: 5）→ 0.vvm を使用
# --------
style_id = 5
vvm_path = os.path.join(BASE_DIR, "models", "vvms", "0.vvm")

print(f"モデルを読み込み中: {vvm_path}")
with VoiceModelFile.open(vvm_path) as model:
    synthesizer.load_voice_model(model)
print("  モデル読み込み完了。")

# ========================================
# 1. テキスト読み上げ（ひらがな/漢字入力）
# ========================================
print("\n--- 1. テキスト読み上げ (tts) ---")
text = "きょうふのみそしる。ここではきものをおぬぎください。"
print(f"  入力テキスト: {text}")

wav = synthesizer.tts(text, style_id)

output_path = os.path.join(BASE_DIR, "output.wav")
with open(output_path, "wb") as f:
    f.write(wav)
print(f"  → {output_path} に保存しました。")

# ============================================================
# 2. アクセント記号付きカナ入力による読み上げ（tts_from_kana）
# ============================================================
print("\n--- 2. アクセント付きカナ読み上げ (tts_from_kana) ---")
# アクセント記法:
#   - 全てのカナはカタカナで記述
#   - アクセント句は / または 、 で区切る（ 、 は無音区間が挿入される）
#   - カナの手前に _ を入れるとそのカナは無声化される
#   - アクセント位置を ' で指定（各アクセント句に1つ必須）
#   - アクセント句末に ？（全角）を入れると疑問文の発音になる
text_kana = "キョ'ウ/フノミソシ'ル、ココ'デ/ハキモノオ'/オヌギクダ'サイ"
print(f"  入力カナ: {text_kana}")

wav2 = synthesizer.tts_from_kana(text_kana, style_id)

output_path2 = os.path.join(BASE_DIR, "output_2.wav")
with open(output_path2, "wb") as f:
    f.write(wav2)
print(f"  → {output_path2} に保存しました。")

print("\n完了！output.wav と output_2.wav を再生してください。")
