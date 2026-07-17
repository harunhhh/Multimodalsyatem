# 開発ログ（Claudeとの会話記録）

このファイルは、本プロジェクトの企画〜要件定義〜初期セットアップまでの経緯をまとめた会話ログ。

---

## 1. 企画の相談

課題ドキュメント「画像音声課題08_IA13L_田中晴.docx」の内容（マルチモーダル対話システムを利用した二度寝防止システム）について、技術的に実現可能か相談。

**結論**:
- ①姿勢認識による起床判定（YOLO pose + OpenCV）→ 実現可能
- ②LLMへの状況伝達と対話生成（Gemini API + VOICEVOX）→ 実現可能
- ③二度寝の常時監視（布団の高さ・幅変化）→ 単眼RGBカメラでは物理的な高さ計測は困難。面積ベースの近似指標として扱う方針で実現可能

## 2. 要件定義

対話形式で要件を詰め、[REQUIREMENTS.md](REQUIREMENTS.md) にまとめた。決定した主な内容:

- **スコープ**: ①姿勢認識起床判定 ②LLM対話 ③二度寝監視 の3機能。スマホ連携・複数ユーザー対応は対象外
- **F1（姿勢認識）**: `yolov8n-pose.pt`使用。手首が頭より高い＋肩腰が垂直＋上半身がフレーム内、をAND条件とし、キーポイント信頼度閾値も課す。累積3秒保持で起床成立。判定間隔は約0.15秒
- **F2（LLM対話）**: Gemini APIで挨拶文生成 → VOICEVOX（ずんだもん、style_id=5）で読み上げ。11kai/voicevox.py の構成を流用
- **F3（二度寝監視）**: 起動時に手動でROI指定。起床成立後、人物がフレームから消えたタイミングでベッド状態を基準フレームとして記録。背景差分の変化ピクセル割合（暫定15〜20%）が10秒継続したら二度寝と判定。監視間隔は1秒
- **F4（異常系）**: カメラ切断→警告音、Gemini APIタイムアウト→定型文フォールバック、手動停止→画面上のボタンクリック
- **状態遷移**: STANDBY → ALARM_RINGING（30秒ごと音量UP） → POSTURE_HOLDING → WOKEN → GREETING → MONITORING → （二度寝時は専用音でRELAPSE_ALARM→ALARM_RINGINGへ、累積タイマーは0リセット）
- **環境**: GPTSoVits共有環境ではなく、新規に `nidone` という専用conda環境（Python 3.10）を作成して一本化

## 3. 環境構築

1. `11kai/`（授業フォルダ）から `dict/`, `models/vvms/0.vvm`, `onnxruntime/` をコピーし、プロジェクト単体で完結する構成にした
2. `11kai/voicevox.py` を `modules/voice.py` としてコピー
3. `nidone` conda環境を新規作成し、`ultralytics`, `opencv-python`, `sounddevice`, `google-generativeai`, CUDA版`torch`/`torchvision`（cu128）, `voicevox_core` をインストール

**トラブルと対応**:
- 最初 `GPTSoVits` 環境に直接パッケージを入れてしまい、共有環境のtorchがCUDA版→CPU版に書き換わる事故が発生 → `nidone` 環境を新規に切り直す方針に変更し、`GPTSoVits` 環境は `torch==2.11.0+cu128` に復元して事なきを得た
- `voice.py` のパス指定バグ（`modules/`配下なのにルート直下前提だった）を修正
- `voicevox_core` のAPI変更（`AccelerationMode.AUTO`→文字列`"AUTO"`、importパスの変更、`.meta`属性の廃止）に追従
- 上記修正後、`nidone`環境で`voice.py`実行→`output.wav`/`output_2.wav`生成を確認し、動作確認完了

## 4. Git管理

- リポジトリ: https://github.com/harunhhh/Multimodalsyatem
- 大容量バイナリ（`dict/`約100MB, `onnxruntime/`約325MB, `models/vvms/0.vvm`約55MB）はGitHubのファイルサイズ上限やプロジェクト規約（CLAUDE.md「大容量ファイルはGit管理外」）に従い `.gitignore` で除外
- 複数PCでの開発再現のため `requirements.txt`（`pip freeze`、CUDA版torch/torchvisionは除く）とセットアップ手順を記載した `README.md` を追加してプッシュ
- CUDA版torchはPyPIに存在しないため、README内で `--index-url https://download.pytorch.org/whl/cu128` を使った個別インストール手順を案内
- プロジェクトフォルダを `anaconsan/gazou/nidone_boushi_system/` から `anaconsan/Multimodalsyatem/`（gitリポジトリ直下）に統合・移動

## 5. 次にやること（未着手）

- F1（YOLO pose姿勢判定）の実装
- F1/F3の閾値の実測・調整（手首/鼻のy座標差、キーポイント信頼度、背景差分の変化割合）
- Gemini APIキーの取得
- F2（LLM対話）・F3（二度寝監視）・GUI（停止ボタン等）の実装
