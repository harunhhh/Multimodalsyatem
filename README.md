# Multimodalsyatem

Webカメラで「本当に起きたか」を見張る目覚まし。ボタンでは止められず、**カメラの前で手を上げる姿勢を3秒続けないとアラームが止まらない**。さらに起床後も布団を監視し続け、二度寝したら専用の音で鳴り直す。

アラームが止まるとGeminiが挨拶文を生成し、VOICEVOX（ずんだもん）が読み上げる。二度寝から起きた時は内容が変わる。

以下3つの機能で構成される。

| 機能 | 内容 | 使用技術 |
|---|---|---|
| F1 | 姿勢認識による起床判定 | YOLOv8 pose |
| F2 | LLMによる対話生成と読み上げ | Gemini API + VOICEVOX |
| F3 | 二度寝の常時監視 | OpenCVの背景差分 |

- 仕様の詳細・状態遷移図 → [REQUIREMENTS.md](REQUIREMENTS.md)
- 開発の経緯 → [CONVERSATION_LOG.md](CONVERSATION_LOG.md)

---

## 必要なもの

| | 条件 |
|---|---|
| OS | **Windows**（`winsound`とVOICEVOXのWindows版wheelを使うため、他OSでは動かない） |
| カメラ | Webカメラ1台。**上半身が映れば十分**で、全身を映す必要はない |
| スピーカー | アラーム音と音声合成の再生に必要 |
| GPU | NVIDIA GPUがあれば速いが、**無くても動く**（CPUで動作する） |
| Python | 3.10（VOICEVOXのwheelが`cp310`指定のため、このバージョンで固定） |
| ネット接続 | Gemini APIの呼び出しに必要。無い場合は定型文にフォールバックする |

---

## セットアップ

### 1. 仮想環境を作る

このプロジェクト専用の仮想環境を作る。名前は任意（以下は `myenv` の例）。

```bash
conda create -n myenv python=3.10 -y
conda activate myenv
```

**既存の環境を流用しないこと。**このプロジェクトはCUDA版のPyTorchを特定のバージョンで入れるため、
他の用途で使っている環境に混ぜると、そちらのPyTorchが上書きされて壊れる。

### 2. PyTorchを先に入れる

CUDA版はPyPIに無いため、`requirements.txt` とは別に専用インデックスから入れる。

```bash
pip install torch==2.11.0 torchvision --index-url https://download.pytorch.org/whl/cu128
```

GPUが無い、またはCUDA環境が無いPCでは、代わりに `pip install torch torchvision` でよい（動作はするが遅くなる）。

### 3. 残りのパッケージを入れる

```bash
pip install -r requirements.txt
```

VOICEVOX本体（`voicevox_core`）は `requirements.txt` の中でGitHub Releaseのwheelを直接参照しているので、追加の操作は不要。

### 4. 大容量ファイルを配置する

**この3つはリポジトリに含まれていない。**合計約480MBあり、GitHubのファイルサイズ上限を超えるため `.gitignore` で除外している。

```
Multimodalsyatem/
├── dict/open_jtalk_dic_utf_8-1.11/   # 日本語の読み方辞書（約100MB）
├── onnxruntime/                       # 音声合成の推論エンジン（約325MB）
└── models/vvms/0.vvm                  # ずんだもんの音声モデル（約55MB）
```

いずれも**VOICEVOX Coreが動作するために必要なデータ**で、Pythonパッケージ（`voicevox_core`）とは別に配布されている。

**入手方法:** [VOICEVOX Core 0.16.4のリリースページ](https://github.com/VOICEVOX/voicevox_core/releases/tag/0.16.4)から、Windows用のダウンローダー `download-windows-x64.exe` を取得して実行する。ONNX Runtime・音声モデル・辞書を一式ダウンロードするツール。**正確なオプションは公式READMEを参照すること**（本ドキュメントの著者は未実行）。

`requirements.txt` が指定している `voicevox_core` は **0.16.4** なので、それに合うバージョンを選ぶこと。実行時のログに `Loaded ONNX Runtime dylib with version '1.17.3'` と出れば正しく読み込めている。

**これらが無いと音声合成の段階で落ちる。**逆に言えば、姿勢判定や二度寝監視だけを試すなら無くても進められる。

### 5. 姿勢推定モデル

`models/yolov8n-pose.pt`（約6.8MB）もリポジトリに含まれていないが、**こちらは自動でダウンロードされる**ので何もしなくてよい。[modules/posture.py](modules/posture.py) が、ファイルが無い場合はultralyticsの標準モデル名として扱うため。

### 6. Gemini APIキーを設定する

[Google AI Studio](https://aistudio.google.com/apikey) でAPIキーを取得し、プロジェクト直下に `.env` を作る。

```
GEMINI_API_KEY=取得したキー
```

[.env.example](.env.example) がテンプレート。`.env` は `.gitignore` に入っているのでコミットされない。

**キーが無くても動く。**その場合はGeminiの呼び出しが失敗し、定型文（「おはようございます」等）にフォールバックする。挨拶が短い定型文になっていたら、キーの設定を疑うとよい。

### 7. 動作確認

```bash
python modules/voice.py
```

`output.wav` と `output_2.wav` が生成されれば、VOICEVOX周りは正常。

---

## 実行する

### カメラの置き方

**ここを外すと動かないので、実行前に決めておく。**次の3つを同時に満たす位置に固定する。

1. **布団が画角に入る** — 二度寝監視の対象範囲（ROI）を布団に取るため
2. **布団の上で上体を起こしたときに、上半身が映る** — アラームを止める姿勢を認識するため
3. **画角から完全に外れられる場所がある** — 監視開始には、いったん誰も映っていない状態が必要なため（後述）

ベッドの足元側や斜め前方から、布団全体と枕元が入るように置くとよい。一度決めたら動かさないこと。基準フレームとの差分で判定するため、カメラが動くと誤検知する。

### 起動

```bash
python main.py
```

**アラームが鳴るまで起動したままにしておく。**閉じると待機も止まる。

動作の記録は **`nidone.log`**（プロジェクト直下）に追記される。画面にも同じ内容が出る。うまくいかない時はこのファイルを見る。

### 起動時の操作

**アラーム時刻を入力**

```
アラーム時刻を入力 (HH:MM) [既定 07:00]: 07:00
```

空のままEnterを押すと [config.py](config.py) の既定値が使われる。動作確認したい時は数分後の時刻を入れるとよい。

> **アラームは指定した「分」の間にしか鳴らない。**その時刻をまたいでから起動した場合は、翌日の同時刻まで待つことになる。PCがスリープしていた場合も鳴らないので、電源設定に注意する。

**布団の範囲を指定**

`Select futon area` というウィンドウにカメラ映像が出るので、**布団の範囲をマウスでドラッグして Enter**。やり直したい時は `r`、中止は `q`。ここで指定した範囲（ROI）の中だけを二度寝監視の対象にする。

範囲の指定はここでしかできない。**取り直したい場合はプログラムを終了して起動しなおす。**

### 動作の流れ

```
STANDBY（待機）
   │  指定時刻になると自動で
   ▼
ALARM_RINGING → POSTURE_HOLDING     ← 手を上げて3秒キープ
   │
   ▼
WOKEN → GREETING                    ← Gemini + VOICEVOXで挨拶（15秒ほど）
   │
   ▼
MONITORING（二度寝監視）
   │  布団の変化が10秒続くと
   ▼
RELAPSE_ALARM → 最初のアラームに戻る（専用の音・累積時間はリセット）
```

### 操作するポイントは3つだけ

**① アラームを止める**

カメラの前で**手を上げる**。判定条件は3つ。

1. 左右どちらかの手首が、鼻より20px以上高い位置にあること
2. 肩のラインが水平に近いこと（＝起き上がっている）
3. 鼻と両肩がはっきり写っていること

**布団の上で上体を起こすだけでよく、立ち上がる必要はない。**腰は判定に使わないので、全身が映っていなくても成立する。

累積3秒で成立する。途中で姿勢が崩れてもリセットされず、足し算される。

**② 挨拶の後、一度カメラの画角から外れる**

画面に赤字で `WAITING: leave the camera view` と出る。

**ここが最もつまずきやすい。**二度寝監視は「誰もいない状態の布団」を基準として撮影してから始まるため、**画角から完全に消えるまで監視は開始されない。**布団に戻ってしまうと、いつまで経っても始まらない。

外れるとログに `布団の基準フレームを記録しました。監視を開始します。` が出る。

**③ 止めたい時はSTOPボタン**

画面右上の赤い `STOP` をクリックする。押した時の動作は状態によって変わる。

| 状態 | STOPを押すと |
|---|---|
| 待機中（STANDBY） | **プログラムが終了する** |
| アラーム中・監視中 | アラームを止めて待機状態に戻る |

つまり**完全に終了したい時は、STOPを2回押す**（アラームを止めて待機に戻す → 待機中にもう一度押して終了）。

ターミナルで `Ctrl+C` を押しても、カメラを解放して正常終了する（[main.py](main.py) が `KeyboardInterrupt` を捕まえて後始末をする）。

**ウィンドウを閉じるだけでは止まらない。**OpenCVのウィンドウを閉じてもプログラムは動き続け、カメラを掴んだままになる。この状態で起動しなおすと「フレーム取得に失敗しました」となる。対処は「うまくいかない時」を参照。

---

## ファイルの説明

### 実行するファイル

| ファイル | 役割 |
|---|---|
| [main.py](main.py) | **本体。**状態遷移を管理する。通常はこれを実行する |
| [config.py](config.py) | 閾値・パラメータの一元管理。調整はすべてここで行う |

### 機能ごとのモジュール

| ファイル | 役割 |
|---|---|
| [modules/posture.py](modules/posture.py) | F1: YOLO poseによる姿勢判定。`check_posture_debug()` は判定の内訳も返す |
| [modules/llm_client.py](modules/llm_client.py) | F2: Geminiによる挨拶文の生成。失敗時は定型文を返す |
| [modules/voice.py](modules/voice.py) | F2: VOICEVOXによる音声合成と再生 |
| [modules/futon_monitor.py](modules/futon_monitor.py) | F3: ROIの選択と、背景差分による布団の変化検出 |
| [modules/alarm.py](modules/alarm.py) | アラーム音のループ再生と音量制御 |
| [modules/ui.py](modules/ui.py) | 映像・状態表示・STOPボタンの描画 |

### 調整・確認用のスクリプト

いずれも単体で実行でき、**main.pyを動かさずに個別の機能だけ試せる。**

| ファイル | 用途 |
|---|---|
| [test_posture.py](test_posture.py) | 姿勢判定の調整。骨格と各条件のOK/NGを表示する。`u`/`d`キーでラベル付きサンプルを収集し `posture_samples.log` に保存できる |
| [test_futon_monitor.py](test_futon_monitor.py) | 二度寝監視の調整。`change_ratio` をリアルタイム表示する |
| [test_roi.py](test_roi.py) | 保存済みのスナップショット画像でROI選択を試す |
| [test_full_flow.py](test_full_flow.py) | アラーム時刻の待機を飛ばして、状態遷移を通しで試す |
| [capture_snapshot.py](capture_snapshot.py) | カメラ映像から静止画を保存する（`assets/snapshots/`） |

> **注意:** これらは pytest ではなく、**目視で確認する手動スクリプト**。`pytest` では動かない。

---

## 調整できる値

すべて [config.py](config.py) にまとまっている。実測で妥当性を確認済みの値には根拠を併記した。

### 姿勢判定（F1）

| 項目 | 既定値 | 説明 |
|---|---|---|
| `POSTURE_HOLD_REQUIRED_SEC` | 3.0 | 起床成立までの累積秒数 |
| `POSTURE_WRIST_ABOVE_NOSE_MARGIN_PX` | 20 | 手首が鼻よりどれだけ高ければ「上げた」とみなすか |
| `POSTURE_KEYPOINT_CONF_THRESHOLD` | 0.5 | 鼻・両肩の検出信頼度の下限 |
| `POSTURE_SHOULDER_TILT_MAX_RATIO` | 0.5 | 肩の傾きの許容量。**実測で起床時0.01〜0.06 / 就寝時0.59〜17.47** と大きく離れており、余裕がある |
| `POSTURE_VERTICAL_ALIGN_MAX_RATIO` | 0.5 | 腰基準の旧判定用。**現在は使われていない** |

### 二度寝監視（F3）

| 項目 | 既定値 | 説明 |
|---|---|---|
| `FUTON_CHANGE_RATIO_THRESHOLD` | 0.15 | 二度寝とみなすROI内の変化割合。**実測で不在時0.00〜0.08 / 布団に戻ると0.31〜0.73** |
| `FUTON_CHANGE_CONFIRM_SEC` | 10.0 | この秒数だけ変化が続いたら確定。一瞬横切った程度では発火しない |
| `MONITORING_TIMEOUT_SEC` | 3600 | 変化が無いまま経過したら待機に戻る |

### アラーム・カメラ

| 項目 | 既定値 | 説明 |
|---|---|---|
| `ALARM_HOUR` / `ALARM_MINUTE` | 7 / 0 | 起動時の入力を空にした場合の既定時刻 |
| `ALARM_BASE_VOLUME` | 0.3 | 開始音量 |
| `ALARM_VOLUME_UP_INTERVAL_SEC` | 30 | この秒数ごとに音量が上がる |
| `ALARM_VOLUME_STEP` | 0.2 | 1回あたりの上げ幅（上限1.0） |
| `CAMERA_INDEX` | 0 | カメラが複数ある場合は1、2…と変える |

---

## うまくいかない時

### カメラが開けない / フレーム取得に失敗する

```
videoio(MSMF): can't grab frame. Error: -1072875772
```

**別のプロセスがカメラを掴んでいる可能性が高い。**このシステムは起動から終了までカメラを保持し続ける仕様のため、**main.pyは同時に1つしか起動できず、実行中は他のアプリからもカメラが使えない。**

残っているプロセスを確認する。

```powershell
Get-CimInstance Win32_Process -Filter "Name='python.exe'" | Select-Object ProcessId, CommandLine
```

`python main.py` が残っていたら終了させる。Teams等のビデオ会議アプリが起動している場合も同様。

### 画面の文字が化ける

`cv2.putText` は**日本語を描画できない。**画面に出す文字列は英字で書くこと。日本語を書くと文字化けし、案内が読めなくなる。

### 姿勢を取ってもアラームが止まらない

[test_posture.py](test_posture.py) を単体で動かし、3条件のどれが赤いかを見る。

- `wrist_above_nose` が赤 → 手が上がりきっていない。`POSTURE_WRIST_ABOVE_NOSE_MARGIN_PX` を下げる
- `is_upright` が赤 → 表示される `ratio` を見て `POSTURE_SHOULDER_TILT_MAX_RATIO` と比べる
- `conf_ok` が赤 → 暗い、または顔や肩が隠れている。照明かカメラ位置を見直す

### 二度寝監視が始まらない

**カメラの画角から完全に外れていない。**人物が検出されなくなるまで基準フレームは撮られない。画面の `WAITING: leave the camera view` が消えて、ログに `布団の基準フレームを記録しました` が出るのを確認する。

### 挨拶が短い定型文になる

Geminiの呼び出しに失敗している。`.env` の `GEMINI_API_KEY` を確認する。定型文は [config.py](config.py) の `GEMINI_FALLBACK_TEXT`（「おはようございます」）と `GEMINI_RELAPSE_FALLBACK_TEXT`。生成に成功していれば、もっと長い文がログに出る。

### CUDAのDLLエラーが出る

```
LoadLibrary failed with error 126 ... voicevox_onnxruntime_providers_cuda.dll
```

**無視してよい。**VOICEVOXがGPUを試して失敗し、CPUにフォールバックしているだけ。続けて `CPUを利用します` と出ていれば正常に動作している。

### 音声合成が遅い

初回はVOICEVOXの初期化に10〜15秒かかる。2回目以降は速くなる。

---

## 既知の制約

- **Windows専用。**`winsound` とVOICEVOXのwheelに依存している
- **カメラを占有する。**実行中は他のアプリでカメラを使えない
- **単眼カメラのため、布団の「高さ」は測れない。**ROI内の変化ピクセル割合という面積ベースの近似指標で代用している
- **照明の変化に弱い。**基準フレームとの差分で判定するため、部屋の明かりを点けると誤検知する可能性がある
- 実際の就寝から起床までを通した長時間の運用テストは未実施
