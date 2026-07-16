# Multimodalsyatem

マルチモーダル対話システムを利用した二度寝防止システム。

詳細な要件定義は [REQUIREMENTS.md](REQUIREMENTS.md) を参照。

## セットアップ（別PCで開発を始めるとき）

### 1. 仮想環境の作成

```bash
conda create -n nidone python=3.10 -y
conda activate nidone
```

### 2. CUDA版 PyTorch のインストール（先に単独で）

PyPIには CUDA 版 wheel が無いため、`requirements.txt` とは別に専用インデックスから入れる。

```bash
pip install torch==2.11.0 torchvision --index-url https://download.pytorch.org/whl/cu128
```

GPUがない/CUDA未整備のPCでは、上記を省略して通常の `pip install torch torchvision` でCPU版にしても動作は可能（速度は落ちる）。

### 3. 残りのパッケージをインストール

```bash
pip install -r requirements.txt
```

`voicevox_core` は requirements.txt 内でGitHub Releaseの wheel を直接参照しているため、追加設定不要。

### 4. 大容量バイナリの配置（Git管理外）

以下はサイズの都合でリポジトリに含めていない。`11kai/`（授業フォルダ）または配布された資産から手動でコピーする。

```
Multimodalsyatem/
├── dict/open_jtalk_dic_utf_8-1.11/   # OpenJTalk辞書（約100MB）
├── onnxruntime/                       # VOICEVOX Core依存（約325MB）
└── models/vvms/0.vvm                  # ずんだもんモデル（約55MB）
```

### 5. 動作確認

```bash
python modules/voice.py
```

`output.wav` / `output_2.wav` が生成されれば成功。
