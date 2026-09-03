"""パラメータ・閾値の一元管理（REQUIREMENTS.md 9-1 参照）"""

# --- カメラ ---
CAMERA_INDEX = 0

# --- F1: 姿勢認識（起床判定） ---
POSE_MODEL_PATH = "models/yolov8n-pose.pt"
POSTURE_CHECK_INTERVAL_SEC = 0.15   # 判定間隔
POSTURE_HOLD_REQUIRED_SEC = 3.0     # 累積保持でWOKENとなる秒数
POSTURE_KEYPOINT_CONF_THRESHOLD = 0.5   # 肩・腰・手首・鼻のconfidence閾値（要実測調整）
POSTURE_WRIST_ABOVE_NOSE_MARGIN_PX = 20  # 手首y座標が鼻y座標よりどれだけ小さければ「高い」とみなすか（要実測調整）
POSTURE_VERTICAL_ALIGN_MAX_RATIO = 0.5  # 【現在未使用】腰基準の旧判定用。布団の上で上体を起こすと誤判定したため廃止
POSTURE_SHOULDER_TILT_MAX_RATIO = 0.5   # 直立判定。肩のy差/肩幅の許容量（実測: 起床0.01〜0.06 / 就寝0.59〜17.47）

# --- F2: LLM対話 ---
GEMINI_API_TIMEOUT_SEC = 10
GEMINI_FALLBACK_TEXT = "おはようございます"
GEMINI_RELAPSE_FALLBACK_TEXT = "二度寝してましたよ。気をつけましょう。"
VOICEVOX_STYLE_ID = 5  # ずんだもん（ノーマル）

# --- F3: 二度寝監視 ---
FUTON_CHECK_INTERVAL_SEC = 1.0
FUTON_CHANGE_RATIO_THRESHOLD = 0.15  # ROI内の変化ピクセル割合閾値（暫定15〜20%、要実測調整）
FUTON_CHANGE_CONFIRM_SEC = 10.0      # 変化が継続したら確定判定するまでの秒数
MONITORING_TIMEOUT_SEC = 60 * 60     # 1時間変化なしでSTANDBYへ

# --- F4: 異常系 ---
ALARM_VOLUME_UP_INTERVAL_SEC = 30
CAMERA_STALE_TIMEOUT_SEC = 5.0  # フレーム更新が止まってから異常とみなすまでの秒数

# --- アラーム音 ---
ALARM_NORMAL_SOUND_PATH = "assets/sounds/alarm_normal.wav"
ALARM_RELAPSE_SOUND_PATH = "assets/sounds/alarm_relapse.wav"
ALARM_BASE_VOLUME = 0.3
ALARM_VOLUME_STEP = 0.2
ALARM_MAX_VOLUME = 1.0

# --- アラーム時刻（暫定・後で設定UI等に置き換え想定） ---
ALARM_HOUR = 7
ALARM_MINUTE = 0
