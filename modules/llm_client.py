"""F2: Gemini APIによる挨拶文生成。

APIキーは環境変数 GEMINI_API_KEY から取得する（.envファイルがあれば自動読み込み）。
タイムアウト/エラー時は config.GEMINI_FALLBACK_TEXT にフォールバックする。
"""

import os

import google.generativeai as genai
from dotenv import load_dotenv

import config

load_dotenv()

_PROMPT_NORMAL = (
    "あなたは目覚まし用の対話アシスタントです。"
    "ユーザーがちょうど起床したところです。短く元気な朝の挨拶を一言、日本語で生成してください。"
)
_PROMPT_RELAPSE = (
    "あなたは目覚まし用の対話アシスタントです。"
    "ユーザーは一度起きたものの、その後布団に戻って二度寝してしまい、今また起床したところです。"
    "二度寝していたことを指摘しつつ、気をつけるよう軽く注意する一言を、日本語で短く生成してください。"
)

_model = None


def _get_model():
    global _model
    if _model is None:
        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            raise RuntimeError("GEMINI_API_KEY 環境変数が設定されていません")
        genai.configure(api_key=api_key)
        _model = genai.GenerativeModel("gemini-flash-latest")
    return _model


def generate_greeting(is_relapse: bool = False) -> str:
    """起床フラグをGeminiに伝え、朝の挨拶文を生成する。

    is_relapse=Trueの場合は、二度寝から起きたことを踏まえた注意喚起の一言を生成する。
    タイムアウト/エラー時はconfig.GEMINI_FALLBACK_TEXT（またはconfig.GEMINI_RELAPSE_FALLBACK_TEXT）にフォールバックする。
    """
    prompt = _PROMPT_RELAPSE if is_relapse else _PROMPT_NORMAL
    fallback = config.GEMINI_RELAPSE_FALLBACK_TEXT if is_relapse else config.GEMINI_FALLBACK_TEXT
    try:
        model = _get_model()
        response = model.generate_content(
            prompt,
            request_options={"timeout": config.GEMINI_API_TIMEOUT_SEC},
        )
        text = response.text.strip()
        return text if text else fallback
    except Exception:
        return fallback
