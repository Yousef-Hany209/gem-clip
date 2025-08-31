"""
config_manager.py（Windows 専用・簡素化）

現行スキーマ（v8）固定で設定を読み書きします。複雑な移行やレガシー名の探索は行いません。
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional, Dict, Any

from gemclip.core import AppConfig
from paths import get_config_file_path


def _read_json(path: Path) -> Optional[dict]:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def _write_json(path: Path, data: dict) -> bool:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        return True
    except Exception:
        return False


def _default_config_dict() -> Dict[str, Any]:
    return {
        "version": 8,
        "prompts": {
            "check": {
                "name": "誤字脱字を修正",
                "model": "gemini-2.5-flash",
                "system_prompt": (
                    "あなたはプロの編集者です。以下のテキストに含まれる誤字、脱字、文法的な誤りを修正し、"
                    "自然で読みやすい文章にしてください。内容は変更せず、修正後のテキストのみを返してください。"
                ),
                "parameters": {"temperature": 0.2},
            },
            "summarize": {
                "name": "複雑な内容を箇条書きで要約",
                "model": "gemini-2.5-flash",
                "system_prompt": (
                    "以下の専門的なテキストの要点を抽出し、簡潔に最大5つまでの箇条書きで要約してください。"
                ),
                "parameters": {"temperature": 0.5},
            },
            "ocr": {
                "name": "画像からテキストを抽出",
                "model": "gemini-2.5-flash",
                "system_prompt": (
                    "この画像に含まれるテキストを正確に読み取り、そのまま出力してください。"
                    "テキストの構造やレイアウトも可能な限り保持してください。"
                ),
                "parameters": {"temperature": 0.1},
            },
        },
        "max_history_size": 20,
        "hotkey_prompt_list": "ctrl+shift+c",
        "hotkey_refine": "ctrl+shift+r",
        "hotkey_matrix": None,
        "hotkey_free_input": None,
        "matrix_row_summary_prompt": None,
        "matrix_col_summary_prompt": None,
        "matrix_matrix_summary_prompt": None,
        "max_flow_steps": 5,
        "language": "auto",
        "theme_mode": "system",
        # Pricing display currency
        "display_currency": "USD",
        "usd_to_display_rate": 1.0,
    }


def _normalize_to_current_schema(data: Dict[str, Any]) -> Dict[str, Any]:
    """入力データに必要キーがなければ補完し、現行v8スキーマへ正規化します。"""
    base = _default_config_dict()
    # 既存値を優先して上書き
    for k, v in data.items():
        base[k] = v
    # レガシー単一ホットキーがあれば prompt_list に反映
    if base.get("hotkey") and not base.get("hotkey_prompt_list"):
        base["hotkey_prompt_list"] = base.get("hotkey")
    base["version"] = 8
    return base


def load_config() -> Optional[AppConfig]:
    """設定を読み込み、存在しなければデフォルトを作成。常に現行スキーマで返す。"""
    config_path: Path = get_config_file_path()
    if not config_path.exists():
        _write_json(config_path, _default_config_dict())
    data = _read_json(config_path) or {}
    norm = _normalize_to_current_schema(data)
    # 外部プロンプト定義の取り込み（既存キーは尊重）
    try:
        _merge_prompts_from_dir(norm)
    except Exception:
        pass
    # 保存してから返す（将来の起動で安定化）
    _write_json(config_path, norm)
    try:
        return AppConfig(**norm)
    except Exception:
        # バリデーション問題時はデフォルトを返す
        default = _default_config_dict()
        _write_json(config_path, default)
        return AppConfig(**default)


def save_config(config: AppConfig) -> None:
    data = config.model_dump(by_alias=True, exclude_none=False)
    data["version"] = 8
    _write_json(get_config_file_path(), data)


def create_default_config() -> None:
    _write_json(get_config_file_path(), _default_config_dict())


# ----------------------- helpers -----------------------
def _merge_prompts_from_dir(config_dict: Dict[str, Any]) -> None:
    """`prompts/*.json` を読み込み、config.prompts に未定義のキーだけ追加する。

    ファイル名（拡張子除く）をキーとして用いる。値は Prompt 互換の辞書を想定。
    """
    import os, json
    here = Path(__file__).parent
    pdir = here / "prompts"
    if not pdir.exists() or not pdir.is_dir():
        return
    prompts = config_dict.get("prompts")
    if not isinstance(prompts, dict):
        prompts = {}
        config_dict["prompts"] = prompts
    for name in os.listdir(pdir):
        if not name.lower().endswith('.json'):
            continue
        key = os.path.splitext(name)[0]
        if key in prompts:
            continue
        try:
            with open(pdir / name, 'r', encoding='utf-8') as f:
                data = json.load(f)
            if isinstance(data, dict) and data.get('name') and data.get('model') and data.get('system_prompt'):
                prompts[key] = data
        except Exception:
            continue
