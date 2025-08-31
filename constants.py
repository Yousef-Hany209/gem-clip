APP_NAME = "Gem Clip"
CONFIG_FILE = "config.json"
API_SERVICE_ID = "gemini_clip"
ICON_FILE = "icon.ico"
COMPLETION_SOUND_FILE = "completion.mp3"
DELETE_ICON_FILE = "delete_icon.png"  # 任意の16-24px程度のPNGを配置すると使用されます

# Token prices are read by runtime from api_price.json via agent logic.
TOKEN_PRICES = {}

# Supported model options for UI selection (id, label) — 外部ファイル models.json から読み込み
def _load_supported_models() -> list[tuple[str, str]]:
    import json, os
    here = os.path.dirname(__file__)
    path = os.path.join(here, "models.json")
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        pairs = []
        for m in data:
            mid = m.get("id")
            label = m.get("label") or mid
            if mid:
                pairs.append((mid, label))
        if pairs:
            return pairs
    except Exception:
        pass
    # フォールバック（ファイル未配置時）
    return [
        ("gemini-2.5-flash-lite", "gemini-2.5-flash-lite (高速、低精度)"),
        ("gemini-2.5-flash", "gemini-2.5-flash (普通)"),
        ("gemini-2.5-pro", "gemini-2.5-pro (低速、高精度)"),
    ]

SUPPORTED_MODELS: list[tuple[str, str]] = _load_supported_models()

def model_id_to_label(model_id: str) -> str:
    for mid, label in SUPPORTED_MODELS:
        if mid == model_id:
            return label
    return SUPPORTED_MODELS[0][1]

def model_label_to_id(label: str) -> str:
    # Option menu stores full label; split-safe mapping
    for mid, lbl in SUPPORTED_MODELS:
        if lbl == label:
            return mid
    # Fallback: if given an id-like string (e.g., from legacy config), accept it
    return label.split(" ")[0]
