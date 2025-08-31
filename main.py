"""
main.py (Windows 専用・簡素化)
レガシーのエントリポイント。実装の重複を避けるため、
``python -m gemclip`` と同じランチャを呼び出します。
"""
from gemclip.__main__ import main as _gemclip_main


if __name__ == "__main__":
    _gemclip_main()
