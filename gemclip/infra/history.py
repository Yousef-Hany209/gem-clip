"""Clipboard history service for Windows.

Collects clipboard changes (text/images/files), normalizes entries,
deduplicates, persists to DB, and publishes updates via a callback.
"""
from __future__ import annotations

from typing import Any, Callable, List, Optional
import threading
import time
import base64
import hashlib
from io import BytesIO
from pathlib import Path

from PIL import Image, ImageGrab
import pyperclip

from gemclip.infra import db


class HistoryService:
    def __init__(self, max_size: int, on_updated: Callable[[List[Any]], None]):
        self._items: List[Any] = []
        self._max_size = max_size
        self._on_updated = on_updated
        self._running = False
        self._thread: Optional[threading.Thread] = None

    # Public API
    @property
    def items(self) -> List[Any]:
        return list(self._items)

    def start(self) -> None:
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(target=self._monitor, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._running = False
        if self._thread and self._thread.is_alive():
            try:
                self._thread.join(timeout=1.0)
            except Exception:
                pass

    def add(self, content: Any) -> None:
        norm = self._normalize(content)
        if norm is None:
            return
        # remove duplicate
        try:
            for i, it in enumerate(self._items):
                if it == norm:
                    self._items.pop(i)
                    break
        except Exception:
            pass
        self._items.insert(0, norm)
        # persist to DB (idempotent via hash in db layer)
        try:
            db.save_clipboard_item(norm, source="user_clipboard")
        except Exception:
            pass
        if len(self._items) > self._max_size:
            self._items = self._items[: self._max_size]
        self._publish()

    # Internal
    def _publish(self) -> None:
        try:
            self._on_updated(self.items)
        except Exception:
            pass

    def _monitor(self) -> None:
        last_sig: Optional[str] = None
        while self._running:
            try:
                items = None
                sig = None
                try:
                    clip = ImageGrab.grabclipboard()
                except Exception:
                    clip = None
                if isinstance(clip, Image.Image):
                    im = clip.convert('RGB') if clip.mode != 'RGB' else clip
                    buf = BytesIO()
                    im.save(buf, format='PNG')
                    img_bytes = buf.getvalue()
                    try:
                        import zlib
                        comp = zlib.compress(img_bytes)
                        encoded = base64.b64encode(comp).decode('utf-8')
                        items = [{"type": "image_compressed", "data": encoded}]
                    except Exception:
                        encoded = base64.b64encode(img_bytes).decode('utf-8')
                        items = [{"type": "image", "data": encoded}]
                    sig = "img:" + hashlib.sha1(img_bytes).hexdigest()
                elif isinstance(clip, list):
                    paths = [p for p in clip if isinstance(p, str)]
                    if paths:
                        items = [{"type": "file", "data": p} for p in paths]
                        sig = "files:" + "|".join(paths)
                if items is None:
                    try:
                        txt = pyperclip.paste()
                    except Exception:
                        txt = ""
                    if txt:
                        items = [{"type": "text", "data": txt}]
                        sig = "text:" + hashlib.sha1(txt.encode('utf-8')).hexdigest()
                if items and sig != last_sig:
                    for it in items:
                        self.add(it)
                    last_sig = sig
                time.sleep(0.5)
            except Exception:
                time.sleep(1)

    def _normalize(self, content: Any) -> Optional[Any]:
        if isinstance(content, str):
            s = content.strip()
            return s if s else None
        if isinstance(content, dict) and "type" in content and "data" in content:
            return content
        return None
