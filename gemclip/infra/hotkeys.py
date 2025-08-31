"""Windows global hotkey manager using RegisterHotKey.

Encapsulates registration and message loop handling. Callers provide a
dispatch callable to marshal callbacks to the UI thread when available.
"""
from __future__ import annotations

from typing import Callable, Dict, List, Optional, Tuple
import threading

import ctypes
from ctypes import wintypes


class WindowsHotkeyManager:
    def __init__(self, dispatch: Callable[[Callable], None]):
        self._dispatch = dispatch
        self._thread: Optional[threading.Thread] = None
        self._user32 = None
        self._id_map: Dict[int, Callable] = {}
        self._registrations: List[Tuple[int, int, Callable]] = []

    def parse_hotkey(self, hotkey_str: Optional[str]) -> Optional[Tuple[int, int]]:
        if not hotkey_str:
            return None
        mods = 0
        parts = [p.strip() for p in hotkey_str.lower().split('+')]
        key_part = parts[-1]
        for mod in parts[:-1]:
            if mod == 'ctrl':
                mods |= 0x0002  # MOD_CONTROL
            elif mod == 'shift':
                mods |= 0x0004  # MOD_SHIFT
            elif mod == 'alt':
                mods |= 0x0001  # MOD_ALT
            elif mod == 'win':
                mods |= 0x0008  # MOD_WIN
        if len(key_part) == 1:
            vk = ord(key_part.upper())
        elif key_part.startswith('f') and key_part[1:].isdigit():
            fn = int(key_part[1:])
            vk = 0x70 + (fn - 1)  # F1 = 0x70
        else:
            return None
        return mods, vk

    def register_hotkeys(self, specs: List[Tuple[str, Callable]]) -> None:
        """Register multiple hotkeys from (hotkey_string, callback) pairs."""
        # Build spec list
        regs: List[Tuple[int, int, Callable]] = []
        for s, cb in specs:
            parsed = self.parse_hotkey(s)
            if parsed:
                regs.append((parsed[0], parsed[1], cb))
        if not regs:
            # Nothing to register; ensure prior state is cleared
            self.unregister_all()
            return
        # Replace registrations and start thread
        self._registrations = regs
        self._start_thread()

    def unregister_all(self) -> None:
        if self._thread and self._thread.is_alive():
            try:
                user32 = ctypes.windll.user32
                user32.PostThreadMessageW(self._thread.ident, 0x0012, 0, 0)  # WM_QUIT
            except Exception:
                pass
            try:
                self._thread.join(timeout=1.5)
            except Exception:
                pass
        # Best-effort explicit unregister
        try:
            user32 = ctypes.windll.user32
            for hid in list(self._id_map.keys()):
                try:
                    user32.UnregisterHotKey(None, hid)
                except Exception:
                    pass
        except Exception:
            pass
        self._thread = None
        self._user32 = None
        self._id_map.clear()

    # Internal
    def _start_thread(self) -> None:
        self.unregister_all()
        self._thread = threading.Thread(target=self._message_loop, daemon=True)
        self._thread.start()

    def _message_loop(self) -> None:
        user32 = ctypes.windll.user32
        self._user32 = user32
        # Register all
        self._id_map.clear()
        next_id = 1
        for mods, vk, cb in self._registrations:
            hid = next_id
            next_id += 1
            try:
                if not user32.RegisterHotKey(None, hid, mods, vk):
                    continue
                self._id_map[hid] = cb
            except Exception:
                pass
        msg = wintypes.MSG()
        while True:
            result = user32.GetMessageW(ctypes.byref(msg), None, 0, 0)
            if result == 0:
                break
            if msg.message == 0x0312:  # WM_HOTKEY
                cb = self._id_map.get(msg.wParam)
                if cb:
                    try:
                        self._dispatch(cb)
                    except Exception:
                        try:
                            cb()
                        except Exception:
                            pass
            user32.TranslateMessage(ctypes.byref(msg))
            user32.DispatchMessageW(ctypes.byref(msg))

