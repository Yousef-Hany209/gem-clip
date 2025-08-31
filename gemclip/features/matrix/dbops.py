"""Database helper operations for the Matrix feature.

These functions operate on the Matrix window instance (`win`) to avoid
coupling DB concerns to the giant UI class. They preserve existing behavior.
"""
from __future__ import annotations

from typing import Any, Dict, Optional
import time

import customtkinter as ctk

from gemclip.infra import db
from i18n import tr


def ensure_db_session_tab(win: Any) -> None:
    try:
        if getattr(win, "_db_session_id", None) is None:
            name = f"Session {time.strftime('%Y-%m-%d %H:%M:%S')}"
            win._db_session_id = db.create_matrix_session(name)
        if getattr(win, "_db_tab_id", None) is None:
            try:
                tab_name = str(win._tabs[win._active_tab_index]["name"])  # type: ignore[index]
            except Exception:
                tab_name = "Default"
            win._db_tab_id = db.create_matrix_tab(win._db_session_id, tab_name, 0)
    except Exception:
        pass


def save_session_to_db(win: Any, name: Optional[str] = None) -> None:
    """Persist ALL open tabs into a DB session (atomic, simplified)."""
    try:
        ensure_db_session_tab(win)
        sess_id = getattr(win, "_db_session_id", None)
        if sess_id is None:
            return
        if name:
            db.rename_matrix_session(sess_id, name)

        def _persist_inputs(tab_id: int) -> None:
            for r_idx, item in enumerate(win.input_data):
                cid = None
                if item["type"] == "text":
                    cid = db.save_clipboard_item(item["data"], source="matrix_input")
                elif item["type"] in ("image", "image_compressed"):
                    b64 = item["data"]
                    if item["type"] == "image_compressed":
                        import zlib, base64 as _b64

                        try:
                            raw = zlib.decompress(_b64.b64decode(b64))
                            b64 = _b64.b64encode(raw).decode("utf-8")
                        except Exception:
                            pass
                    cid = db.save_clipboard_item({"type": "image", "data": b64}, source="matrix_input")
                elif item["type"] == "file":
                    cid = db.save_clipboard_item({"type": "file", "data": item["data"]}, source="matrix_input")
                db.upsert_matrix_input(tab_id, r_idx, cid, attach_files=None)

        existing_by_index: Dict[int, int] = {}
        try:
            rows = db.list_matrix_tabs(sess_id)
            for r in rows:
                try:
                    existing_by_index[int(r["order_index"]) ] = int(r["id"])  # type: ignore[index]
                except Exception:
                    continue
        except Exception:
            existing_by_index = {}

        for order_idx, tab in enumerate(win._tabs or []):  # type: ignore[attr-defined]
            tab_name = str((tab.get("name") if isinstance(tab, dict) else None) or f"Tab {order_idx+1}")
            if order_idx in existing_by_index:
                tab_id = existing_by_index[order_idx]
                try:
                    db.rename_matrix_tab(tab_id, tab_name)
                except Exception:
                    pass
            else:
                tab_id = db.create_matrix_tab(sess_id, tab_name, order_index=order_idx)
            _persist_inputs(tab_id)

            prompts_obj = (tab.get("prompts_obj") if isinstance(tab, dict) else {}) or {}
            if order_idx == getattr(win, "_active_tab_index", 0):
                try:
                    prompts_obj = {pid: (p.model_copy(deep=True) if hasattr(p, "model_copy") else type(p)(**p.model_dump())) for pid, p in win.prompts.items()}
                except Exception:
                    pass

            for c_idx, (_pid, pconf) in enumerate(prompts_obj.items()):
                params = getattr(getattr(pconf, "parameters", None), "model_dump", lambda **_: {})()
                snap = {
                    "name": getattr(pconf, "name", f"Prompt {c_idx+1}"),
                    "model": getattr(pconf, "model", "gemini-2.5-flash-lite"),
                    "system_prompt": getattr(pconf, "system_prompt", ""),
                    "parameters": params,
                }
                db.upsert_matrix_prompt(tab_id, c_idx, snap)

            num_rows = len(win.input_data)
            num_cols = len(prompts_obj)
            chk = []
            var_pack = tab.get("vars") if isinstance(tab, dict) else None
            if var_pack and isinstance(var_pack.get("checkbox_states"), list):
                chk = var_pack["checkbox_states"]
            else:
                st = tab.get("state") if isinstance(tab, dict) else None
                if st and isinstance(st.get("checkbox"), list):
                    chk = st["checkbox"]
            for r in range(num_rows):
                for c in range(num_cols):
                    flag = False
                    if r < len(chk) and isinstance(chk[r], list) and c < len(chk[r]):
                        v = chk[r][c]
                        try:
                            flag = bool(v.get()) if hasattr(v, "get") else bool(v)
                        except Exception:
                            flag = False
                    item = win.input_data[r]
                    cid = None
                    if item["type"] == "text":
                        cid = db.save_clipboard_item(item["data"], source="matrix_input")
                    elif item["type"] in ("image", "image_compressed"):
                        b64 = item["data"]
                        if item["type"] == "image_compressed":
                            import zlib, base64 as _b64

                            try:
                                raw = zlib.decompress(_b64.b64decode(b64))
                                b64 = _b64.b64encode(raw).decode("utf-8")
                            except Exception:
                                pass
                        cid = db.save_clipboard_item({"type": "image", "data": b64}, source="matrix_input")
                    elif item["type"] == "file":
                        cid = db.save_clipboard_item({"type": "file", "data": item["data"]}, source="matrix_input")
                    inp_id = db.upsert_matrix_input(tab_id, r, cid, attach_files=None)
                    p = list(prompts_obj.values())[c]
                    params = getattr(getattr(p, "parameters", None), "model_dump", lambda **_: {})()
                    snap = {
                        "name": getattr(p, "name", f"Prompt {c+1}"),
                        "model": getattr(p, "model", "gemini-2.5-flash-lite"),
                        "system_prompt": getattr(p, "system_prompt", ""),
                        "parameters": params,
                    }
                    prm_id = db.upsert_matrix_prompt(tab_id, c, snap)
                    db.upsert_matrix_cell(tab_id, inp_id, prm_id, flag)

        label = name or getattr(win, "_current_session_name", None) or ""
        if label:
            win._current_session_name = label
            try:
                win._session_label_var.set(f"{tr('matrix.toolbar.session_label')} {label}")
            except Exception:
                pass
    except Exception:
        # Non-fatal save errors are swallowed to avoid crashing the UI
        pass


def ensure_db_input_id(win: Any, r_idx: int) -> Optional[int]:
    try:
        if r_idx in getattr(win, "_db_input_ids", {}):
            return win._db_input_ids[r_idx]
        ensure_db_session_tab(win)
        if getattr(win, "_db_tab_id", None) is None:
            return None
        item = win.input_data[r_idx]
        cid = None
        if item["type"] == "text":
            cid = db.save_clipboard_item(item["data"], source="matrix_input")
        elif item["type"] in ("image", "image_compressed"):
            b64 = item["data"]
            if item["type"] == "image_compressed":
                import zlib, base64 as _b64

                try:
                    raw = zlib.decompress(_b64.b64decode(b64))
                    b64 = _b64.b64encode(raw).decode("utf-8")
                except Exception:
                    pass
            cid = db.save_clipboard_item({"type": "image", "data": b64}, source="matrix_input")
        elif item["type"] == "file":
            cid = db.save_clipboard_item({"type": "file", "data": item["data"]}, source="matrix_input")
        mid = db.upsert_matrix_input(win._db_tab_id, r_idx, cid, attach_files=None)
        win._db_input_ids[r_idx] = mid
        return mid
    except Exception:
        return None


def ensure_db_prompt_id(win: Any, c_idx: int, prompt_config: Any) -> Optional[int]:
    try:
        if c_idx in getattr(win, "_db_prompt_ids", {}):
            return win._db_prompt_ids[c_idx]
        ensure_db_session_tab(win)
        if getattr(win, "_db_tab_id", None) is None:
            return None
        snap = {
            "name": getattr(prompt_config, "name", f"Prompt {c_idx+1}"),
            "model": getattr(prompt_config, "model", "gemini-2.5-flash-lite"),
            "system_prompt": getattr(prompt_config, "system_prompt", ""),
            "parameters": getattr(getattr(prompt_config, "parameters", None), "model_dump", lambda **_: {})(),
            "enable_web": getattr(prompt_config, "enable_web", False),
        }
        pid = db.upsert_matrix_prompt(win._db_tab_id, c_idx, snap)
        win._db_prompt_ids[c_idx] = pid
        return pid
    except Exception:
        return None
