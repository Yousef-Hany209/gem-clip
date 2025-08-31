"""Tab operations for Matrix window (click, DnD, rebuild)."""
from __future__ import annotations

from typing import Any, Optional
import customtkinter as ctk
from i18n import tr
from gemclip.core import Prompt


def rebuild_tabs(win: Any) -> None:
    win._tabs = [t for t in win._tabs if t.get('name') is not None]
    if not win._tabs:
        win._tabs = [{'name': tr('matrix.tab.default'), 'prompts_obj': {}, 'state': None}]
        win._active_tab_index = 0
    if win._active_tab_index >= len(win._tabs):
        win._active_tab_index = max(0, len(win._tabs) - 1)
    active = win._tabs[win._active_tab_index]
    prompts_obj = active.get('prompts_obj') if isinstance(active.get('prompts_obj', {}), dict) else win._deserialize_prompts(active.get('prompts', {}))
    win.prompts = {pid: (p.model_copy(deep=True) if hasattr(p, 'model_copy') else Prompt(**p.model_dump())) for pid, p in prompts_obj.items()} if prompts_obj else {}
    win.checkbox_states = []
    win.results = []
    win._full_results = []
    win._row_summaries = []
    win._col_summaries = []
    win._apply_state(active.get('state'))
    win._render_tabbar()


def on_tab_clicked(win: Any, idx: int) -> None:
    if not isinstance(win._tabs, list) or not win._tabs:
        return
    if idx < 0 or idx >= len(win._tabs):
        return
    if hasattr(win, '_active_tab_index') and idx == win._active_tab_index:
        return
    try:
        if 0 <= win._active_tab_index < len(win._tabs):
            win._tabs[win._active_tab_index]['prompts_obj'] = {pid: (p.model_copy(deep=True) if hasattr(p, 'model_copy') else Prompt(**p.model_dump())) for pid, p in win.prompts.items()}
            win._tabs[win._active_tab_index]['state'] = win._snapshot_state()
    except Exception:
        pass
    win._active_tab_index = idx
    try:
        if not (0 <= win._active_tab_index < len(win._tabs)):
            return
        t = win._tabs[win._active_tab_index]
        prompts_obj = t.get('prompts_obj') if isinstance(t.get('prompts_obj', {}), dict) else win._deserialize_prompts(t.get('prompts', {}))
        win.prompts = {pid: (p.model_copy(deep=True) if hasattr(p, 'model_copy') else Prompt(**p.model_dump())) for pid, p in prompts_obj.items()} if prompts_obj else {}
    except Exception:
        win.prompts = {}
    win._result_textboxes = []
    win._cell_style = []
    win.checkbox_states = []
    win.results = []
    win._full_results = []
    win._row_summaries = []
    win._col_summaries = []
    try:
        win._apply_state(win._tabs[win._active_tab_index].get('state'))
    except Exception:
        pass
    new_frame = ctk.CTkFrame(win.canvas, fg_color="transparent")
    win.scrollable_content_frame = new_frame
    try:
        win.canvas.itemconfigure(win._window_id, window=win.scrollable_content_frame)
    except Exception:
        pass
    win._update_ui()
    win._render_tabbar()


def on_tab_press(win: Any, event, idx: int) -> None:
    win._tab_drag = { 'start_idx': idx, 'current_idx': idx, 'moved': False, 'start_x': event.x_root }


def on_tab_motion(win: Any, event) -> None:
    if not getattr(win, '_tab_drag', None) or win._tab_drag.get('start_idx') is None:
        return
    try:
        if abs(int(event.x_root) - int(win._tab_drag.get('start_x', event.x_root))) > 3:
            win._tab_drag['moved'] = True
    except Exception:
        win._tab_drag['moved'] = True
    win._tab_drag['current_idx'] = win._compute_tab_drop_index(event.x_root)


def on_tab_release(win: Any, event) -> None:
    if not getattr(win, '_tab_drag', None) or win._tab_drag.get('start_idx') is None:
        return
    start = int(win._tab_drag.get('start_idx', 0))
    moved = bool(win._tab_drag.get('moved'))
    drop = win._compute_tab_drop_index(event.x_root)
    win._tab_drag = { 'start_idx': None, 'current_idx': None, 'moved': False }
    if not moved:
        on_tab_clicked(win, start)
        return
    try:
        if not (0 <= start < len(win._tabs)):
            return
        t = win._tabs.pop(start)
        if drop >= len(win._tabs):
            win._tabs.append(t)
            target = len(win._tabs) - 1
        else:
            win._tabs.insert(drop, t)
            target = drop
        on_tab_clicked(win, target)
    except Exception:
        pass
