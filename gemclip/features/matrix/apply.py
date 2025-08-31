"""Apply helpers to map loaded payloads into the Window state.

These helpers keep UI mutations localized while allowing the heavy
matrix window module to slim down.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional
import customtkinter as ctk

from .logic import build_prompts_dict, apply_results_to_state
from .utils import normalize_inputs_list
from i18n import tr


def set_ids_from_payload(win: Any, data: Dict[str, Any]) -> None:
    try:
        if data.get('tab_id') is not None:
            win._db_tab_id = int(data.get('tab_id'))
    except Exception:
        pass
    try:
        sid = data.get('session_id')
        if sid is not None:
            win._db_session_id = int(sid)
    except Exception:
        pass


def apply_inputs_prompts(win: Any, data: Dict[str, Any]) -> None:
    inputs = data.get('inputs') or []
    prompts_snap = data.get('prompts') or []
    win.input_data = normalize_inputs_list(inputs)
    win.prompts = build_prompts_dict(prompts_snap)


def reset_state_arrays(win: Any) -> None:
    win.checkbox_states = []
    win.results = []
    win._full_results = []
    win._row_summaries = []
    win._col_summaries = []
    win._result_textboxes = []
    win._cell_style = []


def apply_results_summaries(win: Any, data: Dict[str, Any]) -> None:
    # Results
    results = data.get('results') or []
    num_rows = len(win.input_data)
    num_cols = len(win.prompts)
    full, trunc = apply_results_to_state(results, num_rows, num_cols)
    win._full_results = full
    win.results = [
        [ctk.StringVar(value=trunc[r][c]) for c in range(num_cols)]
        for r in range(num_rows)
    ]
    # Row summaries
    row_sums = data.get('row_sums') or []
    if row_sums:
        win._row_summaries = [ctk.StringVar(value="") for _ in range(num_rows)]
        for r_idx, text in row_sums:
            if 0 <= r_idx < num_rows:
                win._row_summaries[r_idx].set(text)
        try:
            win._update_row_summary_column()
        except Exception:
            pass
    # Column summaries
    col_sums = data.get('col_sums') or []
    if col_sums:
        win._col_summaries = [ctk.StringVar(value="") for _ in range(num_cols)]
        for c_idx, text in col_sums:
            if 0 <= c_idx < num_cols:
                win._col_summaries[c_idx].set(text)
        try:
            win._update_column_summary_row()
        except Exception:
            pass
    # Final summary
    final_text = data.get('final_text')
    if final_text:
        try:
            win._update_matrix_summary_cell(final_text)
        except Exception:
            pass
    # Checked cells
    try:
        checks = data.get('checks') or []
        while len(win.checkbox_states) < num_rows:
            win.checkbox_states.append([])
        for r in range(num_rows):
            while len(win.checkbox_states[r]) < num_cols:
                win.checkbox_states[r].append(ctk.BooleanVar(value=False))
        for r_idx, c_idx in checks:
            if 0 <= r_idx < len(win.checkbox_states) and 0 <= c_idx < len(win.checkbox_states[r_idx]):
                win.checkbox_states[r_idx][c_idx].set(True)
    except Exception:
        pass


def finalize_canvas(win: Any) -> None:
    try:
        win.canvas.itemconfigure(win._window_id, window=win.scrollable_content_frame)
        win.after(1, lambda: win.canvas.configure(scrollregion=win.canvas.bbox("all")))
    except Exception:
        pass


def update_session_label(win: Any, data: Dict[str, Any]) -> None:
    try:
        sid = data.get('session_id')
        tname = data.get('tab_name') or ''
        tid = data.get('tab_id')
        if sid or tname or tid:
            label = f"DB #{sid} / {tname or 'Tab'} (#{tid})"
            win._current_session_name = label
            win._session_label_var.set(f"{tr('matrix.toolbar.session_label')} {label}")
    except Exception:
        pass

