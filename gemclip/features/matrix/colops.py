"""Column operations for Matrix window (delete, reorder helpers)."""
from __future__ import annotations

from pathlib import Path
import customtkinter as ctk
from tkinter import messagebox

from i18n import tr
import styles
from gemclip.core import DELETE_ICON_FILE


def delete_column(win, col_idx: int) -> None:
    col_letter = chr(ord("A") + col_idx)
    if not messagebox.askyesno(
        tr("matrix.delete_col_title"),
        f"{tr('matrix.delete_col_confirm_fmt', col=col_letter)}\n{tr('common.cannot_undo')}",
    ):
        return
    try:
        for widget in list(win.scrollable_content_frame.grid_slaves(column=col_idx + 1)):
            widget.destroy()
    except Exception:
        pass

    def finalize_delete():
        try:
            prompt_keys = list(win.prompts.keys())
            if 0 <= col_idx < len(prompt_keys):
                del win.prompts[prompt_keys[col_idx]]
            for r_idx in range(len(win.input_data)):
                if r_idx < len(win.checkbox_states) and 0 <= col_idx < len(win.checkbox_states[r_idx]):
                    win.checkbox_states[r_idx].pop(col_idx)
                if r_idx < len(win.results) and 0 <= col_idx < len(win.results[r_idx]):
                    win.results[r_idx].pop(col_idx)
                if r_idx < len(win._full_results) and 0 <= col_idx < len(win._full_results[r_idx]):
                    win._full_results[r_idx].pop(col_idx)
            if win._col_summaries and 0 <= col_idx < len(win._col_summaries):
                win._col_summaries.pop(col_idx)
            if win._column_widths and 0 <= col_idx + 1 < len(win._column_widths):
                win._column_widths.pop(col_idx + 1)
            try:
                win._tabs[win._active_tab_index]["prompts_obj"] = {
                    pid: (p.model_copy(deep=True) if hasattr(p, "model_copy") else type(p)(**p.model_dump()))
                    for pid, p in win.prompts.items()
                }
            except Exception:
                pass
        except Exception:
            pass
        if not win.prompts:
            win._clear_all()
        win._update_ui()

    win.after(10, finalize_delete)
