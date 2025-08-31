"""Layout helpers for Matrix window."""
from __future__ import annotations

from typing import Any
import customtkinter as ctk
import styles


def configure_scrollable_grid(win: Any) -> None:
    frame = win.scrollable_content_frame
    frame.update_idletasks()
    num_prompts = len(win.prompts)
    num_inputs = len(win.input_data)

    total_cols = 1 + num_prompts + (1 if win._row_summaries else 0)
    for i in range(total_cols):
        frame.grid_columnconfigure(i, weight=0)
    frame.grid_columnconfigure(0, weight=0)
    for i in range(1, num_prompts + 1):
        frame.grid_columnconfigure(i, weight=0, minsize=styles.MATRIX_CELL_WIDTH)
    if win._row_summaries:
        frame.grid_columnconfigure(num_prompts + 1, weight=0, minsize=styles.MATRIX_CELL_WIDTH)

    total_rows = 1 + num_inputs + (1 if win._col_summaries else 0)
    for i in range(total_rows):
        frame.grid_rowconfigure(i, weight=0)
    frame.grid_rowconfigure(0, weight=0, minsize=styles.MATRIX_RESULT_CELL_HEIGHT)
    for i in range(1, num_inputs + 1):
        frame.grid_rowconfigure(i, weight=0, minsize=styles.MATRIX_RESULT_CELL_HEIGHT)
    if win._col_summaries:
        frame.grid_rowconfigure(num_inputs + 1, weight=0, minsize=styles.MATRIX_RESULT_CELL_HEIGHT)

    # top-left spacer
    ctk.CTkLabel(frame, text="").grid(row=0, column=0, padx=5, pady=5, sticky="nsew")
    # Reset header frames list
    win._col_header_frames = []


def save_active_tab_vars(win: Any) -> None:
    try:
        if 0 <= win._active_tab_index < len(win._tabs):
            win._tabs[win._active_tab_index]['vars'] = {
                'checkbox_states': win.checkbox_states,
                'results': win.results,
                '_full_results': win._full_results,
                '_row_summaries': win._row_summaries,
                '_col_summaries': win._col_summaries,
                '_result_textboxes': win._result_textboxes,
                '_cell_style': win._cell_style
            }
            win._tabs[win._active_tab_index]['frame'] = win.scrollable_content_frame
    except Exception:
        pass

