"""Export utilities for Matrix (e.g., Excel/TSV via clipboard)."""
from __future__ import annotations

import pyperclip
from tkinter import messagebox

from i18n import tr


def export_to_excel(win) -> None:
    try:
        header_parts = [
            tr("matrix.export.header_input"),
            tr("matrix.export.header_prompt_name"),
            tr("matrix.export.header_system_prompt"),
        ] + [p.name for p in win.prompts.values()]
        if win._row_summaries:
            header_parts.append(tr("matrix.row_summary_header"))
        tsv_data = ["\t".join(header_parts)]
        for r_idx, input_item in enumerate(win.input_data):
            input_display = input_item["data"] if input_item["type"] == "text" else f"[{input_item['type']}]"
            row_parts = [input_display, "", ""]
            for c_idx in range(len(win.prompts)):
                cell_result = (
                    win._full_results[r_idx][c_idx]
                    if r_idx < len(win._full_results) and c_idx < len(win._full_results[r_idx])
                    else ""
                )
                row_parts.append(cell_result.replace("\n", " ").replace("\r", ""))
            if win._row_summaries:
                row_summary = win._row_summaries[r_idx].get() if r_idx < len(win._row_summaries) else ""
                row_parts.append(row_summary.replace("\n", " ").replace("\r", ""))
            tsv_data.append("\t".join(row_parts))
        if win._col_summaries:
            col_summary_parts = [
                tr("matrix.col_summary_header"),
                "",
                "",
            ] + [
                win._col_summaries[c_idx].get().replace("\n", " ").replace("\r", "") if c_idx < len(win._col_summaries) else ""
                for c_idx in range(len(win.prompts))
            ]
            if win._row_summaries:
                col_summary_parts.append("")
            tsv_data.append("\t".join(col_summary_parts))
        pyperclip.copy("\n".join(tsv_data))
        messagebox.showinfo(tr("matrix.export.title"), tr("matrix.export.copied"))
    except Exception as e:
        messagebox.showerror(tr("common.error_title"), tr("matrix.export.error", details=str(e)))

