"""Flow progress dialogs used during multi-step operations in Matrix.

Maintains previous behavior; stores handles on `win` for compatibility.
"""
from __future__ import annotations

import customtkinter as ctk

import styles
from i18n import tr


def show_flow_progress_dialog(win, on_cancel) -> None:
    try:
        if getattr(win, "_flow_dialog", None) and win._flow_dialog.winfo_exists():
            return
    except Exception:
        pass
    dlg = ctk.CTkToplevel(win, fg_color=styles.HISTORY_ITEM_FG_COLOR)
    dlg.title(tr("matrix.flow.running_title"))
    dlg.geometry("360x140")
    dlg.transient(win)
    lbl = ctk.CTkLabel(dlg, text=tr("matrix.flow.running_message"), text_color=styles.HISTORY_ITEM_TEXT_COLOR)
    lbl.pack(padx=16, pady=(20, 10))
    btn = ctk.CTkButton(
        dlg,
        text=tr("matrix.flow.cancel"),
        width=100,
        fg_color=styles.CANCEL_BUTTON_COLOR,
        text_color=styles.CANCEL_BUTTON_TEXT_COLOR,
        command=on_cancel,
    )
    btn.pack(pady=10)
    win._flow_dialog = dlg
    win._flow_dialog_label = lbl


def close_flow_progress_dialog(win) -> None:
    try:
        if getattr(win, "_flow_dialog", None) and win._flow_dialog.winfo_exists():
            try:
                win._flow_dialog.grab_release()
            except Exception:
                pass
            win._flow_dialog.destroy()
    except Exception:
        pass
    win._flow_dialog = None
    win._flow_dialog_label = None

