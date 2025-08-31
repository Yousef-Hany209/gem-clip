"""Popup dialogs for Matrix summaries and cell results.

Keeps UI code out of the Matrix window class. Functions receive the
window instance to reuse existing callbacks and state.
"""
from __future__ import annotations

import pyperclip
import customtkinter as ctk

import styles
from i18n import tr
from gemclip.ui.textbox_utils import setup_textbox_right_click_menu
from CTkMessagebox import CTkMessagebox


def _bring_to_front(parent, toplevel: ctk.CTkToplevel) -> None:
    try:
        toplevel.transient(parent)
        toplevel.lift(parent)
        toplevel.attributes("-topmost", True)
        toplevel.focus_force()
        toplevel.after(200, lambda: toplevel.attributes("-topmost", False))
    except Exception:
        pass


def show_final_summary_popup(win, summary_text: str) -> None:
    popup = ctk.CTkToplevel(win, fg_color=styles.HISTORY_ITEM_FG_COLOR)
    popup.title(tr("matrix.final_summary.title"))
    popup.geometry("700x500")
    _bring_to_front(win, popup)

    textbox = ctk.CTkTextbox(
        popup,
        wrap="word",
        fg_color=styles.HISTORY_ITEM_FG_COLOR,
        text_color=styles.HISTORY_ITEM_TEXT_COLOR,
    )
    setup_textbox_right_click_menu(textbox)
    textbox.pack(fill="both", expand=True, padx=10, pady=10)
    textbox.insert("1.0", summary_text)
    textbox.configure(state="normal")

    button_frame = ctk.CTkFrame(popup, fg_color="transparent")
    button_frame.pack(pady=5)

    ctk.CTkButton(
        button_frame,
        text=tr("common.save"),
        width=100,
        command=lambda: [win._update_matrix_summary_cell(textbox.get("1.0", "end-1c")), popup.destroy()],
    ).pack(side="left", padx=5)
    ctk.CTkButton(
        button_frame,
        text=tr("common.copy"),
        width=100,
        command=lambda: [
            pyperclip.copy(textbox.get("1.0", "end-1c")),
            CTkMessagebox(title=tr("common.copy"), message=tr("common.copied_to_clipboard"), icon="info").wait_window(),
        ],
    ).pack(side="left", padx=5)
    ctk.CTkButton(button_frame, text=tr("common.close"), width=100, command=popup.destroy).pack(
        side="left", padx=5
    )

    # Bind handlers with optional event to be robust across Tk variants
    textbox.bind("<Button-1>", lambda e=None: show_final_summary_popup(win, summary_text))
    textbox.bind("<Enter>", lambda e=None: textbox.configure(cursor="hand2"))
    textbox.bind("<Leave>", lambda e=None: textbox.configure(cursor=""))

    win.wait_window(popup)


def show_full_result_popup(win, r_idx: int, c_idx: int) -> None:
    """Open a popup showing the full cell result.

    Be defensive about indices because rows/columns can be added/removed
    and background operations may momentarily reset `_full_results`.
    Falls back to the visible `results` StringVar when necessary.
    """
    try:
        if 0 <= r_idx < len(win._full_results) and 0 <= c_idx < len(win._full_results[r_idx]):
            full_result = win._full_results[r_idx][c_idx]
        elif 0 <= r_idx < len(getattr(win, "results", [])) and 0 <= c_idx < len(win.results[r_idx]):
            # Fallback to the StringVar currently backing the cell
            try:
                full_result = win.results[r_idx][c_idx].get()
            except Exception:
                full_result = ""
        else:
            # Out of range: show empty text to avoid crashing
            full_result = ""
    except Exception:
        # Any unexpected structure issues — do not crash the UI
        full_result = ""
    popup = ctk.CTkToplevel(win, fg_color=styles.HISTORY_ITEM_FG_COLOR)
    popup.title(tr("matrix.result_preview_title_fmt", row=r_idx + 1, col=c_idx + 1))
    popup.geometry(styles.MATRIX_POPUP_GEOMETRY)
    _bring_to_front(win, popup)

    textbox = ctk.CTkTextbox(
        popup,
        wrap="word",
        fg_color=styles.HISTORY_ITEM_FG_COLOR,
        text_color=styles.HISTORY_ITEM_TEXT_COLOR,
    )
    setup_textbox_right_click_menu(textbox)
    textbox.insert("1.0", full_result)
    textbox.configure(state="normal")
    textbox.pack(fill="both", expand=True, padx=10, pady=10)

    btn_frame = ctk.CTkFrame(popup, fg_color="transparent")
    btn_frame.pack(pady=5)
    ctk.CTkButton(
        btn_frame,
        text=tr("common.save"),
        width=100,
        command=lambda: win._save_full_result_and_close_popup(popup, textbox, r_idx, c_idx),
    ).pack(side="left", padx=5)
    ctk.CTkButton(
        btn_frame,
        text=tr("common.copy"),
        width=100,
        command=lambda: [
            pyperclip.copy(textbox.get("1.0", "end-1c")),
            CTkMessagebox(title=tr("common.copy"), message=tr("common.copied_to_clipboard"), icon="info").wait_window(),
        ],
    ).pack(side="left", padx=5)
    ctk.CTkButton(btn_frame, text=tr("common.close"), width=100, command=popup.destroy).pack(
        side="left", padx=5
    )

    # Optional: ensure cursor feedback works across variants
    textbox.bind("<Enter>", lambda e=None: textbox.configure(cursor="xterm"))
    textbox.bind("<Leave>", lambda e=None: textbox.configure(cursor=""))

    win.wait_window(popup)
    try:
        win.grab_release()
    except Exception:
        pass


def show_full_row_summary_popup(win, r_idx: int) -> None:
    full_summary = win._row_summaries[r_idx].get()
    popup = ctk.CTkToplevel(win, fg_color=styles.HISTORY_ITEM_FG_COLOR)
    popup.title(tr("matrix.row_result_preview_title_fmt", row=r_idx + 1))
    popup.geometry(styles.MATRIX_POPUP_GEOMETRY)
    _bring_to_front(win, popup)

    textbox = ctk.CTkTextbox(
        popup,
        wrap="word",
        fg_color=styles.HISTORY_ITEM_FG_COLOR,
        text_color=styles.HISTORY_ITEM_TEXT_COLOR,
    )
    setup_textbox_right_click_menu(textbox)
    textbox.insert("1.0", full_summary)
    textbox.configure(state="normal")
    textbox.pack(fill="both", expand=True, padx=10, pady=10)

    btn_frame = ctk.CTkFrame(popup, fg_color="transparent")
    btn_frame.pack(pady=5)
    ctk.CTkButton(
        btn_frame,
        text=tr("common.save"),
        width=100,
        command=lambda: win._save_full_row_summary_and_close_popup(popup, textbox, r_idx),
    ).pack(side="left", padx=5)
    ctk.CTkButton(
        btn_frame,
        text=tr("common.copy"),
        width=100,
        command=lambda: [
            pyperclip.copy(textbox.get("1.0", "end-1c")),
            CTkMessagebox(title=tr("common.copy"), message=tr("common.copied_to_clipboard"), icon="info").wait_window(),
        ],
    ).pack(side="left", padx=5)
    ctk.CTkButton(btn_frame, text=tr("common.close"), width=100, command=popup.destroy).pack(
        side="left", padx=5
    )

    textbox.bind("<Enter>", lambda e=None: textbox.configure(cursor="xterm"))
    textbox.bind("<Leave>", lambda e=None: textbox.configure(cursor=""))

    win.wait_window(popup)
    try:
        win.grab_release()
    except Exception:
        pass


def show_full_col_summary_popup(win, c_idx: int) -> None:
    full_summary = win._col_summaries[c_idx].get()
    popup = ctk.CTkToplevel(win, fg_color=styles.HISTORY_ITEM_FG_COLOR)
    popup.title(tr("matrix.col_result_preview_title_fmt", col=chr(ord("A") + c_idx)))
    popup.geometry(styles.MATRIX_POPUP_GEOMETRY)
    _bring_to_front(win, popup)

    textbox = ctk.CTkTextbox(
        popup,
        wrap="word",
        fg_color=styles.HISTORY_ITEM_FG_COLOR,
        text_color=styles.HISTORY_ITEM_TEXT_COLOR,
    )
    setup_textbox_right_click_menu(textbox)
    textbox.insert("1.0", full_summary)
    textbox.configure(state="normal")
    textbox.pack(fill="both", expand=True, padx=10, pady=10)

    btn_frame = ctk.CTkFrame(popup, fg_color="transparent")
    btn_frame.pack(pady=5)
    ctk.CTkButton(
        btn_frame,
        text=tr("common.save"),
        width=100,
        command=lambda: win._save_full_col_summary_and_close_popup(popup, textbox, c_idx),
    ).pack(side="left", padx=5)
    ctk.CTkButton(
        btn_frame,
        text=tr("common.copy"),
        width=100,
        command=lambda: [
            pyperclip.copy(textbox.get("1.0", "end-1c")),
            CTkMessagebox(title=tr("common.copy"), message=tr("common.copied_to_clipboard"), icon="info").wait_window(),
        ],
    ).pack(side="left", padx=5)
    ctk.CTkButton(btn_frame, text=tr("common.close"), width=100, command=popup.destroy).pack(
        side="left", padx=5
    )

    win.wait_window(popup)
    try:
        win.grab_release()
    except Exception:
        pass
