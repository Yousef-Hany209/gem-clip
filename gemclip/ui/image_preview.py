"""Image preview popup for Matrix inputs.

Replicates legacy behavior while keeping UI code outside the window class.
"""
from __future__ import annotations

from io import BytesIO
import base64
from tkinter import messagebox

import customtkinter as ctk
from PIL import Image

import styles
from i18n import tr


def _bring_to_front(parent, toplevel: ctk.CTkToplevel) -> None:
    try:
        toplevel.transient(parent)
        toplevel.lift(parent)
        toplevel.attributes("-topmost", True)
        toplevel.focus_force()
        toplevel.after(200, lambda: toplevel.attributes("-topmost", False))
    except Exception:
        pass


def show_image_preview(win, row_idx: int) -> None:
    item = win.input_data[row_idx]
    if item.get("type") not in ("image", "image_compressed"):
        messagebox.showinfo(tr("common.info"), tr("matrix.not_image_row"))
        return
    try:
        raw = base64.b64decode(item["data"])
        if item["type"] == "image_compressed":
            import zlib

            raw = zlib.decompress(raw)
        image = Image.open(BytesIO(raw))
        popup = ctk.CTkToplevel(win, fg_color=styles.HISTORY_ITEM_FG_COLOR)
        popup.title(tr("matrix.image_preview_title_fmt", row=row_idx + 1))
        max_width = win.winfo_width() * 0.8
        max_height = win.winfo_height() * 0.8
        image.thumbnail((max_width, max_height), Image.LANCZOS)
        ctk_image = ctk.CTkImage(light_image=image, dark_image=image, size=image.size)
        image_label = ctk.CTkLabel(popup, image=ctk_image, text="", text_color=styles.HISTORY_ITEM_TEXT_COLOR)
        image_label.pack(padx=10, pady=10)
        close_button = ctk.CTkButton(
            popup,
            text=tr("common.close"),
            command=popup.destroy,
            fg_color=styles.CANCEL_BUTTON_COLOR,
            text_color=styles.CANCEL_BUTTON_TEXT_COLOR,
        )
        close_button.pack(pady=5)
        _bring_to_front(win, popup)
        win.wait_window(popup)
        try:
            win.grab_release()
        except Exception:
            pass
    except Exception as e:
        messagebox.showerror(tr("common.error_title"), tr("matrix.image_preview_failed", details=str(e)))

