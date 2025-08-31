"""Tabbar rendering for Matrix window (decoupled from class to reduce size)."""
from __future__ import annotations

import customtkinter as ctk
from pathlib import Path
from typing import Any

import styles
from gemclip.core import DELETE_ICON_FILE
from i18n import tr


def render_tabbar(win: Any) -> None:
    # Clear and rebuild tab buttons
    for w in win.tabbar_frame.winfo_children():
        w.destroy()
    max_tabs = 5
    slot_w = win._compute_slot_width()
    win._tab_slot_width = slot_w
    # Determine bottom blend color (match matrix canvas background)
    try:
        cur_fg = win.canvas_frame.cget("fg_color")
        appearance_mode_index = 0 if ctk.get_appearance_mode() == "Light" else 1
        blend_color = cur_fg[appearance_mode_index] if isinstance(cur_fg, tuple) else cur_fg
    except Exception:
        mc = styles.MATRIX_CANVAS_BACKGROUND_COLOR
        appearance_mode_index = 0 if ctk.get_appearance_mode() == "Light" else 1
        blend_color = mc[appearance_mode_index] if isinstance(mc, tuple) else mc
    for i in range(max_tabs):
        is_real = i < len(win._tabs)
        is_active = (i == getattr(win, '_active_tab_index', -1))
        if not is_real:
            fg = styles.MATRIX_TOP_BG_COLOR
        else:
            fg = blend_color if is_active else ("#D9D9D9", "#1F1F1F")
        outer = ctk.CTkFrame(win.tabbar_frame, corner_radius=0, border_width=0, fg_color=fg, border_color=styles.MATRIX_HEADER_BORDER_COLOR)
        outer.pack(side="left", padx=4, pady=0)
        outer.configure(width=slot_w, height=34)
        outer.pack_propagate(False)
        if is_real:
            name = str(win._tabs[i].get('name') or tr('matrix.tab.auto_name_fmt', n=i+1))
            inner = ctk.CTkFrame(outer, fg_color="transparent")
            inner.pack(fill="both", expand=True, padx=10, pady=(2,4))
            title = ctk.CTkLabel(inner, text=name, font=styles.MATRIX_FONT_BOLD, anchor="w")
            title.pack(side="left", fill="x", expand=True)
            title.bind("<Double-Button-1>", lambda e, idx=i: win._rename_tab(idx))
            try:
                icon_path = Path(DELETE_ICON_FILE)
                if icon_path.exists():
                    from PIL import Image
                    _img = Image.open(icon_path)
                    _img.thumbnail((16,16))
                    tab_delete_icon = ctk.CTkImage(light_image=_img, dark_image=_img, size=(16,16))
                else:
                    tab_delete_icon = None
            except Exception:
                tab_delete_icon = None
            close_btn = ctk.CTkButton(inner, text="", image=tab_delete_icon, width=24, height=24,
                                      fg_color=styles.MATRIX_DELETE_BUTTON_COLOR,
                                      hover_color=styles.MATRIX_DELETE_BUTTON_HOVER_COLOR,
                                      text_color=styles.DEFAULT_BUTTON_TEXT_COLOR,
                                      command=lambda idx=i: win._delete_tab_index(idx))
            close_btn.pack(side="right")
            for wdg in (outer, inner, title):
                wdg.bind("<ButtonPress-1>", lambda e, idx=i: win._on_tab_press(e, idx))
                wdg.bind("<B1-Motion>", win._on_tab_motion)
                wdg.bind("<ButtonRelease-1>", win._on_tab_release)
        else:
            pass
    win._adjust_tabbar_widths()


def compute_slot_width(win: Any) -> int:
    max_tabs = 5
    try:
        total_w = max(int(win.tabbar_frame.winfo_width()) - 40, 600)
    except Exception:
        total_w = 1000
    return max(140, total_w // max_tabs)


def adjust_tabbar_widths(win: Any) -> None:
    slot_w = compute_slot_width(win)
    if getattr(win, '_tab_slot_width', None) == slot_w:
        return
    win._tab_slot_width = slot_w
    for child in win.tabbar_frame.winfo_children():
        try:
            child.configure(width=slot_w)
        except Exception:
            pass
