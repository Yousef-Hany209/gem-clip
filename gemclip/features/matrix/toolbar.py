"""Toolbar builders for Matrix window."""
from __future__ import annotations

from pathlib import Path
import customtkinter as ctk

import styles
from i18n import tr


def build_toolbar(win) -> None:
    toolbar_frame = ctk.CTkFrame(win, fg_color=styles.MATRIX_TOP_BG_COLOR)
    toolbar_frame.pack(fill="x", padx=10, pady=10)
    for i in range(12):
        toolbar_frame.grid_columnconfigure(i, weight=1)
    toolbar_frame.grid_columnconfigure(11, weight=0)
    for i in (5, 6, 7, 8):
        toolbar_frame.grid_columnconfigure(i, weight=0)

    btn_add_input = ctk.CTkButton(
        toolbar_frame,
        text=tr("matrix.add_input"),
        command=win._add_input_row,
        fg_color=styles.DEFAULT_BUTTON_FG_COLOR,
        text_color=styles.DEFAULT_BUTTON_TEXT_COLOR,
    )
    btn_add_input.grid(row=0, column=0, padx=5, pady=5, sticky="ew")
    btn_add_input.bind("<Enter>", lambda e=None: win._show_tooltip(tr("matrix.add_input_hint")))
    btn_add_input.bind("<Leave>", lambda e=None: win._hide_tooltip())
    btn_add_input.bind("<ButtonPress-1>", lambda e=None: win._hide_tooltip())

    btn_add_prompt = ctk.CTkButton(
        toolbar_frame,
        text=tr("matrix.add_prompt"),
        command=win._add_prompt_column,
        fg_color=styles.DEFAULT_BUTTON_FG_COLOR,
        text_color=styles.DEFAULT_BUTTON_TEXT_COLOR,
    )
    btn_add_prompt.grid(row=0, column=1, padx=5, pady=5, sticky="ew")
    btn_add_prompt.bind("<Enter>", lambda e=None: win._show_tooltip(tr("matrix.add_prompt_hint")))
    btn_add_prompt.bind("<Leave>", lambda e=None: win._hide_tooltip())
    btn_add_prompt.bind("<ButtonPress-1>", lambda e=None: win._hide_tooltip())

    btn_add_set = ctk.CTkButton(
        toolbar_frame,
        text=tr("matrix.add_set"),
        command=win._add_prompt_set_tab,
        fg_color=styles.DEFAULT_BUTTON_FG_COLOR,
        text_color=styles.DEFAULT_BUTTON_TEXT_COLOR,
    )
    btn_add_set.grid(row=0, column=2, padx=5, pady=5, sticky="ew")
    btn_add_set.bind("<Enter>", lambda e=None: win._show_tooltip(tr("matrix.add_set_hint")))
    btn_add_set.bind("<Leave>", lambda e=None: win._hide_tooltip())

    btn_clear = ctk.CTkButton(
        toolbar_frame,
        text=tr("matrix.clear"),
        command=win._clear_active_set,
        fg_color=styles.DEFAULT_BUTTON_FG_COLOR,
        text_color=styles.DEFAULT_BUTTON_TEXT_COLOR,
    )
    btn_clear.grid(row=0, column=3, padx=5, pady=5, sticky="ew")
    btn_clear.bind("<Enter>", lambda e=None: win._show_tooltip(tr("matrix.clear_hint")))
    btn_clear.bind("<Leave>", lambda e=None: win._hide_tooltip())

    btn_set_manager = ctk.CTkButton(
        toolbar_frame,
        text=tr("matrix.set_manager"),
        command=win._open_set_manager,
        fg_color=styles.MATRIX_BUTTON_COLOR,
        hover_color=styles.MATRIX_BUTTON_HOVER_COLOR,
        text_color=styles.DEFAULT_BUTTON_TEXT_COLOR,
        corner_radius=6,
    )
    btn_set_manager.grid(row=0, column=4, padx=5, pady=5, sticky="ew")
    btn_set_manager.bind("<Enter>", lambda e=None: win._show_tooltip(tr("matrix.set_manager_hint")))
    btn_set_manager.bind("<Leave>", lambda e=None: win._hide_tooltip())

    session_label = ctk.CTkLabel(toolbar_frame, textvariable=win._session_label_var, anchor="w")
    session_label.grid(row=0, column=5, padx=(18, 0), pady=5, sticky="w")

    btn_save = ctk.CTkButton(
        toolbar_frame,
        text=tr("matrix.toolbar.save"),
        width=80,
        command=win._handle_save_action,
        fg_color=styles.DEFAULT_BUTTON_FG_COLOR,
        text_color=styles.DEFAULT_BUTTON_TEXT_COLOR,
    )
    btn_save.grid(row=0, column=6, padx=5, pady=5, sticky="ew")
    btn_save.bind("<Enter>", lambda e=None: win._show_tooltip(tr("matrix.toolbar.save_hint")))
    btn_save.bind("<Leave>", lambda e=None: win._hide_tooltip())

    btn_load = ctk.CTkButton(
        toolbar_frame,
        text=tr("matrix.toolbar.load"),
        width=80,
        command=win._open_session_loader,
        fg_color=styles.MATRIX_BUTTON_COLOR,
        hover_color=styles.MATRIX_BUTTON_HOVER_COLOR,
        text_color=styles.DEFAULT_BUTTON_TEXT_COLOR,
        corner_radius=6,
    )
    btn_load.grid(row=0, column=7, padx=5, pady=5, sticky="ew")
    btn_load.bind("<Enter>", lambda e=None: win._show_tooltip(tr("matrix.toolbar.load_hint")))
    btn_load.bind("<Leave>", lambda e=None: win._hide_tooltip())

    btn_session_manager = ctk.CTkButton(
        toolbar_frame,
        text=tr("matrix.toolbar.manage"),
        width=90,
        command=win._open_session_manager,
        fg_color=styles.MATRIX_BUTTON_COLOR,
        hover_color=styles.MATRIX_BUTTON_HOVER_COLOR,
        text_color=styles.DEFAULT_BUTTON_TEXT_COLOR,
        corner_radius=6,
    )
    btn_session_manager.grid(row=0, column=8, padx=5, pady=5, sticky="ew")
    btn_session_manager.bind("<Enter>", lambda e=None: win._show_tooltip(tr("matrix.toolbar.manage_hint")))
    btn_session_manager.bind("<Leave>", lambda e=None: win._hide_tooltip())

    try:
        config_icon_path = Path("config.ico")
        if config_icon_path.exists():
            from PIL import Image

            icon_img = Image.open(config_icon_path)
            size = (24, 24)
            config_icon = ctk.CTkImage(light_image=icon_img, dark_image=icon_img, size=size)
            settings_button = ctk.CTkButton(
                toolbar_frame,
                text="",
                image=config_icon,
                width=28,
                height=28,
                command=win._open_summary_settings,
            )
        else:
            raise FileNotFoundError
    except Exception:
        settings_button = ctk.CTkButton(
            toolbar_frame,
            text=tr("settings.title"),
            width=60,
            command=win._open_summary_settings,
            fg_color=styles.DEFAULT_BUTTON_FG_COLOR,
            text_color=styles.DEFAULT_BUTTON_TEXT_COLOR,
        )

    settings_button.grid(row=0, column=11, padx=5, pady=5, sticky="e")

