"""View helpers for Matrix feature (small UI widgets and renderers)."""
import tkinter as tk
import customtkinter as ctk
from io import BytesIO
from pathlib import Path
import base64
from PIL import Image

import styles
from i18n import tr
from gemclip.ui.textbox_utils import setup_textbox_right_click_menu


class SizerGrip(tk.Frame):
    def __init__(self, parent, **kwargs):
        super().__init__(parent, **kwargs)
        self.bind("<B1-Motion>", self._on_motion)
        self.bind("<ButtonPress-1>", self._on_press)
        self.configure(cursor="sizing")

    def _on_press(self, event):
        self._start_x = event.x
        self._start_y = event.y
        self._toplevel = self.winfo_toplevel()

    def _on_motion(self, event):
        x, y = (event.x, event.y)
        w, h = (self.master.winfo_width() + x, self.master.winfo_height() + y)
        self.master.configure(width=w, height=h)


def update_row_summary_column(win) -> None:
    """Render or hide the row summary column on the right."""
    num_prompts = len(win.prompts)
    summary_col_idx = num_prompts + 1

    for widget in win.scrollable_content_frame.grid_slaves(column=summary_col_idx):
        widget.destroy()

    if win._row_summaries:
        win.scrollable_content_frame.grid_columnconfigure(
            summary_col_idx, weight=0, minsize=styles.MATRIX_CELL_WIDTH
        )
        summary_header_frame = ctk.CTkFrame(
            win.scrollable_content_frame,
            border_width=1,
            border_color=styles.MATRIX_HEADER_BORDER_COLOR,
            width=styles.MATRIX_CELL_WIDTH,
            height=styles.MATRIX_RESULT_CELL_HEIGHT,
        )
        summary_header_frame.grid_propagate(False)
        ctk.CTkLabel(
            summary_header_frame,
            text=tr("matrix.row_summary_header"),
            font=styles.MATRIX_FONT_BOLD,
        ).pack(fill="x", padx=2, pady=2)
        summary_header_frame.grid(
            row=0, column=summary_col_idx, padx=5, pady=5, sticky="nsew"
        )

        for r_idx, summary_var in enumerate(win._row_summaries):
            summary_cell_frame = ctk.CTkFrame(
                win.scrollable_content_frame,
                border_width=1,
                border_color=styles.MATRIX_CELL_BORDER_COLOR,
                width=styles.MATRIX_CELL_WIDTH,
                height=styles.MATRIX_RESULT_CELL_HEIGHT,
            )
            summary_cell_frame.grid_propagate(False)
            summary_cell_frame.grid(
                row=r_idx + 1,
                column=summary_col_idx,
                padx=5,
                pady=5,
                sticky="nsew",
            )
            inner_h = max(20, styles.MATRIX_RESULT_CELL_HEIGHT - 6)
            summary_textbox = ctk.CTkTextbox(
                summary_cell_frame,
                width=styles.MATRIX_CELL_WIDTH,
                height=inner_h,
                wrap="word",
                fg_color=styles.HISTORY_ITEM_FG_COLOR,
                text_color=styles.HISTORY_ITEM_TEXT_COLOR,
            )
            setup_textbox_right_click_menu(summary_textbox)
            summary_textbox.insert("1.0", summary_var.get())
            summary_textbox.configure(state="disabled")
            summary_textbox.pack(fill="both", expand=True, padx=2, pady=2)
            summary_textbox.bind(
                "<Button-1>", lambda e=None, r=r_idx: win._show_full_row_summary_popup(r)
            )
            summary_textbox.bind("<Enter>", lambda e=None: summary_textbox.configure(cursor="hand2"))
            summary_textbox.bind("<Leave>", lambda e=None: summary_textbox.configure(cursor=""))
            summary_var.trace_add(
                "write",
                lambda name, index, mode, sv=summary_var, tb=summary_textbox: win._update_textbox_from_stringvar(
                    sv, tb
                ),
            )
    else:
        win.scrollable_content_frame.grid_columnconfigure(
            summary_col_idx, weight=0, minsize=0
        )


def update_column_summary_row(win) -> None:
    """Render or hide the column summary row at the bottom."""
    num_inputs = len(win.input_data)
    summary_row_idx = num_inputs + 1

    for widget in win.scrollable_content_frame.grid_slaves(row=summary_row_idx):
        widget.destroy()

    if win._col_summaries:
        win.scrollable_content_frame.grid_rowconfigure(
            summary_row_idx, weight=0, minsize=styles.MATRIX_RESULT_CELL_HEIGHT
        )
        summary_header_frame = ctk.CTkFrame(
            win.scrollable_content_frame,
            border_width=1,
            border_color=styles.MATRIX_HEADER_BORDER_COLOR,
            height=styles.MATRIX_RESULT_CELL_HEIGHT,
        )
        summary_header_frame.grid_propagate(False)
        ctk.CTkLabel(
            summary_header_frame,
            text=tr("matrix.col_summary_header"),
            font=styles.MATRIX_FONT_BOLD,
        ).pack(fill="x", padx=2, pady=2)
        summary_header_frame.grid(
            row=summary_row_idx, column=0, padx=5, pady=5, sticky="nsew"
        )

        for c_idx, summary_var in enumerate(win._col_summaries):
            summary_cell_frame = ctk.CTkFrame(
                win.scrollable_content_frame,
                border_width=1,
                border_color=styles.MATRIX_CELL_BORDER_COLOR,
                width=styles.MATRIX_CELL_WIDTH,
                height=styles.MATRIX_RESULT_CELL_HEIGHT,
            )
            summary_cell_frame.grid(
                row=summary_row_idx, column=c_idx + 1, padx=5, pady=5, sticky="nsew"
            )
            summary_cell_frame.grid_propagate(False)

            summary_cell_frame.grid_rowconfigure(0, weight=1)
            summary_cell_frame.grid_columnconfigure(0, weight=1)

            inner_h = max(20, styles.MATRIX_RESULT_CELL_HEIGHT - 6)
            summary_textbox = ctk.CTkTextbox(
                summary_cell_frame,
                width=styles.MATRIX_CELL_WIDTH,
                height=inner_h,
                wrap="word",
                fg_color=styles.HISTORY_ITEM_FG_COLOR,
                text_color=styles.HISTORY_ITEM_TEXT_COLOR,
            )
            setup_textbox_right_click_menu(summary_textbox)
            summary_textbox.insert("1.0", summary_var.get())
            summary_textbox.configure(state="disabled")
            summary_textbox.grid(row=0, column=0, padx=2, pady=2, sticky="nsew")
            summary_textbox.bind(
                "<Button-1>", lambda e=None, c=c_idx: win._show_full_col_summary_popup(c)
            )
            summary_textbox.bind("<Enter>", lambda e=None: summary_textbox.configure(cursor="hand2"))
            summary_textbox.bind("<Leave>", lambda e=None: summary_textbox.configure(cursor=""))
            summary_var.trace_add(
                "write",
                lambda *args, sv=summary_var, tb=summary_textbox: win._update_textbox_from_stringvar(
                    sv, tb
                ),
            )

            sizer = SizerGrip(summary_cell_frame)
            sizer.grid(row=1, column=1, sticky="se")
    else:
        win.scrollable_content_frame.grid_rowconfigure(
            summary_row_idx, weight=0, minsize=0
        )


def update_matrix_summary_cell(win, summary_text: str) -> None:
    """Render the bottom-right final summary cell as a resizable area."""
    num_inputs = len(win.input_data)
    num_prompts = len(win.prompts)
    summary_row_idx = num_inputs + 1
    summary_col_idx = num_prompts + 1

    for widget in win.scrollable_content_frame.grid_slaves(
        row=summary_row_idx, column=summary_col_idx
    ):
        widget.destroy()

    resizable_frame = tk.Frame(win.scrollable_content_frame, borderwidth=1, relief="solid")
    resizable_frame.grid(row=summary_row_idx, column=summary_col_idx, padx=5, pady=5, sticky="nsew")
    resizable_frame.grid_rowconfigure(0, weight=1)
    resizable_frame.grid_columnconfigure(0, weight=1)

    inner_h = max(20, styles.MATRIX_RESULT_CELL_HEIGHT - 6)
    summary_textbox = ctk.CTkTextbox(
        resizable_frame,
        width=styles.MATRIX_CELL_WIDTH,
        height=inner_h,
        wrap="word",
        fg_color=styles.HISTORY_ITEM_FG_COLOR,
        text_color=styles.HISTORY_ITEM_TEXT_COLOR,
    )
    setup_textbox_right_click_menu(summary_textbox)
    summary_textbox.grid(row=0, column=0, padx=2, pady=2, sticky="nsew")
    summary_textbox.insert("1.0", summary_text)
    summary_textbox.configure(state="disabled")
    summary_textbox.bind("<Button-1>", lambda e=None: win._show_final_summary_popup(summary_text))
    summary_textbox.bind("<Enter>", lambda e=None: summary_textbox.configure(cursor="hand2"))
    summary_textbox.bind("<Leave>", lambda e=None: summary_textbox.configure(cursor=""))

    sizer = SizerGrip(resizable_frame)
    sizer.grid(row=1, column=1, sticky="se")


def compute_col_drop_index(win, x_root: int) -> int:
    frames = [f for f in win._col_header_frames if f is not None]
    if not frames:
        return 0
    mids = []
    for f in frames:
        try:
            left = f.winfo_rootx()
            w = f.winfo_width() or styles.MATRIX_CELL_WIDTH
            mids.append(left + w / 2)
        except Exception:
            mids.append(0)
    if x_root <= mids[0]:
        return 0
    if x_root >= mids[-1]:
        return len(frames) - 1
    best = 0
    best_d = float("inf")
    for i, m in enumerate(mids):
        d = abs(x_root - m)
        if d < best_d:
            best_d = d
            best = i
    return best


def draw_col_drop_indicator(win, x_root: int) -> None:
    try:
        frames = [f for f in win._col_header_frames if f is not None and f.winfo_exists()]
        if not frames:
            return
        lefts = []
        rights = []
        for f in frames:
            lx = f.winfo_rootx()
            w = f.winfo_width() or styles.MATRIX_CELL_WIDTH
            lefts.append(lx)
            rights.append(lx + w)
        # choose nearest boundary
        boundaries = sorted(set(lefts + rights))
        if not boundaries:
            return
        nearest = min(boundaries, key=lambda b: abs(b - x_root))
        # Create or move a slim indicator widget
        try:
            if getattr(win, "_col_drop_indicator_widget", None) is None or not win._col_drop_indicator_widget.winfo_exists():
                win._col_drop_indicator_widget = tk.Frame(win, bg="white", width=2, height=win.winfo_height())
                win._col_drop_indicator_widget.place(x=0, y=0)
            rel_x = max(0, nearest - win.winfo_rootx())
            win._col_drop_indicator_widget.place(x=rel_x, y=win.scrollable_content_frame.winfo_rooty() - win.winfo_rooty(), height=win.scrollable_content_frame.winfo_height())
        except Exception:
            pass
    except Exception:
        pass


def add_input_row_widgets(win, row_idx: int, input_item):
    while len(win._full_results) <= row_idx:
        win._full_results.append([])
    input_cell_frame = ctk.CTkFrame(
        win.scrollable_content_frame,
        border_width=1,
        border_color=styles.MATRIX_CELL_BORDER_COLOR,
        width=styles.MATRIX_CELL_WIDTH,
        height=styles.MATRIX_RESULT_CELL_HEIGHT,
    )
    input_cell_frame.grid(row=row_idx + 1, column=0, padx=5, pady=5, sticky="nsew")
    input_cell_frame.grid_propagate(False)
    input_cell_frame.grid_columnconfigure(2, weight=1)
    while len(win._input_row_frames) <= row_idx:
        win._input_row_frames.append(None)
    win._input_row_frames[row_idx] = input_cell_frame

    row_header_frame = ctk.CTkFrame(input_cell_frame)
    row_header_frame.grid(row=0, column=0, padx=(5, 2), pady=2, sticky="w")
    row_header_frame.grid_columnconfigure(0, weight=1)

    row_num_label = ctk.CTkLabel(row_header_frame, text=f"{row_idx + 1}", font=styles.MATRIX_FONT_BOLD)
    row_num_label.grid(row=0, column=0, sticky="w")

    delete_row_icon = None
    try:
        icon_path = Path(styles.DELETE_ICON_FILE) if hasattr(styles, 'DELETE_ICON_FILE') else None
        if icon_path and icon_path.exists():
            icon_img = Image.open(icon_path)
            size = (16, 16)
            icon_img.thumbnail(size)
            delete_row_icon = ctk.CTkImage(light_image=icon_img, dark_image=icon_img, size=size)
    except Exception:
        delete_row_icon = None

    del_btn = ctk.CTkButton(
        row_header_frame,
        text="" if delete_row_icon else tr("common.delete"),
        image=delete_row_icon,
        width=24,
        height=24,
        fg_color=styles.MATRIX_DELETE_BUTTON_COLOR,
        hover_color=styles.MATRIX_DELETE_BUTTON_HOVER_COLOR,
        command=lambda r=row_idx: win._delete_row(r),
    )
    del_btn.grid(row=0, column=1, padx=(5, 0), sticky="e")

    input_label = ctk.CTkLabel(
        input_cell_frame,
        text=tr("action.input").rstrip(':'),
        font=styles.MATRIX_FONT_BOLD,
        text_color=styles.HISTORY_ITEM_TEXT_COLOR,
    )
    input_label.grid(row=0, column=1, padx=(5, 2), pady=2, sticky="w")

    if input_item["type"] == "text":
        input_entry = ctk.CTkEntry(
            input_cell_frame,
            placeholder_text=tr("matrix.input_placeholder", n=row_idx + 1),
            fg_color=styles.HISTORY_ITEM_FG_COLOR,
            text_color=styles.HISTORY_ITEM_TEXT_COLOR,
        )
        input_entry.insert(0, input_item["data"])
        input_entry.configure(state="readonly")
        input_entry.bind("<Button-1>", lambda e=None, r=row_idx: win._open_history_edit_dialog(r))
        input_entry.grid(row=0, column=2, padx=2, pady=2, sticky="ew")
    elif input_item["type"] in ("image", "image_compressed"):
        try:
            raw_bytes = base64.b64decode(input_item["data"])
            if input_item["type"] == "image_compressed":
                import zlib

                raw_bytes = zlib.decompress(raw_bytes)
            image = Image.open(BytesIO(raw_bytes))
            image.thumbnail(styles.MATRIX_IMAGE_THUMBNAIL_SIZE)
            ctk_image = ctk.CTkImage(
                light_image=image, dark_image=image, size=styles.MATRIX_IMAGE_THUMBNAIL_SIZE
            )
            image_label = ctk.CTkLabel(
                input_cell_frame, image=ctk_image, text="", text_color=styles.HISTORY_ITEM_TEXT_COLOR
            )
            image_label.grid(row=0, column=2, padx=2, pady=2, sticky="w")
            image_label.bind("<Button-1>", lambda e=None, r=row_idx: win._show_image_preview(r))
        except Exception:
            error_label = ctk.CTkLabel(
                input_cell_frame,
                text=tr("matrix.image_error"),
                text_color=styles.NOTIFICATION_COLORS["error"],
            )
            error_label.grid(row=0, column=2, padx=2, pady=2, sticky="w")
    elif input_item["type"] == "file":
        file_path = Path(input_item["data"])
        file_label = ctk.CTkLabel(
            input_cell_frame, text=file_path.name, text_color=styles.HISTORY_ITEM_TEXT_COLOR, anchor="w"
        )
        file_label.grid(row=0, column=2, padx=2, pady=2, sticky="ew")
        file_label.bind("<Enter>", lambda e=None, p=str(file_path): win._show_tooltip(p))
        file_label.bind("<Leave>", lambda e=None: win._hide_tooltip())

    attach_button = ctk.CTkButton(
        input_cell_frame,
        text=tr("action.attach"),
        width=50,
        fg_color=styles.FILE_ATTACH_BUTTON_COLOR,
        text_color=styles.DEFAULT_BUTTON_TEXT_COLOR,
        command=lambda r=row_idx: win._select_input_source(r),
    )
    attach_button.grid(row=0, column=3, padx=2, pady=2, sticky="e")

    history_button = ctk.CTkButton(
        input_cell_frame,
        text=tr("history.button"),
        width=50,
        fg_color=styles.DEFAULT_BUTTON_FG_COLOR,
        text_color=styles.DEFAULT_BUTTON_TEXT_COLOR,
        command=lambda r=row_idx: win._show_clipboard_history_popup(r),
    )
    history_button.grid(row=0, column=4, padx=2, pady=2, sticky="e")
    input_cell_frame.grid_columnconfigure(4, weight=0)

    for col_idx in range(len(win.prompts)):
        win._create_result_cell(row_idx, col_idx)
