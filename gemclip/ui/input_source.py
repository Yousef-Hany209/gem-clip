"""Input source selection dialog for Matrix rows."""
from __future__ import annotations

from io import BytesIO
from pathlib import Path
import base64
from tkinter import filedialog, messagebox

from PIL import Image

from i18n import tr


def select_input_source(win, row_idx: int) -> None:
    file_path = filedialog.askopenfilename(
        title=tr("matrix.select_input_file"),
        filetypes=[
            ("All Supported", "*.png *.jpg *.jpeg *.gif *.bmp *.webp *.pdf *.txt *.md *.csv *.py *.mp3 *.wav *.xlsx *.doc *.docx"),
            ("Images", "*.png *.jpg *.jpeg *.gif *.bmp *.webp"),
            ("PDF", "*.pdf"),
            ("Text", "*.txt *.md *.csv"),
            ("All Files", "*.*"),
        ],
    )
    if not file_path:
        return
    file_path_obj = Path(file_path)
    file_type = "file"
    data_content = file_path
    if file_path_obj.suffix.lower() in [".png", ".jpg", ".jpeg", ".gif", ".bmp", ".webp"]:
        try:
            image = Image.open(file_path)
            if image.mode != "RGB":
                image = image.convert("RGB")
            with BytesIO() as buffer:
                image.save(buffer, format="PNG")
                data_content = base64.b64encode(buffer.getvalue()).decode("utf-8")
            file_type = "image"
        except Exception as e:
            messagebox.showerror(tr("common.error_title"), tr("matrix.image_preview_failed", details=str(e)))
            return
    win.input_data[row_idx] = {"type": file_type, "data": data_content}
    win._update_input_row_display(row_idx)

