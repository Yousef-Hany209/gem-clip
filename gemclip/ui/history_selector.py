"""Clipboard history selection popup for Matrix.

Extracted from the legacy window to reduce monolith size.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import base64
import customtkinter as ctk

from gemclip.infra import db
import styles
from i18n import tr


class ClipboardHistorySelectorPopup(ctk.CTkToplevel):
    """Small scrollable list to pick a history item."""

    def __init__(
        self,
        parent_app: ctk.CTk,
        clipboard_history: List[Dict[str, Any]],
        on_select_callback: Callable[[Dict[str, Any]], None],
        on_destroy_callback: Optional[Callable] = None,
        page_limit: int = 20,
        initial_offset: int = 0,
    ):
        super().__init__(parent_app)
        self.transient(parent_app)

        # Non-modal overlay-like
        self.withdraw()
        self.overrideredirect(True)
        self.attributes("-topmost", True)

        self.on_select_callback = on_select_callback
        self._on_destroy_callback = on_destroy_callback
        self._history_items = clipboard_history
        self._buttons: List[ctk.CTkButton] = []
        self._current_selection_index = 0
        self._is_destroying = False
        self._page_limit = max(1, int(page_limit))
        self._offset = max(0, int(initial_offset))

        self.main_frame = ctk.CTkFrame(self, fg_color=styles.POPUP_BG_COLOR)
        self.main_frame.pack(fill="both", expand=True)
        self.main_frame.grid_rowconfigure(0, weight=1)
        self.main_frame.grid_columnconfigure(0, weight=1)

        self.scrollable_frame = ctk.CTkScrollableFrame(self.main_frame)
        self.scrollable_frame.grid(row=0, column=0, sticky="nsew", padx=5, pady=5)
        self.scrollable_frame.grid_columnconfigure(0, weight=1)

        for i, item in enumerate(self._history_items):
            try:
                if item.get("type") == "text":
                    text = item.get("data", "")
                    label = text.replace("\n", " ")[:80] + ("..." if len(text) > 80 else "")
                elif item.get("type") in ("image", "image_compressed"):
                    label = tr("history.image")
                elif item.get("type") == "file":
                    label = f"[" + tr("history.file_name_prefix", name=Path(item.get('data', '')).name) + "]"
                else:
                    label = tr("history.unknown")
            except Exception:
                label = tr("history.display_error", error="")

            button = ctk.CTkButton(
                self.scrollable_frame,
                text=label,
                command=lambda i=item: self._on_item_selected(i),
                anchor="w",
                fg_color=styles.DEFAULT_BUTTON_FG_COLOR,
                text_color=styles.DEFAULT_BUTTON_TEXT_COLOR,
            )
            button.grid(row=i, column=0, sticky="ew", padx=5, pady=2)
            self._buttons.append(button)

        footer = ctk.CTkFrame(self.main_frame, fg_color="transparent")
        footer.grid(row=1, column=0, sticky="ew", padx=5, pady=(0, 5))
        footer.grid_columnconfigure(0, weight=1)
        footer.grid_columnconfigure(1, weight=1)
        self.more_button = ctk.CTkButton(
            footer,
            text=tr("common.more"),
            command=self._load_more,
            fg_color=styles.DEFAULT_BUTTON_FG_COLOR,
            text_color=styles.DEFAULT_BUTTON_TEXT_COLOR,
        )
        self.more_button.grid(row=0, column=0, padx=(0, 4), sticky="ew")
        cancel_button = ctk.CTkButton(
            footer,
            text=tr("common.cancel"),
            command=self.destroy,
            fg_color=styles.CANCEL_BUTTON_COLOR,
            text_color=styles.CANCEL_BUTTON_TEXT_COLOR,
        )
        cancel_button.grid(row=0, column=1, padx=(4, 0), sticky="ew")

        self.bind("<Escape>", lambda e=None: self.destroy())
        self._update_selection_highlight()

    def destroy(self) -> None:
        if not self._is_destroying:
            self._is_destroying = True
            try:
                if self._on_destroy_callback:
                    self._on_destroy_callback()
            except Exception:
                pass
        return super().destroy()

    def show_at_cursor(self) -> None:
        try:
            x = self.winfo_pointerx()
            y = self.winfo_pointery()
            self.geometry(f"480x360+{x}+{y}")
        except Exception:
            self.geometry("480x360+200+200")
        self.deiconify()
        self.lift(self.master)
        self.focus_force()

    def _on_item_selected(self, item: Dict[str, Any]):
        try:
            self.master.after(1, self.on_select_callback, item)
        finally:
            self.destroy()

    def _update_selection_highlight(self):
        for i, button in enumerate(self._buttons):
            if i == self._current_selection_index:
                button.configure(
                    border_color=styles.HIGHLIGHT_BORDER_COLOR,
                    border_width=styles.HIGHLIGHT_BORDER_WIDTH,
                )
            else:
                button.configure(border_width=0)

    def _load_more(self):
        try:
            rows = db.get_clipboard_items(limit=self._page_limit, offset=self._offset, q=None)
        except Exception:
            rows = []
        # Append
        added = 0
        for r in rows:
            try:
                t = r["type"]
                if t == "text":
                    self._history_items.append({"type": "text", "data": r["text"] or ""})
                elif t == "image":
                    b = r["image_blob"]
                    b64 = base64.b64encode(b).decode("utf-8") if b else ""
                    self._history_items.append({"type": "image", "data": b64})
                elif t == "file":
                    self._history_items.append({"type": "file", "data": r["file_path"] or ""})
                else:
                    # Unknown types as text for safety
                    self._history_items.append({"type": "text", "data": str(r.get("text", "")) if hasattr(r, 'get') else str(r["text"]) if "text" in r.keys() else ""})
                added += 1
            except Exception:
                continue
        # Rebuild simple list for now
        for b in self._buttons:
            try:
                b.destroy()
            except Exception:
                pass
        self._buttons.clear()
        for i, item in enumerate(self._history_items):
            try:
                if item.get("type") == "text":
                    text = item.get("data", "")
                    label = text.replace("\n", " ")[:80] + ("..." if len(text) > 80 else "")
                elif item.get("type") in ("image", "image_compressed"):
                    label = tr("history.image")
                elif item.get("type") == "file":
                    label = f"[" + tr("history.file_name_prefix", name=Path(item.get('data', '')).name) + "]"
                else:
                    label = tr("history.unknown")
            except Exception:
                label = tr("history.display_error", error="")
            button = ctk.CTkButton(
                self.scrollable_frame,
                text=label,
                command=lambda i=item: self._on_item_selected(i),
                anchor="w",
                fg_color=styles.DEFAULT_BUTTON_FG_COLOR,
                text_color=styles.DEFAULT_BUTTON_TEXT_COLOR,
            )
            button.grid(row=i, column=0, sticky="ew", padx=5, pady=2)
            self._buttons.append(button)
        self._offset += added
        # Disable more button if no more rows
        try:
            if added < self._page_limit:
                self.more_button.configure(state="disabled")
        except Exception:
            pass
        self._update_selection_highlight()
