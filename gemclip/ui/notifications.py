"""Notification service wrapper for popup notifications.

Encapsulates NotificationPopup creation/update and keeps last instance.
"""
from __future__ import annotations

from typing import Optional, Literal

from ui_components import NotificationPopup


class NotificationService:
    def __init__(self, parent_app):
        self._parent_app = parent_app
        self._popup: Optional[NotificationPopup] = None

    def show(self, title: str, message: str, level: Literal["info", "warning", "error", "success"] = "info", duration_ms: Optional[int] = 3000):
        if self._popup and self._popup.winfo_exists():
            self._popup.reconfigure(title, message, level, duration_ms)
        else:
            self._popup = NotificationPopup(
                title=title,
                message=message,
                parent_app=self._parent_app,
                level=level,
                on_destroy_callback=self._on_popup_destroy,
            )
            self._popup.show_at_cursor(title, message, level, duration_ms)

    def update_message(self, chunk: str):
        if self._popup and self._popup.winfo_exists():
            self._popup.update_message(chunk)

    def _on_popup_destroy(self):
        self._popup = None

