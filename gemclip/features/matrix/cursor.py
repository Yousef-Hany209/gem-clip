"""Cursor update helpers for Matrix window.

Updates the window cursor based on proximity to borders.
"""
from __future__ import annotations


def start_cursor_monitoring(win) -> None:
    def check_cursor():
        if getattr(win, "_is_closing", False) or not win.winfo_exists():
            return
        try:
            mouse_x = win.winfo_pointerx()
            mouse_y = win.winfo_pointery()
            window_x = win.winfo_rootx()
            window_y = win.winfo_rooty()
            width = win.winfo_width()
            height = win.winfo_height()
            rel_x = mouse_x - window_x
            rel_y = mouse_y - window_y
            if 0 <= rel_x <= width and 0 <= rel_y <= height:
                update_cursor_direct(win, rel_x, rel_y)
            else:
                if win.cget("cursor") != "":
                    win.configure(cursor="")
        except Exception:
            try:
                win.configure(cursor="")
            except Exception:
                pass
        if win.winfo_exists():
            win._cursor_update_job = win.after(100, check_cursor)

    check_cursor()


def update_cursor_direct(win, x, y) -> None:
    try:
        border_width = 8
        width = win.winfo_width()
        height = win.winfo_height()
        cursor_type = ""
        on_top_border = y < border_width
        on_bottom_border = y > height - border_width
        on_left_border = x < border_width
        on_right_border = x > width - border_width
        if on_top_border and on_left_border:
            cursor_type = "top_left_corner"
        elif on_top_border and on_right_border:
            cursor_type = "top_right_corner"
        elif on_bottom_border and on_left_border:
            cursor_type = "bottom_left_corner"
        elif on_bottom_border and on_right_border:
            cursor_type = "bottom_right_corner"
        elif on_top_border or on_bottom_border:
            cursor_type = "sb_v_double_arrow"
        elif on_left_border or on_right_border:
            cursor_type = "sb_h_double_arrow"
        if win.cget("cursor") != cursor_type:
            try:
                win.configure(cursor=cursor_type)
            except Exception:
                pass
    except Exception:
        pass

