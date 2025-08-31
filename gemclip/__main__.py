"""Gem Clip module runner.

Run with ``python -m gemclip``.
Optional flags allow quick overrides for language, theme, and log level.
"""

from __future__ import annotations


def main() -> None:
    import argparse
    import logging
    from gemclip.core import setup_logging
    from config_manager import load_config, save_config
    from app import ClipboardToolApp

    parser = argparse.ArgumentParser(prog="gemclip", add_help=True)
    parser.add_argument("--lang", choices=["auto", "en", "ja"], help="UI language override")
    parser.add_argument("--theme", choices=["system", "light", "dark"], help="Theme override")
    parser.add_argument(
        "--log-level",
        choices=["debug", "info", "warning", "error"],
        default="info",
        help="Logging level",
    )
    args = parser.parse_args()

    level_map = {
        "debug": logging.DEBUG,
        "info": logging.INFO,
        "warning": logging.WARNING,
        "error": logging.ERROR,
    }
    setup_logging(level_map.get(args.log_level, logging.INFO))

    # Apply lightweight overrides to config before app init
    try:
        cfg = load_config()
        if cfg:
            changed = False
            if args.lang is not None and getattr(cfg, "language", None) != args.lang:
                cfg.language = args.lang
                changed = True
            if args.theme is not None and getattr(cfg, "theme_mode", None) != args.theme:
                cfg.theme_mode = args.theme  # type: ignore
                changed = True
            if changed:
                save_config(cfg)
    except Exception:
        pass

    app_instance = ClipboardToolApp()
    app_instance.run()


if __name__ == "__main__":
    main()
