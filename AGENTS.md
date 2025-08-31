# Repository Guidelines

## Project Structure & Module Organization
- Source: root `*.py` (e.g., `main.py`, `app.py`, `agent.py`, `db.py`, `paths.py`, `ui_components.py`).
- Package: `gemclip/` (feature, infra, and UI helpers).
- Assets and config: `locales/` (i18n JSON), `prompts/` and `prompt_set/` (prompt templates), icons and media in repo root.
- Entry point: run `main.py` (initializes logging and launches the CustomTkinter UI).
- Data/config paths: resolved via `paths.py` under `%APPDATA%/Gem Clip/` on Windows (e.g., `config.json`, `logs/`, `data/`).

## Build, Test, and Development Commands
- Run app: `python -m gemclip` — starts the desktop UI (preferred).
  - Options: `--lang auto|en|ja`, `--theme system|light|dark`, `--log-level info|debug|...`
- Legacy: `python main.py` — forwards to the same launcher.
- Run tests: `python -m pytest -q` — executes `test_*.py` (uses pytest fixtures like `tmp_path`).
- Lint/format: follow PEP 8; if you use tools locally, prefer `black` and `ruff` (no repo config committed).

## Coding Style & Naming Conventions
- Python: 4-space indentation, type hints where practical, docstrings for public functions.
- Names: `snake_case` for functions/variables, `PascalCase` for classes, `UPPER_SNAKE` for constants.
- Files: module names `snake_case.py`; JSON files named descriptively (may include locale-specific names as in `prompts/`).

## Testing Guidelines
- Framework: pytest. Place tests alongside root (e.g., `test_db.py`) or as `tests/` if it grows.
- Names: files `test_*.py`, tests `test_*`.
- Focus: database (`db.py`), path handling (`paths.py`), and critical agent logic. Use `tmp_path` and env overrides as in tests to avoid writing to user dirs.

## Commit & Pull Request Guidelines
- Commits: use Conventional Commits where possible: `feat:`, `fix:`, `refactor:`, `test:`, `docs:`.
- PRs: include a concise description, linked issues, test notes, and screenshots/GIFs for UI changes. Note any schema or config migrations.

## Security & Configuration Tips
- Configuration lives under `%APPDATA%/Gem Clip/`. Do not hardcode credentials or user paths.
- External JSONs (`models.json`, `api_price.json`) are loaded at runtime; keep formats stable and avoid secrets.
- Image/clipboard handling uses Pillow; validate inputs and guard errors.

## i18n & UI Notes
- Text strings come from `locales/*.json` via `i18n.py`. When adding UI, add keys for both `en.json` and `ja.json`.
- Keep UI responsive and accessible; follow existing styles in `styles.py` and widgets in `ui_components.py`.

最後の回答は必ず日本語で行うこと。
