"""
db.py
======

SQLite storage for clipboard items, runs, and matrix artifacts.
Initial implementation focuses on clipboard persistence with a forward‑looking
schema to support runs and matrix data.
"""
from __future__ import annotations

import sqlite3
import threading
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from PIL import Image
from io import BytesIO
import base64
import zlib
import hashlib
import json

# Lazy import of paths to avoid circular import issues when executed as script
try:
    from . import paths  # type: ignore
except Exception:
    import importlib.util, os
    _paths_spec = importlib.util.spec_from_file_location(
        "paths", os.path.join(Path(__file__).parent, "paths.py")
    )
    assert _paths_spec and _paths_spec.loader
    paths = importlib.util.module_from_spec(_paths_spec)
    _paths_spec.loader.exec_module(paths)  # type: ignore


# Use re-entrant lock because some DB helpers call other helpers that also
# acquire the DB lock (e.g., get_matrix_tab_content -> get_clipboard_item).
# A normal Lock would deadlock in those cases.
_DB_LOCK = threading.RLock()
_CONN: Optional[sqlite3.Connection] = None


def _db_path() -> Path:
    return paths.get_data_dir() / "gem_clip.sqlite3"


def _connect() -> sqlite3.Connection:
    global _CONN
    if _CONN is not None:
        return _CONN
    db_file = _db_path()
    db_file.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_file), check_same_thread=False)
    conn.row_factory = sqlite3.Row
    with conn:
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("PRAGMA synchronous=NORMAL;")
        conn.execute("PRAGMA foreign_keys=ON;")
    _CONN = conn
    _migrate(conn)
    return conn


def _migrate(conn: sqlite3.Connection) -> None:
    with conn:
        conn.execute(
            "CREATE TABLE IF NOT EXISTS schema_version (version INTEGER NOT NULL)"
        )
        cur = conn.execute("SELECT version FROM schema_version ORDER BY version DESC LIMIT 1")
        row = cur.fetchone()
        current = row[0] if row else 0
        if current < 1:
            _apply_v1(conn)
            conn.execute("DELETE FROM schema_version")
            conn.execute("INSERT INTO schema_version(version) VALUES (1)")


def _apply_v1(conn: sqlite3.Connection) -> None:
    # Clipboard items
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS clipboard_item (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            type TEXT NOT NULL, -- text | image | file
            text TEXT,
            image_blob BLOB,
            image_thumb BLOB,
            file_path TEXT,
            file_name TEXT,
            content_hash TEXT NOT NULL UNIQUE,
            source TEXT
        );
        CREATE INDEX IF NOT EXISTS idx_clipboard_item_created_at ON clipboard_item(created_at DESC);
        CREATE INDEX IF NOT EXISTS idx_clipboard_item_type ON clipboard_item(type);

        -- Runs
        CREATE TABLE IF NOT EXISTS run (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            run_type TEXT NOT NULL,
            started_at TEXT NOT NULL,
            finished_at TEXT,
            status TEXT,
            api_model_id TEXT,
            parameters_json TEXT,
            safety_info_json TEXT,
            pricing_input_tokens INTEGER,
            pricing_output_tokens INTEGER,
            pricing_cost_usd REAL,
            prompt_snapshot_json TEXT,
            user_note TEXT
        );
        CREATE INDEX IF NOT EXISTS idx_run_started_at ON run(started_at DESC);
        CREATE INDEX IF NOT EXISTS idx_run_type ON run(run_type);

        CREATE TABLE IF NOT EXISTS run_input (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            run_id INTEGER NOT NULL REFERENCES run(id) ON DELETE CASCADE,
            clipboard_item_id INTEGER REFERENCES clipboard_item(id) ON DELETE SET NULL,
            extra_files_json TEXT,
            is_refine_prev_output INTEGER
        );

        CREATE TABLE IF NOT EXISTS run_output (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            run_id INTEGER NOT NULL REFERENCES run(id) ON DELETE CASCADE,
            prompt_id TEXT,
            input_id INTEGER REFERENCES run_input(id) ON DELETE SET NULL,
            content_text TEXT,
            content_blob BLOB,
            error_json TEXT,
            rating INTEGER,
            tags TEXT,
            copied_to_clipboard_at TEXT
        );
        CREATE INDEX IF NOT EXISTS idx_run_output_run ON run_output(run_id);

        CREATE TABLE IF NOT EXISTS feedback (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            run_output_id INTEGER NOT NULL REFERENCES run_output(id) ON DELETE CASCADE,
            rating INTEGER,
            label TEXT,
            comment TEXT,
            created_at TEXT NOT NULL
        );

        -- Matrix scaffolding (minimal v1)
        CREATE TABLE IF NOT EXISTS matrix_session (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            meta_json TEXT
        );
        CREATE TABLE IF NOT EXISTS matrix_tab (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id INTEGER NOT NULL REFERENCES matrix_session(id) ON DELETE CASCADE,
            name TEXT NOT NULL,
            order_index INTEGER NOT NULL,
            created_at TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS matrix_input (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            tab_id INTEGER NOT NULL REFERENCES matrix_tab(id) ON DELETE CASCADE,
            order_index INTEGER NOT NULL,
            clipboard_item_id INTEGER REFERENCES clipboard_item(id) ON DELETE SET NULL,
            attach_files_json TEXT
        );
        CREATE TABLE IF NOT EXISTS matrix_prompt (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            tab_id INTEGER NOT NULL REFERENCES matrix_tab(id) ON DELETE CASCADE,
            prompt_snapshot_json TEXT NOT NULL,
            order_index INTEGER NOT NULL
        );
        CREATE TABLE IF NOT EXISTS matrix_cell (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            tab_id INTEGER NOT NULL REFERENCES matrix_tab(id) ON DELETE CASCADE,
            input_id INTEGER NOT NULL REFERENCES matrix_input(id) ON DELETE CASCADE,
            prompt_id INTEGER NOT NULL REFERENCES matrix_prompt(id) ON DELETE CASCADE,
            checked INTEGER
        );
        CREATE TABLE IF NOT EXISTS matrix_result (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            tab_id INTEGER NOT NULL REFERENCES matrix_tab(id) ON DELETE CASCADE,
            input_id INTEGER NOT NULL REFERENCES matrix_input(id) ON DELETE CASCADE,
            prompt_id INTEGER NOT NULL REFERENCES matrix_prompt(id) ON DELETE CASCADE,
            run_id INTEGER REFERENCES run(id) ON DELETE SET NULL,
            output_id INTEGER REFERENCES run_output(id) ON DELETE SET NULL,
            final_text TEXT,
            full_output_json TEXT,
            error_json TEXT,
            created_at TEXT NOT NULL
        );
        CREATE INDEX IF NOT EXISTS idx_matrix_result_triplet ON matrix_result(tab_id, input_id, prompt_id);

        CREATE TABLE IF NOT EXISTS matrix_row_summary (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            tab_id INTEGER NOT NULL REFERENCES matrix_tab(id) ON DELETE CASCADE,
            input_id INTEGER NOT NULL REFERENCES matrix_input(id) ON DELETE CASCADE,
            run_id INTEGER REFERENCES run(id) ON DELETE SET NULL,
            output_id INTEGER REFERENCES run_output(id) ON DELETE SET NULL,
            text TEXT,
            created_at TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS matrix_col_summary (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            tab_id INTEGER NOT NULL REFERENCES matrix_tab(id) ON DELETE CASCADE,
            prompt_id INTEGER NOT NULL REFERENCES matrix_prompt(id) ON DELETE CASCADE,
            run_id INTEGER REFERENCES run(id) ON DELETE SET NULL,
            output_id INTEGER REFERENCES run_output(id) ON DELETE SET NULL,
            text TEXT,
            created_at TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS matrix_final_summary (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            tab_id INTEGER NOT NULL REFERENCES matrix_tab(id) ON DELETE CASCADE,
            run_id INTEGER REFERENCES run(id) ON DELETE SET NULL,
            output_id INTEGER REFERENCES run_output(id) ON DELETE SET NULL,
            text TEXT,
            created_at TEXT NOT NULL
        );
        """
    )


def _now_iso() -> str:
    """Return current UTC time as ISO8601 string with 'Z' suffix.

    Uses timezone-aware datetime to avoid deprecation of datetime.utcnow().
    """
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def _thumb_from_image_bytes(png_bytes: bytes, size: int = 128) -> bytes:
    try:
        with Image.open(BytesIO(png_bytes)) as im:
            im = im.convert("RGB")
            im.thumbnail((size, size))
            buf = BytesIO()
            im.save(buf, format="PNG")
            return buf.getvalue()
    except Exception:
        return b""


def _normalize_image_bytes_from_history(item: Dict[str, Any]) -> Optional[bytes]:
    """Decode base64 from history item to PNG bytes, handling 'image'/'image_compressed'."""
    try:
        raw_b64 = item.get("data", "")
        raw = base64.b64decode(raw_b64)
        # Try decompress for 'image_compressed'
        if item.get("type") == "image_compressed":
            try:
                raw = zlib.decompress(raw)
            except Exception:
                pass
        # Ensure PNG
        with Image.open(BytesIO(raw)) as im:
            if im.mode != "RGB":
                im = im.convert("RGB")
            buf = BytesIO()
            im.save(buf, format="PNG")
            return buf.getvalue()
    except Exception:
        return None


def _content_hash_for_item(item: Any) -> Optional[str]:
    try:
        if isinstance(item, str):
            s = item.strip()
            if not s:
                return None
            return hashlib.sha256(("text:" + s).encode("utf-8")).hexdigest()
        if isinstance(item, dict):
            t = item.get("type")
            d = item.get("data", "")
            if t == "text":
                return hashlib.sha256(("text:" + str(d)).encode("utf-8")).hexdigest()
            if t in ("image", "image_compressed"):
                png = _normalize_image_bytes_from_history(item)
                if png:
                    return hashlib.sha256(b"img:" + png).hexdigest()
                # fallback to raw base64
                return hashlib.sha256(("imgb64:" + str(d)).encode("utf-8")).hexdigest()
            if t == "file":
                return hashlib.sha256(("file:" + str(d)).encode("utf-8")).hexdigest()
    except Exception:
        return None
    return None


def save_clipboard_item(item: Any, source: str = "user_clipboard") -> Optional[int]:
    """Insert clipboard item if not exists; return existing/new id.

    Supports item as str or dict {type,data} from in‑memory history.
    """
    h = _content_hash_for_item(item)
    if not h:
        return None
    conn = _connect()
    with _DB_LOCK:
        cur = conn.execute("SELECT id FROM clipboard_item WHERE content_hash=?", (h,))
        row = cur.fetchone()
        if row:
            # Update updated_at
            conn.execute("UPDATE clipboard_item SET updated_at=? WHERE id=?", (_now_iso(), row[0]))
            return int(row[0])
        now = _now_iso()
        fields = {
            "created_at": now,
            "updated_at": now,
            "type": None,
            "text": None,
            "image_blob": None,
            "image_thumb": None,
            "file_path": None,
            "file_name": None,
            "content_hash": h,
            "source": source,
        }
        if isinstance(item, str):
            fields["type"] = "text"
            fields["text"] = item
        elif isinstance(item, dict):
            t = item.get("type")
            d = item.get("data")
            if t == "text":
                fields["type"] = "text"
                fields["text"] = str(d)
            elif t in ("image", "image_compressed"):
                fields["type"] = "image"
                png = _normalize_image_bytes_from_history(item)
                if png:
                    fields["image_blob"] = png
                    fields["image_thumb"] = _thumb_from_image_bytes(png)
            elif t == "file":
                fields["type"] = "file"
                p = str(d)
                fields["file_path"] = p
                try:
                    fields["file_name"] = Path(p).name
                except Exception:
                    fields["file_name"] = p
            else:
                fields["type"] = str(t or "unknown")
                fields["text"] = str(d)
        else:
            return None

        placeholders = ",".join(["?" for _ in fields])
        columns = ",".join(fields.keys())
        conn.execute(
            f"INSERT INTO clipboard_item({columns}) VALUES ({placeholders})",
            tuple(fields.values()),
        )
        return int(conn.execute("SELECT last_insert_rowid()").fetchone()[0])


def get_clipboard_items(limit: int = 50, offset: int = 0, item_type: Optional[str] = None, q: Optional[str] = None) -> List[sqlite3.Row]:
    conn = _connect()
    where: List[str] = []
    params: List[Any] = []
    if item_type:
        where.append("type=?")
        params.append(item_type)
    if q:
        where.append("(text LIKE ? OR file_name LIKE ? OR file_path LIKE ?)")
        like = f"%{q}%"
        params.extend([like, like, like])
    where_sql = ("WHERE " + " AND ".join(where)) if where else ""
    sql = f"SELECT * FROM clipboard_item {where_sql} ORDER BY created_at DESC LIMIT ? OFFSET ?"
    params.extend([limit, offset])
    with _DB_LOCK:
        cur = conn.execute(sql, tuple(params))
        return cur.fetchall()


# --------------------- Run persistence helpers ---------------------

def create_run(run_type: str, api_model_id: str, parameters: Dict[str, Any], prompt_snapshot: Dict[str, Any]) -> int:
    conn = _connect()
    with _DB_LOCK, conn:
        cur = conn.execute(
            """
            INSERT INTO run(run_type, started_at, status, api_model_id, parameters_json, prompt_snapshot_json)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                run_type,
                _now_iso(),
                "running",
                api_model_id,
                json.dumps(parameters, ensure_ascii=False),
                json.dumps(prompt_snapshot, ensure_ascii=False),
            ),
        )
        return int(cur.lastrowid)


def finish_run(run_id: int, status: str, input_tokens: Optional[int] = None, output_tokens: Optional[int] = None, cost_usd: Optional[float] = None) -> None:
    conn = _connect()
    with _DB_LOCK, conn:
        conn.execute(
            """
            UPDATE run
            SET finished_at=?, status=?, pricing_input_tokens=COALESCE(?, pricing_input_tokens),
                pricing_output_tokens=COALESCE(?, pricing_output_tokens), pricing_cost_usd=COALESCE(?, pricing_cost_usd)
            WHERE id=?
            """,
            (_now_iso(), status, input_tokens, output_tokens, cost_usd, run_id),
        )


def add_run_input(run_id: int, clipboard_item_id: Optional[int], extra_files: Optional[List[str]] = None, is_refine_prev_output: bool = False) -> int:
    conn = _connect()
    with _DB_LOCK, conn:
        cur = conn.execute(
            """
            INSERT INTO run_input(run_id, clipboard_item_id, extra_files_json, is_refine_prev_output)
            VALUES (?, ?, ?, ?)
            """,
            (run_id, clipboard_item_id, json.dumps(extra_files or [], ensure_ascii=False), 1 if is_refine_prev_output else 0),
        )
        return int(cur.lastrowid)


def add_run_output(
    run_id: int,
    prompt_id: Optional[str],
    input_id: Optional[int],
    content_text: Optional[str] = None,
    error_json: Optional[Dict[str, Any]] = None,
    content_blob: Optional[bytes] = None,
) -> int:
    conn = _connect()
    with _DB_LOCK, conn:
        cur = conn.execute(
            """
            INSERT INTO run_output(run_id, prompt_id, input_id, content_text, error_json, content_blob)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                run_id,
                prompt_id,
                input_id,
                content_text,
                json.dumps(error_json, ensure_ascii=False) if error_json else None,
                content_blob,
            ),
        )
        return int(cur.lastrowid)


def mark_output_copied(output_id: int) -> None:
    conn = _connect()
    with _DB_LOCK, conn:
        conn.execute(
            "UPDATE run_output SET copied_to_clipboard_at=? WHERE id=?",
            (_now_iso(), output_id),
        )


# --------------------- Matrix helpers ---------------------

def create_matrix_session(name: str) -> int:
    conn = _connect()
    now = _now_iso()
    with _DB_LOCK, conn:
        cur = conn.execute(
            "INSERT INTO matrix_session(name, created_at, updated_at, meta_json) VALUES (?, ?, ?, ?)",
            (name, now, now, None),
        )
        return int(cur.lastrowid)


def create_matrix_tab(session_id: int, name: str, order_index: int = 0) -> int:
    conn = _connect()
    with _DB_LOCK, conn:
        cur = conn.execute(
            "INSERT INTO matrix_tab(session_id, name, order_index, created_at) VALUES (?, ?, ?, ?)",
            (session_id, name, order_index, _now_iso()),
        )
        tab_id = int(cur.lastrowid)
        try:
            # Touch parent session updated_at
            conn.execute(
                "UPDATE matrix_session SET updated_at=? WHERE id=?",
                (_now_iso(), session_id),
            )
        except Exception:
            pass
        return tab_id


def upsert_matrix_input(tab_id: int, order_index: int, clipboard_item_id: Optional[int], attach_files: Optional[List[str]] = None) -> int:
    conn = _connect()
    with _DB_LOCK, conn:
        # Try find existing
        cur = conn.execute(
            "SELECT id FROM matrix_input WHERE tab_id=? AND order_index=?",
            (tab_id, order_index),
        )
        row = cur.fetchone()
        if row:
            attach_json = json.dumps(attach_files, ensure_ascii=False) if attach_files is not None else None
            conn.execute(
                "UPDATE matrix_input SET clipboard_item_id=COALESCE(?, clipboard_item_id), attach_files_json=COALESCE(?, attach_files_json) WHERE id=?",
                (clipboard_item_id, attach_json, row[0]),
            )
            mid = int(row[0])
            try:
                conn.execute(
                    "UPDATE matrix_session SET updated_at=? WHERE id=(SELECT session_id FROM matrix_tab WHERE id=?)",
                    (_now_iso(), tab_id),
                )
            except Exception:
                pass
            return mid
        cur = conn.execute(
            "INSERT INTO matrix_input(tab_id, order_index, clipboard_item_id, attach_files_json) VALUES (?, ?, ?, ?)",
            (tab_id, order_index, clipboard_item_id, json.dumps(attach_files or [], ensure_ascii=False)),
        )
        mid = int(cur.lastrowid)
        try:
            conn.execute(
                "UPDATE matrix_session SET updated_at=? WHERE id=(SELECT session_id FROM matrix_tab WHERE id=?)",
                (_now_iso(), tab_id),
            )
        except Exception:
            pass
        return mid


def upsert_matrix_prompt(tab_id: int, order_index: int, prompt_snapshot: Dict[str, Any]) -> int:
    conn = _connect()
    with _DB_LOCK, conn:
        cur = conn.execute(
            "SELECT id FROM matrix_prompt WHERE tab_id=? AND order_index=?",
            (tab_id, order_index),
        )
        row = cur.fetchone()
        if row:
            conn.execute(
                "UPDATE matrix_prompt SET prompt_snapshot_json=? WHERE id=?",
                (json.dumps(prompt_snapshot, ensure_ascii=False), row[0]),
            )
            pid = int(row[0])
            try:
                conn.execute(
                    "UPDATE matrix_session SET updated_at=? WHERE id=(SELECT session_id FROM matrix_tab WHERE id=?)",
                    (_now_iso(), tab_id),
                )
            except Exception:
                pass
            return pid
        cur = conn.execute(
            "INSERT INTO matrix_prompt(tab_id, prompt_snapshot_json, order_index) VALUES (?, ?, ?)",
            (tab_id, json.dumps(prompt_snapshot, ensure_ascii=False), order_index),
        )
        pid = int(cur.lastrowid)
        try:
            conn.execute(
                "UPDATE matrix_session SET updated_at=? WHERE id=(SELECT session_id FROM matrix_tab WHERE id=?)",
                (_now_iso(), tab_id),
            )
        except Exception:
            pass
        return pid


def add_matrix_result(tab_id: int, input_id: int, prompt_id: int, run_id: Optional[int], output_id: Optional[int], final_text: str, full_output_json: Optional[Dict[str, Any]] = None, error_json: Optional[Dict[str, Any]] = None) -> int:
    conn = _connect()
    with _DB_LOCK, conn:
        cur = conn.execute(
            """
            INSERT INTO matrix_result(tab_id, input_id, prompt_id, run_id, output_id, final_text, full_output_json, error_json, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (tab_id, input_id, prompt_id, run_id, output_id, final_text, json.dumps(full_output_json, ensure_ascii=False) if full_output_json else None, json.dumps(error_json, ensure_ascii=False) if error_json else None, _now_iso()),
        )
        rid = int(cur.lastrowid)
        try:
            conn.execute(
                "UPDATE matrix_session SET updated_at=? WHERE id=(SELECT session_id FROM matrix_tab WHERE id=?)",
                (_now_iso(), tab_id),
            )
        except Exception:
            pass
        return rid


def add_matrix_row_summary(tab_id: int, input_id: int, text: str, run_id: Optional[int] = None, output_id: Optional[int] = None) -> int:
    conn = _connect()
    with _DB_LOCK, conn:
        cur = conn.execute(
            "INSERT INTO matrix_row_summary(tab_id, input_id, run_id, output_id, text, created_at) VALUES (?, ?, ?, ?, ?, ?)",
            (tab_id, input_id, run_id, output_id, text, _now_iso()),
        )
        sid = int(cur.lastrowid)
        try:
            conn.execute(
                "UPDATE matrix_session SET updated_at=? WHERE id=(SELECT session_id FROM matrix_tab WHERE id=?)",
                (_now_iso(), tab_id),
            )
        except Exception:
            pass
        return sid


def add_matrix_col_summary(tab_id: int, prompt_id: int, text: str, run_id: Optional[int] = None, output_id: Optional[int] = None) -> int:
    conn = _connect()
    with _DB_LOCK, conn:
        cur = conn.execute(
            "INSERT INTO matrix_col_summary(tab_id, prompt_id, run_id, output_id, text, created_at) VALUES (?, ?, ?, ?, ?, ?)",
            (tab_id, prompt_id, run_id, output_id, text, _now_iso()),
        )
        sid = int(cur.lastrowid)
        try:
            conn.execute(
                "UPDATE matrix_session SET updated_at=? WHERE id=(SELECT session_id FROM matrix_tab WHERE id=?)",
                (_now_iso(), tab_id),
            )
        except Exception:
            pass
        return sid


def add_matrix_final_summary(tab_id: int, text: str, run_id: Optional[int] = None, output_id: Optional[int] = None) -> int:
    conn = _connect()
    with _DB_LOCK, conn:
        cur = conn.execute(
            "INSERT INTO matrix_final_summary(tab_id, run_id, output_id, text, created_at) VALUES (?, ?, ?, ?, ?)",
            (tab_id, run_id, output_id, text, _now_iso()),
        )
        sid = int(cur.lastrowid)
        try:
            conn.execute(
                "UPDATE matrix_session SET updated_at=? WHERE id=(SELECT session_id FROM matrix_tab WHERE id=?)",
                (_now_iso(), tab_id),
            )
        except Exception:
            pass
        return sid


# --------------------- Matrix browsing helpers ---------------------

def list_matrix_sessions() -> List[sqlite3.Row]:
    conn = _connect()
    with _DB_LOCK:
        cur = conn.execute("SELECT id, name, created_at, updated_at FROM matrix_session ORDER BY created_at DESC")
        return cur.fetchall()


def list_matrix_tabs(session_id: int) -> List[sqlite3.Row]:
    conn = _connect()
    with _DB_LOCK:
        cur = conn.execute(
            "SELECT id, name, order_index, created_at FROM matrix_tab WHERE session_id=? ORDER BY order_index ASC, id ASC",
            (session_id,),
        )
        return cur.fetchall()


def get_clipboard_item(item_id: int) -> Optional[sqlite3.Row]:
    conn = _connect()
    with _DB_LOCK:
        cur = conn.execute("SELECT * FROM clipboard_item WHERE id=?", (item_id,))
        return cur.fetchone()


def get_matrix_tab_content(tab_id: int) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Return (inputs, prompts) for a tab.

    inputs: List of normalized dicts like {'type': 'text'|'image'|'file', 'data': ...}
    prompts: List of prompt snapshot dicts (as stored)
    """
    conn = _connect()
    inputs: List[Dict[str, Any]] = []
    prompts: List[Dict[str, Any]] = []
    with _DB_LOCK:
        # Inputs
        cur = conn.execute(
            "SELECT order_index, clipboard_item_id, attach_files_json FROM matrix_input WHERE tab_id=? ORDER BY order_index ASC",
            (tab_id,),
        )
        rows = cur.fetchall()
        for r in rows:
            item = None
            if r['clipboard_item_id']:
                ci = get_clipboard_item(int(r['clipboard_item_id']))
                if ci:
                    if ci['type'] == 'text':
                        item = {"type": "text", "data": ci['text'] or ''}
                    elif ci['type'] == 'image':
                        b = ci['image_blob']
                        b64 = base64.b64encode(b).decode('utf-8') if b else ''
                        item = {"type": "image", "data": b64}
                    elif ci['type'] == 'file':
                        item = {"type": "file", "data": ci['file_path'] or ''}
            if not item:
                item = {"type": "text", "data": ""}
            inputs.append(item)
        # Prompts
        cur = conn.execute(
            "SELECT order_index, prompt_snapshot_json FROM matrix_prompt WHERE tab_id=? ORDER BY order_index ASC",
            (tab_id,),
        )
        rows = cur.fetchall()
        for r in rows:
            try:
                snap = json.loads(r['prompt_snapshot_json'] or '{}')
            except Exception:
                snap = {}
            prompts.append(snap)
    return inputs, prompts


def rename_matrix_session(session_id: int, new_name: str) -> None:
    conn = _connect()
    with _DB_LOCK, conn:
        conn.execute("UPDATE matrix_session SET name=?, updated_at=? WHERE id=?", (new_name, _now_iso(), session_id))


def delete_matrix_session(session_id: int) -> None:
    conn = _connect()
    with _DB_LOCK, conn:
        conn.execute("DELETE FROM matrix_session WHERE id=?", (session_id,))


def rename_matrix_tab(tab_id: int, new_name: str) -> None:
    conn = _connect()
    with _DB_LOCK, conn:
        conn.execute("UPDATE matrix_tab SET name=? WHERE id=?", (new_name, tab_id))
        try:
            conn.execute(
                "UPDATE matrix_session SET updated_at=? WHERE id=(SELECT session_id FROM matrix_tab WHERE id=?)",
                (_now_iso(), tab_id),
            )
        except Exception:
            pass


def delete_matrix_tab(tab_id: int) -> None:
    conn = _connect()
    with _DB_LOCK, conn:
        # Touch session updated_at before delete (we still can resolve session via subselect)
        try:
            conn.execute(
                "UPDATE matrix_session SET updated_at=? WHERE id=(SELECT session_id FROM matrix_tab WHERE id=?)",
                (_now_iso(), tab_id),
            )
        except Exception:
            pass
        conn.execute("DELETE FROM matrix_tab WHERE id=?", (tab_id,))


def get_matrix_results_positions(tab_id: int) -> List[Tuple[int, int, str]]:
    """Return list of (row_index, col_index, final_text) for a tab."""
    conn = _connect()
    with _DB_LOCK:
        cur = conn.execute(
            """
            SELECT mi.order_index AS r, mp.order_index AS c, mr.final_text AS t
            FROM matrix_result mr
            JOIN matrix_input mi ON mi.id = mr.input_id
            JOIN matrix_prompt mp ON mp.id = mr.prompt_id
            WHERE mr.tab_id=?
            ORDER BY mi.order_index ASC, mp.order_index ASC, mr.id ASC
            """,
            (tab_id,),
        )
        return [(int(r[0]), int(r[1]), r[2] or '') for r in cur.fetchall()]


def upsert_matrix_cell(tab_id: int, input_id: int, prompt_id: int, checked: bool) -> int:
    """Upsert a matrix_cell row for the given (tab,input,prompt) with checked state.

    Returns the row id.
    """
    conn = _connect()
    with _DB_LOCK, conn:
        cur = conn.execute(
            "SELECT id FROM matrix_cell WHERE tab_id=? AND input_id=? AND prompt_id=?",
            (tab_id, input_id, prompt_id),
        )
        row = cur.fetchone()
        if row:
            conn.execute(
                "UPDATE matrix_cell SET checked=? WHERE id=?",
                (1 if checked else 0, int(row[0])),
            )
            rid = int(row[0])
        else:
            cur = conn.execute(
                "INSERT INTO matrix_cell(tab_id, input_id, prompt_id, checked) VALUES (?, ?, ?, ?)",
                (tab_id, input_id, prompt_id, 1 if checked else 0),
            )
            rid = int(cur.lastrowid)
        try:
            conn.execute(
                "UPDATE matrix_session SET updated_at=? WHERE id=(SELECT session_id FROM matrix_tab WHERE id=?)",
                (_now_iso(), tab_id),
            )
        except Exception:
            pass
        return rid


def get_matrix_checked_positions(tab_id: int) -> List[Tuple[int, int]]:
    """Return (row_index, col_index) for checked cells in a tab."""
    conn = _connect()
    with _DB_LOCK:
        cur = conn.execute(
            """
            SELECT mi.order_index AS r, mp.order_index AS c
            FROM matrix_cell mc
            JOIN matrix_input mi ON mi.id = mc.input_id
            JOIN matrix_prompt mp ON mp.id = mc.prompt_id
            WHERE mc.tab_id=? AND COALESCE(mc.checked,0) != 0
            ORDER BY mi.order_index ASC, mp.order_index ASC
            """,
            (tab_id,),
        )
        return [(int(r[0]), int(r[1])) for r in cur.fetchall()]


def get_matrix_row_summaries(tab_id: int) -> List[Tuple[int, str]]:
    conn = _connect()
    with _DB_LOCK:
        cur = conn.execute(
            """
            SELECT mi.order_index AS r, mrs.text AS t
            FROM matrix_row_summary mrs
            JOIN matrix_input mi ON mi.id = mrs.input_id
            WHERE mrs.tab_id=?
            ORDER BY mi.order_index ASC, mrs.id DESC
            """,
            (tab_id,),
        )
        rows = cur.fetchall()
        # Deduplicate by row index keeping first occurrence (latest by id)
        seen = set()
        out: List[Tuple[int,str]] = []
        for r in rows:
            ri = int(r[0])
            if ri in seen:
                continue
            seen.add(ri)
            out.append((ri, r[1] or ''))
        return out


def get_matrix_col_summaries(tab_id: int) -> List[Tuple[int, str]]:
    conn = _connect()
    with _DB_LOCK:
        cur = conn.execute(
            """
            SELECT mp.order_index AS c, mcs.text AS t
            FROM matrix_col_summary mcs
            JOIN matrix_prompt mp ON mp.id = mcs.prompt_id
            WHERE mcs.tab_id=?
            ORDER BY mp.order_index ASC, mcs.id DESC
            """,
            (tab_id,),
        )
        rows = cur.fetchall()
        seen = set()
        out: List[Tuple[int,str]] = []
        for r in rows:
            ci = int(r[0])
            if ci in seen:
                continue
            seen.add(ci)
            out.append((ci, r[1] or ''))
        return out


def get_matrix_final_summary(tab_id: int) -> Optional[str]:
    conn = _connect()
    with _DB_LOCK:
        cur = conn.execute(
            "SELECT text FROM matrix_final_summary WHERE tab_id=? ORDER BY id DESC LIMIT 1",
            (tab_id,),
        )
        row = cur.fetchone()
        return row[0] if row else None


def get_matrix_tab_meta(tab_id: int) -> Optional[Dict[str, Any]]:
    """Return basic metadata for a tab: session_id, name, order_index.

    This helper avoids leaking sqlite details to UI code.
    """
    conn = _connect()
    with _DB_LOCK:
        cur = conn.execute(
            "SELECT session_id, name, order_index FROM matrix_tab WHERE id=?",
            (tab_id,),
        )
        row = cur.fetchone()
        if not row:
            return None
        return {"session_id": int(row[0]), "name": row[1], "order_index": int(row[2])}
