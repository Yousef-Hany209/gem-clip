import os
import sys
import time
import importlib


def _prepare_tmp_env(tmp_path):
    # Point app data dirs to tmp to avoid writing to user dirs
    os.environ["XDG_CONFIG_HOME"] = str(tmp_path)
    os.environ["HOME"] = str(tmp_path)
    os.environ["APPDATA"] = str(tmp_path)
    # Reload modules to pick up new env
    if "paths" in sys.modules:
        importlib.reload(sys.modules["paths"])  # type: ignore
    if "db" in sys.modules:
        importlib.reload(sys.modules["db"])  # type: ignore
    import paths  # noqa: F401
    import db  # noqa: F401
    return sys.modules["db"], sys.modules["paths"]


def test_matrix_cell_upsert_and_get(tmp_path):
    db, _ = _prepare_tmp_env(tmp_path)
    sess_id = db.create_matrix_session("Test Session")
    tab_id = db.create_matrix_tab(sess_id, "Tab1", 0)

    # Prepare input and prompt
    cid = db.save_clipboard_item("hello", source="test")
    assert cid is not None
    mid = db.upsert_matrix_input(tab_id, 0, cid, attach_files=None)
    assert isinstance(mid, int)
    pid = db.upsert_matrix_prompt(tab_id, 0, {"name": "P1", "model": "m", "system_prompt": "", "parameters": {}})
    assert isinstance(pid, int)

    # Check ON
    db.upsert_matrix_cell(tab_id, mid, pid, True)
    checks = db.get_matrix_checked_positions(tab_id)
    assert (0, 0) in checks

    # Check OFF
    db.upsert_matrix_cell(tab_id, mid, pid, False)
    checks = db.get_matrix_checked_positions(tab_id)
    assert (0, 0) not in checks


def test_results_and_summaries(tmp_path):
    db, _ = _prepare_tmp_env(tmp_path)
    sess_id = db.create_matrix_session("S")
    tab_id = db.create_matrix_tab(sess_id, "T", 0)
    cid = db.save_clipboard_item("x", source="test")
    mid = db.upsert_matrix_input(tab_id, 0, cid, attach_files=["/tmp/a.txt"])  # set list to ensure stored
    pid = db.upsert_matrix_prompt(tab_id, 0, {"name": "P", "model": "m", "system_prompt": "", "parameters": {}})

    rid = db.add_matrix_result(tab_id, mid, pid, None, None, "final")
    assert isinstance(rid, int)
    pos = db.get_matrix_results_positions(tab_id)
    assert pos and pos[0][0] == 0 and pos[0][1] == 0 and pos[0][2] == "final"

    rsid = db.add_matrix_row_summary(tab_id, mid, "row sum")
    csid = db.add_matrix_col_summary(tab_id, pid, "col sum")
    fsid = db.add_matrix_final_summary(tab_id, "matrix sum")
    assert rsid and csid and fsid

    r_sums = db.get_matrix_row_summaries(tab_id)
    assert any(r == 0 and t for r, t in r_sums)
    c_sums = db.get_matrix_col_summaries(tab_id)
    assert any(c == 0 and t for c, t in c_sums)
    m_sum = db.get_matrix_final_summary(tab_id)
    assert m_sum == "matrix sum"


def test_session_updated_at_touched(tmp_path):
    db, _ = _prepare_tmp_env(tmp_path)
    sess_id = db.create_matrix_session("Touch")
    conn = db._connect()
    cur = conn.execute("SELECT created_at, updated_at FROM matrix_session WHERE id=?", (sess_id,))
    created_at, updated_at = cur.fetchone()
    # Make sure at least one second passes to guarantee different timestamp resolution
    time.sleep(1)
    db.create_matrix_tab(sess_id, "T1", 0)
    cur = conn.execute("SELECT updated_at FROM matrix_session WHERE id=?", (sess_id,))
    updated2 = cur.fetchone()[0]
    assert updated2 >= updated_at

