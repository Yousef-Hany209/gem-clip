"""Logic helpers for Matrix Batch Processor.

Pure helpers that prepare data fetched from DB and transform snapshots into
runtime structures suitable for the UI layer.
"""
from __future__ import annotations

from typing import List, Dict, Any, Tuple
import logging

from gemclip.infra import db
from gemclip.core import Prompt
from .utils import build_prompt_from_snapshot, truncate_result


def fetch_tab_payload(tab_id: int) -> Dict[str, Any]:
    """Fetch all artifacts for a matrix tab and return a payload dict.

    Keys: tab_id, session_id, tab_name, inputs, prompts, results, row_sums,
    col_sums, final_text, checks
    """
    inputs, prompts = db.get_matrix_tab_content(tab_id)
    meta = db.get_matrix_tab_meta(tab_id) or {}
    results = db.get_matrix_results_positions(tab_id)
    row_sums = db.get_matrix_row_summaries(tab_id)
    col_sums = db.get_matrix_col_summaries(tab_id)
    final_text = db.get_matrix_final_summary(tab_id)
    try:
        checks = db.get_matrix_checked_positions(tab_id)
    except Exception:
        checks = []
    try:
        logging.debug(
            f"MATRIX[fetch_tab_payload] tab_id={tab_id} inputs={len(inputs)} prompts={len(prompts)} results_positions={len(results)} row_sums={len(row_sums)} col_sums={len(col_sums)} final_text={'yes' if final_text else 'no'}"
        )
        # 先頭の結果1〜2件だけ短くプレビュー
        preview = [(r, c, (str(t)[:30] + '...') if t and len(str(t)) > 30 else str(t)) for r, c, t in (results[:2] if results else [])]
        logging.debug(f"MATRIX[fetch_tab_payload] results_preview={preview}")
    except Exception:
        pass
    return {
        'tab_id': tab_id,
        'session_id': meta.get('session_id'),
        'tab_name': meta.get('name'),
        'inputs': inputs,
        'prompts': prompts,
        'results': results,
        'row_sums': row_sums,
        'col_sums': col_sums,
        'final_text': final_text,
        'checks': checks,
    }


def build_prompts_dict(prompts_snap: List[Dict[str, Any]]) -> Dict[str, Prompt]:
    """Build Prompt objects from DB snapshots keyed by db_1, db_2, ..."""
    new_prompts: Dict[str, Prompt] = {}
    for idx, snap in enumerate(prompts_snap):
        p = build_prompt_from_snapshot(snap or {}, idx)
        new_prompts[f"db_{idx+1}"] = p
    return new_prompts


def apply_results_to_state(
    results_positions: List[Tuple[int, int, str]],
    num_rows: int,
    num_cols: int,
    max_truncate: int = 100,
) -> Tuple[List[List[str]], List[List[str]]]:
    """Return (full_results, truncated_results) grids from result positions.

    - full_results[r][c] stores full text
    - truncated_results[r][c] stores preview text
    """
    full: List[List[str]] = [["" for _ in range(num_cols)] for _ in range(num_rows)]
    trunc: List[List[str]] = [["" for _ in range(num_cols)] for _ in range(num_rows)]
    for r, c, text in results_positions or []:
        if 0 <= r < num_rows and 0 <= c < num_cols:
            full[r][c] = text
            trunc[r][c] = truncate_result(text, max_truncate)
    return full, trunc


def build_state_snapshot(
    checkbox: List[List[bool]],
    full_results: List[List[str]],
    row_summaries: List[str],
    col_summaries: List[str],
) -> Dict[str, Any]:
    """Create a serializable state dict for a tab.

    This mirrors the legacy structure used by the window class.
    """
    return {
        'checkbox': [[bool(v) for v in row] for row in (checkbox or [])],
        'full_results': [[str(c or '') for c in row] for row in (full_results or [])],
        'row_summaries': [str(s or '') for s in (row_summaries or [])],
        'col_summaries': [str(s or '') for s in (col_summaries or [])],
    }


def unpack_state(
    state: Dict[str, Any],
    num_rows: int,
    num_cols: int,
) -> Tuple[List[List[bool]], List[List[str]], List[str], List[str]]:
    """Extract typed arrays from a state dict with bounds-safety."""
    chk_src = state.get('checkbox') or []
    fr_src = state.get('full_results') or []
    rs_src = state.get('row_summaries') or []
    cs_src = state.get('col_summaries') or []

    checkbox: List[List[bool]] = []
    for r in range(num_rows):
        row = []
        src = chk_src[r] if r < len(chk_src) else []
        for c in range(num_cols):
            row.append(bool(src[c]) if c < len(src) else False)
        checkbox.append(row)

    full_results: List[List[str]] = []
    for r in range(num_rows):
        row: List[str] = []
        src = fr_src[r] if r < len(fr_src) else []
        for c in range(num_cols):
            row.append(str(src[c]) if c < len(src) else '')
        full_results.append(row)

    row_summaries = [str(rs_src[r]) if r < len(rs_src) else '' for r in range(num_rows)]
    col_summaries = [str(cs_src[c]) if c < len(cs_src) else '' for c in range(num_cols)]

    return checkbox, full_results, row_summaries, col_summaries
