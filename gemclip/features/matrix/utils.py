"""Utilities for matrix batch processor.

Contains small, pure helpers extracted from the legacy window module to
improve testability and reduce file size.
"""
from __future__ import annotations

from typing import List, Dict, Any, Optional
from gemclip.core import Prompt, PromptParameters
from pathlib import Path


def normalize_inputs_list(inputs: Optional[List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
    """Normalize raw input item dicts to a safe, minimal structure.

    Each item becomes {"type": <text|file|image|image_compressed>, "data": <value>}.
    Unknown types are coerced to text with stringified data.
    """
    norm_inputs: List[Dict[str, Any]] = []
    if not inputs:
        return norm_inputs
    for it in inputs:
        try:
            t = (it or {}).get('type')
            d = (it or {}).get('data')
            if t in ('text', 'file', 'image', 'image_compressed'):
                norm_inputs.append({'type': t, 'data': d})
            else:
                norm_inputs.append({'type': 'text', 'data': str(d or '')})
        except Exception:
            norm_inputs.append({'type': 'text', 'data': ''})
    return norm_inputs


def truncate_result(result: str, max_length: int = 100) -> str:
    if len(result) > max_length:
        return result[:max_length] + "..."
    return result


def build_prompt_from_snapshot(snap: Dict[str, Any], idx: int = 0) -> Prompt:
    """Coerce a DB prompt snapshot dict into a Prompt model safely."""
    try:
        name = (snap or {}).get('name') or f"Prompt {idx+1}"
        model = (snap or {}).get('model') or 'gemini-2.5-flash-lite'
        sys = (snap or {}).get('system_prompt') or ''
        params_in = (snap or {}).get('parameters') or {}
        coerced: Dict[str, Any] = {}
        try:
            if 'temperature' in params_in and params_in['temperature'] is not None:
                coerced['temperature'] = float(params_in['temperature'])
        except Exception:
            pass
        for k in ('top_p', 'top_k', 'max_output_tokens'):
            try:
                if k in params_in and params_in[k] is not None:
                    coerced[k] = float(params_in[k]) if k != 'top_k' else int(params_in[k])
            except Exception:
                pass
        if 'stop_sequences' in params_in and isinstance(params_in['stop_sequences'], list):
            coerced['stop_sequences'] = [str(s) for s in params_in['stop_sequences']]
        pp = PromptParameters(**coerced) if coerced else PromptParameters()
        return Prompt(name=name, model=model, system_prompt=sys, parameters=pp)
    except Exception:
        return Prompt(name=f"Prompt {idx+1}", model='gemini-2.5-flash-lite', system_prompt='')


def ensure_prompt_set_dir(base: Optional[Path] = None) -> Path:
    """Return the prompt_set directory path, creating it if necessary.

    Base defaults to current working directory. The directory path is
    `<base>/prompt_set`.
    """
    try:
        base = base or Path.cwd()
    except Exception:
        base = Path('.')
    d = base / 'prompt_set'
    d.mkdir(parents=True, exist_ok=True)
    return d
