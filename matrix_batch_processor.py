import mimetypes
import customtkinter as ctk
import tkinter as tk
import logging

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
from tkinter import messagebox, filedialog
from typing import List, Dict, Any, Optional, Callable
import asyncio
import threading
import time
import pyperclip
from gemclip.core import LlmAgent, Prompt
from PIL import Image
from io import BytesIO
import base64
from google.generativeai import types
import google.generativeai as genai
# from google.api_core import exceptions
import styles
from gemclip.features.matrix.utils import normalize_inputs_list, truncate_result, build_prompt_from_snapshot
from gemclip.features.matrix.logic import (
    fetch_tab_payload,
    build_prompts_dict,
    apply_results_to_state,
    build_state_snapshot,
    unpack_state,
)
from gemclip.features.matrix.apply import (
    set_ids_from_payload,
    apply_inputs_prompts,
    reset_state_arrays,
    apply_results_summaries,
    finalize_canvas,
    update_session_label,
)
from gemclip.infra import db
from i18n import tr
from pathlib import Path
from gemclip.core import DELETE_ICON_FILE
import traceback
from gemclip.core import create_image_part # create_image_partをインポート
from gemclip.features.matrix.view import SizerGrip
from gemclip.features.matrix.tabbar import render_tabbar, compute_slot_width, adjust_tabbar_widths
from gemclip.features.matrix.layout import configure_scrollable_grid, save_active_tab_vars
from gemclip.features.matrix.tabops import rebuild_tabs, on_tab_clicked, on_tab_press, on_tab_motion, on_tab_release
from google.generativeai.generative_models import GenerativeModel # GenerativeModelをインポート
from history_dialogs import HistoryEditDialog
from gemclip.ui.textbox_utils import setup_textbox_right_click_menu
from CTkMessagebox import CTkMessagebox

# UI feature flags (keep interface simple by default)
SHOW_FILE_SESSION_UI = False  # Hide legacy file session save/load UI; prefer DB sessions + import/export


def apply_loaded_tab_preserve_tabs(win, data: dict):
    """Apply a loaded tab payload without destroying other tabs.

    - Updates active tab's name and prompts
    - Rebuilds UI arrays and applies results/summaries
    - Snapshots state back into the active tab
    """
    logging.debug("DEBUG: apply_loaded_tab_preserve_tabs - start")
    try:
        import time as _t
        win._just_loaded_at = _t.time()
    except Exception:
        pass

    # IDs, inputs, prompts
    set_ids_from_payload(win, data)
    apply_inputs_prompts(win, data)

    # Update active tab meta without nuking others
    try:
        tab_name = str(data.get('tab_name') or tr('matrix.tab.default'))
        prompts_obj = {pid: (p.model_copy(deep=True) if hasattr(p, 'model_copy') else Prompt(**p.model_dump())) for pid, p in win.prompts.items()}
        if not isinstance(getattr(win, '_tabs', None), list) or not win._tabs:
            win._tabs = [{'name': tab_name, 'prompts_obj': prompts_obj, 'state': None}]
            win._active_tab_index = 0
        else:
            idx = int(getattr(win, '_active_tab_index', 0) or 0)
            if idx < 0 or idx >= len(win._tabs):
                idx = 0
                win._active_tab_index = 0
            win._tabs[idx]['name'] = tab_name
            win._tabs[idx]['prompts_obj'] = prompts_obj
            win._tabs[idx]['state'] = None
        try:
            win._tabs[win._active_tab_index]['frame'] = win.scrollable_content_frame
        except Exception:
            pass
    except Exception:
        pass

    # Reset UI arrays and rebuild
    reset_state_arrays(win)
    win._update_ui()
    finalize_canvas(win)
    try:
        win._render_tabbar()
    except Exception:
        pass

    # Apply results and summaries into UI
    try:
        apply_results_summaries(win, data)
        try:
            if 0 <= win._active_tab_index < len(win._tabs):
                win._tabs[win._active_tab_index]['state'] = win._snapshot_state()
        except Exception:
            pass
    except Exception:
        logging.exception("ERROR: apply_loaded_tab_preserve_tabs - apply results failed")

class MatrixBatchProcessorWindow(ctk.CTkToplevel):
    def __init__(self, prompts: Dict[str, Prompt], on_processing_completed: Callable, llm_agent_factory: Callable[[str, Prompt], LlmAgent], notification_callback: Callable[[str, str, str], None], worker_loop: asyncio.AbstractEventLoop, parent_app: ctk.CTk, agent: Any):
        super().__init__(parent_app)
        # Ensure tooltip is hidden on destroy and on any click within app
        self.bind("<Destroy>", lambda e: self._hide_tooltip())
        try:
            self.bind_all("<Button-1>", lambda e: self._hide_tooltip())
        except Exception:
            pass
        self.protocol("WM_DELETE_WINDOW", self.on_closing)
        self.prompts = prompts
        logging.debug(f"DEBUG: __init__ - Initial self.prompts keys: {list(self.prompts.keys())}")
        try:
            self._initial_prompts = {pid: (p.model_copy(deep=True) if hasattr(p, 'model_copy') else Prompt(**p.model_dump())) for pid, p in self.prompts.items()}
            logging.debug(f"DEBUG: __init__ - _initial_prompts keys: {list(self._initial_prompts.keys())}")
        except Exception:
            self._initial_prompts = dict(self.prompts)
            logging.debug(f"DEBUG: __init__ - _initial_prompts (exception) keys: {list(self._initial_prompts.keys())}")
        self.on_processing_completed = on_processing_completed
        self.llm_agent_factory = llm_agent_factory
        self.notification_callback = notification_callback
        self.worker_loop = worker_loop
        self.parent_app = parent_app
        self.agent = agent
        self.geometry(styles.MATRIX_WINDOW_GEOMETRY)
        self.title(tr("matrix.window_title")) # ウィンドウタイトルを設定
        self.resizable(True, True)
        # NOTE: Do not mark as transient to keep normal window controls and allow other windows to lift above it
        self._is_closing = False

        self._cursor_update_job = None
        self._start_cursor_monitoring()

        self.processing_tasks: List[asyncio.Task] = []
        self.semaphore = asyncio.Semaphore(5)
        self.total_tasks = 0
        self.completed_tasks = 0
        self.progress_lock = threading.Lock()
        # 入力セルのフレーム参照を保持して、部分更新で再描画を最小化
        self._input_row_frames: List[ctk.CTkFrame] = []

        # 進捗エリアの背景はMATRIX_TOP_BG_COLOR（キャンバスより少し暗め）
        self.progress_frame = ctk.CTkFrame(self, fg_color=styles.MATRIX_TOP_BG_COLOR)
        self.progress_frame.pack(fill="x", padx=10, pady=(0, 5))
        self.progress_label = ctk.CTkLabel(self.progress_frame, text=tr("matrix.progress_fmt", done=0, total=0), font=styles.MATRIX_FONT_BOLD)
        self.progress_label.pack(fill="x")


        self.input_data: List[Dict[str, Any]] = [{"type": "text", "data": ""}]
        self.checkbox_states: List[List[ctk.BooleanVar]] = []
        self.results: List[List[ctk.StringVar]] = []
        self._full_results: List[List[str]] = []
        self._row_summaries: List[ctk.StringVar] = []
        self._col_summaries: List[ctk.StringVar] = []
        self._history_popup: Optional['ClipboardHistorySelectorPopup'] = None

        self.summarize_row_button: Optional[ctk.CTkButton] = None
        self.summarize_col_button: Optional[ctk.CTkButton] = None

        # --- Drag-and-drop (column reorder) state ---
        self._col_header_frames: List[ctk.CTkFrame] = []
        self._col_drag_data: Dict[str, Any] = {}
        self._col_drag_active_frame: Optional[ctk.CTkFrame] = None
        self._col_drop_line_id: Optional[int] = None  # legacy (canvas)
        self._col_drop_indicator_widget: Optional[tk.Frame] = None

        # --- Flow run state ---
        try:
            self.max_flow_steps: int = int(getattr(self.agent.config, 'max_flow_steps', 5))
        except Exception:
            self.max_flow_steps: int = 5
        self._result_textboxes: List[List[Optional[ctk.CTkTextbox]]] = []
        self._cell_style: List[List[str]] = []  # "normal" or "flow"
        self._flow_cancel_requested: bool = False
        self._flow_tasks: List[asyncio.Task] = []
        # DB session/tab identifiers and caches
        self._db_session_id: Optional[int] = None
        self._db_tab_id: Optional[int] = None
        self._db_input_ids: Dict[int, int] = {}
        self._db_prompt_ids: Dict[int, int] = {}

        # --- Session UI state ---
        self._current_session_name: Optional[str] = None
        self._session_label_var: ctk.StringVar = ctk.StringVar(value=f"{tr('matrix.toolbar.session_label')} {tr('common.unspecified')}")

        # --- UIリサイズ用プロパティ ---
        # 各列の幅と各行の高さを保持するリスト。0番目は固定列/ヘッダ行に対応。
        self._column_widths: List[int] = []
        self._row_heights: List[int] = []
        # リサイズ中の状態管理
        self._current_column_resizing: Optional[int] = None
        self._col_resize_start_x: int = 0
        self._col_resize_initial_width: int = 0
        self._current_row_resizing: Optional[int] = None
        self._row_resize_start_y: int = 0
        self._row_resize_initial_height: int = 0

        self._create_toolbar()
        self._init_tabs()
        self._create_main_grid_frame()
        self.after(100, self._update_ui) # 遅延させてUIを更新
        self.state('zoomed') # ウィンドウを最大化

    # --- Helpers: normalize inputs structure ---
    def _normalize_inputs_list(self, inputs: Optional[List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
        return normalize_inputs_list(inputs)

    def _load_tab_async_safely(self, tab_id: int):
        logging.debug(f"DEBUG: _load_tab_async_safely - tab_id={tab_id} のロード準備中")
        try:
            # 古いグラブが残っていないことを確認
            try:
                cur = self.tk.call('grab', 'current')
                if cur:
                    logging.debug(f"DEBUG: _load_tab_async_safely - 古いグラブを解放中: {cur}")
                    try:
                        # Release grab by widget path directly to avoid stale object conversion
                        self.tk.call('grab', 'release', cur)
                    except Exception as e:
                        logging.warning(f"WARNING: _load_tab_async_safely - grab release 失敗: {e}")
                        try:
                            self.grab_release()
                        except Exception:
                            pass
            except Exception as e:
                logging.warning(f"WARNING: _load_tab_async_safely - グラブ確認失敗: {e}")
                pass # グラブが設定されていない場合も考慮

            # このウィンドウが無効化されていた場合、再有効化
            try:
                self.attributes('-disabled', False)
                logging.debug("DEBUG: _load_tab_async_safely - ウィンドウを再有効化しました。")
            except Exception as e:
                logging.warning(f"WARNING: _load_tab_async_safely - ウィンドウの再有効化失敗: {e}")
                pass

            # UIスレッドで同期的にロード・適用（安定優先）
            logging.debug(f"DEBUG: _load_tab_async_safely - DBからデータ取得開始 (tab_id={tab_id})")
            try:
                payload = fetch_tab_payload(tab_id)
                logging.debug(
                    f"DEBUG: _load_tab_async_safely - DBデータ取得完了。入力数: {len(payload.get('inputs') or [])}, プロンプト数: {len(payload.get('prompts') or [])}"
                )
            except Exception as e:
                logging.exception(f"ERROR: _load_tab_async_safely - DBデータ取得中にエラーが発生しました (tab_id={tab_id})")
                try:
                    self.progress_label.configure(text=tr("matrix.session.load_failed", details=str(e)))
                except Exception:
                    pass
                return

            # 適用
            try:
                apply_loaded_tab_preserve_tabs(self, payload) if 'apply_loaded_tab_preserve_tabs' in globals() else self._apply_loaded_tab(payload)
                logging.debug("DEBUG: _load_tab_async_safely - ロード済みタブの適用が完了しました。")
                try:
                    self.lift(); self.focus_force()
                except Exception:
                    pass
                # Update session label for clarity
                update_session_label(self, payload)
            except Exception as e:
                logging.exception(f"ERROR: _load_tab_async_safely - ロード済みタブの適用中にエラーが発生しました")
                try:
                    self.progress_label.configure(text=tr("matrix.session.load_failed", details=str(e)))
                except Exception:
                    pass
        except Exception as e:
            logging.exception(f"ERROR: _load_tab_async_safely - 非同期ロード処理の開始中にエラーが発生しました")
            # Avoid modal popups; update inline
            try:
                self.progress_label.configure(text=tr("matrix.session.load_failed", details=str(e)))
            except Exception:
                pass

    def _apply_loaded_tab(self, data: dict):
        logging.debug(f"DEBUG: _apply_loaded_tab - ロード済みデータ適用開始")
        try:
            import time as _t
            self._just_loaded_at = _t.time()
        except Exception:
            pass

        # IDs, inputs, prompts
        set_ids_from_payload(self, data)
        logging.debug(
            f"DEBUG: _apply_loaded_tab - 適用する入力数: {len(data.get('inputs') or [])}, プロンプトスナップショット数: {len(data.get('prompts') or [])}"
        )
        apply_inputs_prompts(self, data)
        logging.debug(f"DEBUG: _apply_loaded_tab - 入力データ正規化後、数: {len(self.input_data)}")
        logging.debug(f"DEBUG: _apply_loaded_tab - プロンプト構築後、数: {len(self.prompts)}")

        # タブバー表示名をDBのタブ名に合わせる（単一タブとして反映）
        try:
            tab_name = str(data.get('tab_name') or tr('matrix.tab.default'))
            self._tabs = [{'name': tab_name, 'prompts_obj': {pid: (p.model_copy(deep=True) if hasattr(p, 'model_copy') else Prompt(**p.model_dump())) for pid, p in self.prompts.items()}, 'state': None}]
            self._active_tab_index = 0
            # 既存のフレーム参照を置き換える
            try:
                self._tabs[0]['frame'] = self.scrollable_content_frame
            except Exception:
                pass
        except Exception:
            pass

        # 状態配列のリセットと再描画
        reset_state_arrays(self)
        logging.debug("DEBUG: _apply_loaded_tab - UI状態配列をリセットしました。")
        self._update_ui()
        logging.debug("DEBUG: _apply_loaded_tab - UIを更新しました。")
        finalize_canvas(self)
        try:
            self._render_tabbar()
        except Exception:
            pass

        # 結果とサマリーを適用
        try:
            logging.debug(
                f"DEBUG: _apply_loaded_tab - 適用する結果数: {len(data.get('results') or [])}, 行数: {len(self.input_data)}, 列数: {len(self.prompts)}"
            )
            apply_results_summaries(self, data)

        except Exception as e:
            logging.exception(f"ERROR: _apply_loaded_tab - 結果とサマリーの適用中にエラーが発生しました")
            try:
                CTkMessagebox(title=tr("common.error"), message=tr("matrix.session.load_failed", details=f"apply results: {e}"), icon="cancel")
            except Exception:
                pass

    # --- DB helpers ---
    def _ensure_db_session_tab(self):
        from gemclip.features.matrix.dbops import ensure_db_session_tab
        return ensure_db_session_tab(self)

    def _save_session_to_db(self, name: Optional[str] = None):
        from gemclip.features.matrix.dbops import save_session_to_db
        return save_session_to_db(self, name)

    def _ensure_db_input_id(self, r_idx: int) -> Optional[int]:
        from gemclip.features.matrix.dbops import ensure_db_input_id
        return ensure_db_input_id(self, r_idx)

    def _ensure_db_prompt_id(self, c_idx: int, prompt_config: Prompt) -> Optional[int]:
        from gemclip.features.matrix.dbops import ensure_db_prompt_id
        return ensure_db_prompt_id(self, c_idx, prompt_config)

    def _load_session_tab_from_db(self, tab_id: int):
        try:
            inputs, prompts = db.get_matrix_tab_content(tab_id)
        except Exception as e:
            CTkMessagebox(title=tr("common.error"), message=tr("matrix.session.load_failed", details=f"get_matrix_tab_content: {e}"), icon="cancel").wait_window()
            return
        # Rebuild inputs with normalization
        try:
            self.input_data = self._normalize_inputs_list(inputs or [])
        except Exception as e:
            self.input_data = [{'type':'text','data': ''}]
        # Rebuild prompts mapping safely
        from gemclip.core import PromptParameters
        def _safe_prompt_from_snapshot(idx: int, snap: Dict[str, Any]) -> Prompt:
            try:
                name = (snap or {}).get('name') or f"Prompt {idx+1}"
                model = (snap or {}).get('model') or 'gemini-2.5-flash-lite'
                sys = (snap or {}).get('system_prompt') or ''
                params_in = (snap or {}).get('parameters') or {}
                # Coerce numeric fields when present, ignore invalids
                coerced = {}
                try:
                    if 'temperature' in params_in and params_in['temperature'] is not None:
                        coerced['temperature'] = float(params_in['temperature'])
                except Exception:
                    pass
                for k in ('top_p','top_k','max_output_tokens'):
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
        new_prompts: Dict[str, Prompt] = {}
        try:
            for idx, snap in enumerate(prompts or []):
                p = _safe_prompt_from_snapshot(idx, snap or {})
                new_prompts[f"db_{idx+1}"] = p
            self.prompts = new_prompts
        except Exception as e:
            CTkMessagebox(title=tr("common.error"), message=tr("matrix.session.load_failed", details=f"prompt build: {e}"), icon="cancel").wait_window()
            return
        # Reset state arrays and repaint
        self.checkbox_states = []
        self.results = []
        self._full_results = []
        self._row_summaries = []
        self._col_summaries = []
        self._result_textboxes = []
        self._cell_style = []
        self._update_ui()
        # Apply past results and summaries if available
        try:
            results = db.get_matrix_results_positions(tab_id)
            # Ensure arrays sized
            num_rows = len(self.input_data)
            num_cols = len(self.prompts)
            while len(self.results) < num_rows:
                self.results.append([])
            for r in range(num_rows):
                while len(self.results[r]) < num_cols:
                    self.results[r].append(ctk.StringVar(value=""))
            while len(self._full_results) < num_rows:
                self._full_results.append([])
            for r in range(num_rows):
                while len(self._full_results[r]) < num_cols:
                    self._full_results[r].append("")
            try:
                full, trunc = apply_results_to_state(results, num_rows, num_cols)
                self._full_results = full
                for r in range(num_rows):
                    for c in range(num_cols):
                        self.results[r][c].set(trunc[r][c])
            except Exception:
                pass
            # Row summaries
            row_sums = db.get_matrix_row_summaries(tab_id)
            if row_sums:
                # init
                self._row_summaries = [ctk.StringVar(value="") for _ in range(num_rows)]
                for r_idx, text in row_sums:
                    if 0 <= r_idx < num_rows:
                        self._row_summaries[r_idx].set(text)
                self._update_row_summary_column()
            # Column summaries
            col_sums = db.get_matrix_col_summaries(tab_id)
            if col_sums:
                self._col_summaries = [ctk.StringVar(value="") for _ in range(num_cols)]
                for c_idx, text in col_sums:
                    if 0 <= c_idx < num_cols:
                        self._col_summaries[c_idx].set(text)
                self._update_column_summary_row()
            # Final summary
            final_text = db.get_matrix_final_summary(tab_id)
            if final_text:
                self._update_matrix_summary_cell(final_text)
            # Checked cells
            try:
                checks = db.get_matrix_checked_positions(tab_id)
                # Ensure checkbox arrays sized
                while len(self.checkbox_states) < len(self.input_data):
                    self.checkbox_states.append([])
                num_cols = len(self.prompts)
                for r in range(len(self.input_data)):
                    while len(self.checkbox_states[r]) < num_cols:
                        self.checkbox_states[r].append(ctk.BooleanVar(value=False))
                for r_idx, c_idx in checks:
                    if 0 <= r_idx < len(self.checkbox_states) and 0 <= c_idx < len(self.checkbox_states[r_idx]):
                        self.checkbox_states[r_idx][c_idx].set(True)
            except Exception:
                pass
        except Exception as e:
            CTkMessagebox(title=tr("common.error"), message=tr("matrix.session.load_failed", details=f"apply results: {e}"), icon="cancel").wait_window()



    def _ensure_db_prompt_id(self, c_idx: int, prompt_config: Prompt) -> Optional[int]:
        try:
            if c_idx in self._db_prompt_ids:
                return self._db_prompt_ids[c_idx]
            self._ensure_db_session_tab()
            if self._db_tab_id is None:
                return None
            snap = {
                'name': prompt_config.name,
                'model': prompt_config.model,
                'system_prompt': prompt_config.system_prompt,
                'parameters': getattr(prompt_config, 'parameters', None).model_dump() if getattr(prompt_config, 'parameters', None) else {},
                'enable_web': getattr(prompt_config, 'enable_web', False),
            }
            pid = db.upsert_matrix_prompt(self._db_tab_id, c_idx, snap)
            self._db_prompt_ids[c_idx] = pid
            return pid
        except Exception:
            return None

    def on_closing(self):
        """ウィンドウが閉じられる際の処理（1ダイアログで完結）。"""
        try:
            self._hide_tooltip()
        except Exception:
            pass
        # 事前に状態をスナップショット
        try:
            if hasattr(self, '_tabs') and self._tabs and 0 <= self._active_tab_index < len(self._tabs):
                self._tabs[self._active_tab_index]['prompts_obj'] = {pid: (p.model_copy(deep=True) if hasattr(p, 'model_copy') else Prompt(**p.model_dump())) for pid, p in self.prompts.items()}
                self._tabs[self._active_tab_index]['state'] = self._snapshot_state()
        except Exception:
            pass

        choice = self._confirm_close_or_save()
        if choice is None or choice == 'cancel':
            return
        if choice == 'save':
            try:
                self._quick_save_session()
            except Exception:
                pass
        # 共通クリーンアップ
        self._is_closing = True
        try:
            if hasattr(self, '_cursor_update_job') and self._cursor_update_job:
                self.after_cancel(self._cursor_update_job)
                self._cursor_update_job = None
        except Exception:
            pass
        try:
            if hasattr(self, 'processing_tasks'):
                for task in self.processing_tasks:
                    if not task.done():
                        task.cancel()
        except Exception:
            pass
        self.destroy()

    def _confirm_close_or_save(self) -> str | None:
        """単一ダイアログで 保存して終了 / 保存せず終了 / キャンセル を選ばせる。

        Returns: 'save' | 'discard' | 'cancel' | None
        """
        dlg = ctk.CTkToplevel(self, fg_color=styles.HISTORY_ITEM_FG_COLOR)
        try:
            dlg.transient(self)
            dlg.lift(self)
            dlg.attributes('-topmost', True)
            dlg.after(200, lambda: dlg.attributes('-topmost', False))
        except Exception:
            pass
        dlg.title(tr("confirm.close_or_save_title"))
        dlg.geometry("420x140")
        frame = ctk.CTkFrame(dlg, fg_color="transparent")
        frame.pack(fill='both', expand=True, padx=12, pady=12)
        ctk.CTkLabel(frame, text=tr("confirm.close_or_save_message"), text_color=styles.HISTORY_ITEM_TEXT_COLOR, anchor='w', justify='left', wraplength=380).pack(fill='x', pady=(0,8))
        btns = ctk.CTkFrame(frame, fg_color="transparent")
        btns.pack()
        result: list[str] = []
        ctk.CTkButton(btns, text=tr("common.save_and_close"), command=lambda: (result.append('save'), dlg.destroy()), fg_color=styles.DEFAULT_BUTTON_FG_COLOR, text_color=styles.DEFAULT_BUTTON_TEXT_COLOR).pack(side='left', padx=6)
        ctk.CTkButton(btns, text=tr("common.close_without_saving"), command=lambda: (result.append('discard'), dlg.destroy()), fg_color=styles.CANCEL_BUTTON_COLOR, text_color=styles.CANCEL_BUTTON_TEXT_COLOR).pack(side='left', padx=6)
        ctk.CTkButton(btns, text=tr("common.cancel"), command=lambda: (result.append('cancel'), dlg.destroy())).pack(side='left', padx=6)
        self.wait_window(dlg)
        return result[0] if result else None

    def _quick_save_session(self) -> None:
        """ダイアログ無しで即保存。未設定なら日時名で保存。"""
        try:
            import time as _t
            name = (self._current_session_name or '').strip() or _t.strftime('%Y%m%d_%H%M%S')
            self._save_session_to_db(name)
            self._current_session_name = name
            try:
                self._session_label_var.set(f"{tr('matrix.toolbar.session_label')} {name}")
            except Exception:
                pass
        except Exception:
            pass

    def on_prompts_updated(self, updated_prompts: Dict[str, Prompt]):
        """外部（プロンプト管理）での変更を即時反映する。
        - 設定画面の「マトリクス」チェックはデフォルトタブに表示するプロンプト。
        - デフォルト以外がアクティブでも、デフォルトタブの内容のみ更新する。
        - アクティブがデフォルトの場合は表示も即時更新。
        """
        try:
            filtered = {pid: p for pid, p in updated_prompts.items() if getattr(p, 'include_in_matrix', False)}
            # デフォルトタブを探す
            default_idx = next((i for i, t in enumerate(self._tabs) if str(t.get('name')) == tr('matrix.tab.default')), None)
            if default_idx is None:
                # なければ作る
                self._tabs.insert(0, {'name': tr('matrix.tab.default'), 'prompts_obj': {}, 'state': None})
                default_idx = 0
            # デフォルトタブのプロンプトを更新
            self._tabs[default_idx]['prompts_obj'] = {pid: (p.model_copy(deep=True) if hasattr(p, 'model_copy') else Prompt(**p.model_dump())) for pid, p in filtered.items()}
            logging.debug(f"DEBUG: on_prompts_updated - default tab prompts updated: {list(filtered.keys())}")
            # アクティブがデフォルトなら表示も更新
            if self._active_tab_index == default_idx:
                self.prompts = {pid: (p.model_copy(deep=True) if hasattr(p, 'model_copy') else Prompt(**p.model_dump())) for pid, p in filtered.items()}
                self.checkbox_states = []
                self.results = []
                self._full_results = []
                self._update_ui()
        except Exception:
            try:
                self._update_ui()
            except Exception:
                pass

    def _create_toolbar(self):
        from gemclip.features.matrix.toolbar import build_toolbar
        return build_toolbar(self)

    def _handle_save_action(self):
        """統合された保存アクション: セッション名があれば上書き、なければ名前を付けて保存ダイアログを表示。"""
        # 常に保存ダイアログを開く。デフォルト名は以下の通り:
        # - セッション未設定: 日時(yyyymmdd_hhmmss)
        # - セッション設定済: 現在のセッション名
        default_name = (self._current_session_name or '').strip()
        if not default_name:
            try:
                default_name = time.strftime('%Y%m%d_%H%M%S')
            except Exception:
                default_name = ''

        dlg = ctk.CTkToplevel(self)
        dlg.title(tr("matrix.toolbar.save_as"))
        dlg.transient(self)
        try:
            dlg.lift(self)
            dlg.attributes('-topmost', True)
            dlg.focus_force()
            dlg.after(200, lambda: dlg.attributes('-topmost', False))
        except Exception:
            pass
        dlg# removed: .grab_set()
        frm = ctk.CTkFrame(dlg)
        frm.pack(fill="both", expand=True, padx=12, pady=12)
        ctk.CTkLabel(frm, text=tr("matrix.session.placeholder"), anchor="w").pack(fill="x")
        entry = ctk.CTkEntry(frm)
        entry.pack(fill="x", pady=(6, 12))
        try:
            if default_name:
                entry.insert(0, default_name)
            entry.focus_set()
            entry.select_range(0, 'end')
        except Exception:
            pass
        btns = ctk.CTkFrame(frm)
        btns.pack(fill="x")
        def do_save_as_new_name():
            name = (entry.get() or '').strip()
            if not name:
                try:
                    CTkMessagebox(title=tr("common.warning"), message=tr("matrix.session.name_required"), icon="warning").wait_window()
                except Exception:
                    pass
                return
            try:
                self._save_session_to_db(name)
                self._current_session_name = name # 新しいセッション名を保持
                self._session_label_var.set(f"{tr('matrix.toolbar.session_label')} {name}")
                CTkMessagebox(title=tr("common.success"), message=tr("matrix.session.saved"), icon="info").wait_window()
            except Exception as e:
                try:
                    CTkMessagebox(title=tr("common.error"), message=tr("matrix.session.save_failed", details=str(e)), icon="cancel").wait_window()
                except Exception:
                    pass
            finally:
                try:
                    dlg.destroy()
                except Exception:
                    pass
        ctk.CTkButton(btns, text=tr("common.save"), command=do_save_as_new_name, fg_color=styles.DEFAULT_BUTTON_FG_COLOR, text_color=styles.DEFAULT_BUTTON_TEXT_COLOR).pack(side="left", padx=4)
        ctk.CTkButton(btns, text=tr("common.cancel"), command=lambda: dlg.destroy()).pack(side="left", padx=4)
        try:
            entry.bind('<Control-Return>', lambda e: do_save_as_new_name())
        except Exception:
            pass

    # --- Prompt set tabs (max 5) ---
    def _serialize_prompts(self, prompts_dict: Dict[str, Prompt]) -> Dict[str, dict]:
        out: Dict[str, dict] = {}
        for pid, p in prompts_dict.items():
            try:
                out[pid] = p.model_dump(by_alias=True, exclude_none=True)
            except Exception:
                out[pid] = {
                    'name': getattr(p, 'name', ''),
                    'model': getattr(p, 'model', ''),
                    'system_prompt': getattr(p, 'system_prompt', ''),
                    'thinking_level': getattr(p, 'thinking_level', 'Balanced'),
                    'enable_web': getattr(p, 'enable_web', False),
                    'parameters': getattr(getattr(p, 'parameters', None), 'model_dump', lambda **_: {})()
                }
        return out

    def _deserialize_prompts(self, data: Dict[str, dict]) -> Dict[str, Prompt]:
        out: Dict[str, Prompt] = {}
        for pid, pd in data.items():
            try:
                out[pid] = Prompt(**pd)
            except Exception:
                continue
        return out

    def _snapshot_state(self) -> dict:
        chk: list[list[bool]] = []
        for r in self.checkbox_states:
            try:
                chk.append([bool(v.get()) if hasattr(v, 'get') else bool(v) for v in r])
            except Exception:
                chk.append([False for _ in r])
        results_full: list[list[str]] = []
        for r in self._full_results:
            results_full.append([str(c or '') for c in r])
        row_summ = [sv.get() if hasattr(sv, 'get') else str(sv or '') for sv in self._row_summaries]
        col_summ = [sv.get() if hasattr(sv, 'get') else str(sv or '') for sv in self._col_summaries]
        return build_state_snapshot(chk, results_full, row_summ, col_summ)

    def _apply_state(self, state: Optional[dict]):
        if not state:
            return
        num_rows = len(self.input_data)
        num_cols = len(self.prompts)
        checkbox, full_results, row_summ, col_summ = unpack_state(state, num_rows, num_cols)
        # Checkbox
        self.checkbox_states = [
            [ctk.BooleanVar(value=checkbox[r][c]) for c in range(num_cols)]
            for r in range(num_rows)
        ]
        # Results
        self._full_results = full_results
        self.results = [
            [ctk.StringVar(value=truncate_result(full_results[r][c])) for c in range(num_cols)]
            for r in range(num_rows)
        ]
        # Summaries
        self._row_summaries = [ctk.StringVar(value=row_summ[r]) for r in range(num_rows)]
        self._col_summaries = [ctk.StringVar(value=col_summ[c]) for c in range(num_cols)]
    def _prompt_set_dir(self) -> Path:
        # Delegate to shared util for consistency
        from gemclip.features.matrix.utils import ensure_prompt_set_dir
        return ensure_prompt_set_dir()




    def _init_tabs(self):
        # Custom browser-like tab bar container
        # タブバーの背景は進捗エリアと統一（MATRIX_TOP_BG_COLOR）
        self.tabbar_frame = ctk.CTkFrame(self, fg_color=styles.MATRIX_TOP_BG_COLOR)
        self.tabbar_frame.pack(fill='x', padx=10, pady=(0,0))
        # Per-tab storage
        self._tabs: list[dict] = []
        self._active_tab_index: int = 0
        self._tab_slot_width: Optional[int] = None
        # Load from session or default and render
        self._start_with_default_only: bool = True
        self._load_session_or_default()
        self._render_tabbar()
        # Resize: adjust tab widths only (no rebuild) to avoid flicker
        self.tabbar_frame.bind("<Configure>", lambda e: self._adjust_tabbar_widths())

    def _rebuild_tabs(self):
        return rebuild_tabs(self)

    def _on_tab_clicked(self, idx: int):
        # Bounds check
        if not isinstance(self._tabs, list) or not self._tabs:
            return
        if idx < 0 or idx >= len(self._tabs):
            return
        if hasattr(self, '_active_tab_index') and idx == self._active_tab_index:
            return
        try:
            if 0 <= self._active_tab_index < len(self._tabs):
                self._tabs[self._active_tab_index]['prompts_obj'] = {pid: (p.model_copy(deep=True) if hasattr(p, 'model_copy') else Prompt(**p.model_dump())) for pid, p in self.prompts.items()}
                self._tabs[self._active_tab_index]['state'] = self._snapshot_state()
        except Exception:
            pass
        self._active_tab_index = idx
        try:
            if not (0 <= self._active_tab_index < len(self._tabs)):
                return
            t = self._tabs[self._active_tab_index]
            prompts_obj = t.get('prompts_obj') if isinstance(t.get('prompts_obj', {}), dict) else self._deserialize_prompts(t.get('prompts', {}))
            self.prompts = {pid: (p.model_copy(deep=True) if hasattr(p, 'model_copy') else Prompt(**p.model_dump())) for pid, p in prompts_obj.items()} if prompts_obj else {}
            logging.debug(f"DEBUG: _on_tab_clicked - self.prompts keys set to: {list(self.prompts.keys())}")
        except Exception as e:
            self.prompts = {}
            logging.error(f"ERROR: _on_tab_clicked - Error setting self.prompts: {e}")
        # 状態を適用し新規フレームに描画（確実な切替を優先）
        self._result_textboxes = []
        self._cell_style = []
        self.checkbox_states = []
        self.results = []
        self._full_results = []
        self._row_summaries = []
        self._col_summaries = []
        try:
            self._apply_state(self._tabs[self._active_tab_index].get('state'))
        except Exception:
            pass
        new_frame = ctk.CTkFrame(self.canvas, fg_color="transparent")
        self.scrollable_content_frame = new_frame
        try:
            self.canvas.itemconfigure(self._window_id, window=self.scrollable_content_frame)
        except Exception:
            pass
        self._update_ui()
        self._render_tabbar()

    def _render_tabbar(self):
        return render_tabbar(self)

    def _compute_slot_width(self) -> int:
        return compute_slot_width(self)

    def _adjust_tabbar_widths(self):
        return adjust_tabbar_widths(self)

    # --- Tab drag & drop handlers ---
    def _compute_tab_drop_index(self, x_root: int) -> int:
        try:
            children = [c for c in self.tabbar_frame.winfo_children()][:len(self._tabs)]
            if not children:
                return 0
            centers = [c.winfo_rootx() + (c.winfo_width() // 2) for c in children]
            for i, cx in enumerate(centers):
                if x_root < cx:
                    return i
            return len(children) - 1
        except Exception:
            return 0

    def _on_tab_press(self, event, idx: int):
        return on_tab_press(self, event, idx)

    def _on_tab_motion(self, event):
        return on_tab_motion(self, event)

    def _on_tab_release(self, event):
        return on_tab_release(self, event)

    def _delete_tab_index(self, idx: int):
        if not (0 <= idx < len(self._tabs)):
            return
        try:
            tab_name = str(self._tabs[idx].get('name') or '')
        except Exception:
            tab_name = ''
        if not messagebox.askyesno(tr("common.delete_confirm_title"), tr("matrix.tab.delete_confirm", name=tab_name)):
            return
        del self._tabs[idx]
        if not self._tabs:
            self._tabs = [{'name': tr('matrix.tab.default'), 'prompts_obj': {}, 'state': None}]
            self._active_tab_index = 0
        else:
            if self._active_tab_index >= len(self._tabs):
                self._active_tab_index = len(self._tabs) - 1
        # 再描画と内容の即時切替
        self._rebuild_tabs()
        # 高速切替: 既存フレームがあれば表示差し替え、なければ構築
        tab_frame = self._tabs[self._active_tab_index].get('frame')
        if tab_frame:
            self.scrollable_content_frame = tab_frame
            try:
                self.canvas.itemconfigure(self._window_id, window=self.scrollable_content_frame)
                self.after(1, lambda: self.canvas.configure(scrollregion=self.canvas.bbox("all")))
            except Exception:
                pass
        else:
            self._update_ui()
        try:
            self._save_session()
        except Exception:
            pass

    def _rename_tab(self, idx: int):
        try:
            current = str(self._tabs[idx].get('name') or '')
        except Exception:
            current = ''
        new_name = self._prompt_text_input(tr("matrix.tab.new_name_title"), tr("matrix.tab.new_name_label"), default=current)
        if not new_name:
            return
        self._tabs[idx]['name'] = new_name
        self._render_tabbar()
        try:
            self._save_session()
        except Exception:
            pass

    def _load_session_or_default(self):
        """
        セッションをロードするか、デフォルトのタブを初期化します。
        ファイルベースのセッション管理は廃止されたため、常にDBからロードを試みるか、デフォルトタブを作成します。
        """
        # DBから既存のセッションをロードするロジック（簡略化）
        try:
            # 最新のセッションIDを取得 (または特定の方法でセッションを選択)
            # ここでは、簡略化のため、常にデフォルトタブを初期化します。
            # 実際のアプリケーションでは、db.list_matrix_sessions() などを使用して
            # ユーザーにセッションを選択させるか、最後に開いたセッションをロードします。
            pass
        except Exception as e:
            logging.error(f"ERROR: _load_session_or_default - Error loading session from DB: {e}")
            # エラーが発生しても続行し、デフォルトのタブを作成します。

        # デフォルトの単一タブを現在のプロンプトから作成 (Promptオブジェクトを格納)
        base_prompts = getattr(self, '_initial_prompts', self.prompts)
        base_prompts_filtered = {pid: p for pid, p in base_prompts.items() if getattr(p, 'include_in_matrix', False)}
        self._tabs = [{'name': tr('matrix.tab.default'), 'prompts_obj': {pid: (p.model_copy(deep=True) if hasattr(p, 'model_copy') else Prompt(**p.model_dump())) for pid, p in base_prompts_filtered.items()}, 'state': None}]
        self._active_tab_index = 0
        logging.debug(f"DEBUG: _load_session_or_default - Initializing with default tab from filtered base_prompts keys: {list(base_prompts_filtered.keys())}")
        # UIタブを構築
        self._rebuild_tabs()
        # 初回構築後、必要に応じて将来のロードでセッションを開けるようにフラグを無効化
        self._start_with_default_only = False




    def _add_prompt_set_tab(self):
        # Limit to 5 tabs
        if len(self._tabs) >= 5:
            CTkMessagebox(title=tr("matrix.tab.limit_title"), message=tr("matrix.tab.limit_message", max=5), icon="warning").wait_window()
            return
        # Show preset chooser
        preset = self._choose_preset_dialog()
        if preset is None:
            return
        name, prompts = preset
        # Snapshot current active tab BEFORE switching
        try:
            if 0 <= self._active_tab_index < len(self._tabs):
                self._tabs[self._active_tab_index]['prompts_obj'] = {pid: (p.model_copy(deep=True) if hasattr(p, 'model_copy') else Prompt(**p.model_dump())) for pid, p in self.prompts.items()}
                self._tabs[self._active_tab_index]['state'] = self._snapshot_state()
        except Exception:
            pass
        if not name:
            # Default tab name
            name = tr("matrix.tab.auto_name_fmt", n=len(self._tabs)+1)
        # Ensure display name uniqueness to avoid user confusion
        base_name = name
        suffix = 1
        existing = {t.get('name') for t in self._tabs}
        while name in existing:
            suffix += 1
            name = f"{base_name} ({suffix})"
        # Use prompts from preset (deserialize to Prompt objects). Empty set -> {}
        prompts_obj = self._deserialize_prompts(prompts) if isinstance(prompts, dict) else {}
        self._tabs.append({'name': name, 'prompts_obj': prompts_obj, 'state': None})
        self._active_tab_index = len(self._tabs) - 1
        self._rebuild_tabs()
        self._update_ui()

    def _save_active_prompt_set(self):
        """プリセット保存: 現在のアクティブタブのプロンプトセットのみを
        名前付きJSON（`prompt_set/<name>.json`）として保存します。これは再利用用のテンプレートで、
        セッションのタブ状態・結果は含みません。"""
        # Ask for name
        name = self._prompt_text_input(tr("matrix.preset.save_title"), tr("matrix.preset.save_label"))
        if not name:
            return
        try:
            # Save prompts to file
            import json, time
            serialized = self._serialize_prompts(self.prompts)
            self._tabs[self._active_tab_index]['name'] = name
            self._tabs[self._active_tab_index]['prompts_obj'] = {pid: (p.model_copy(deep=True) if hasattr(p, 'model_copy') else Prompt(**p.model_dump())) for pid, p in self.prompts.items()}
            file = self._prompt_set_dir() / f"{name}.json"
            file.write_text(json.dumps({'name': name, 'prompts': serialized}, ensure_ascii=False, indent=2), encoding='utf-8')
            # Update tab title
            self._rebuild_tabs()
            CTkMessagebox(title=tr("common.success"), message=tr("matrix.preset.saved"), icon="info").wait_window()
        except Exception as e:
            CTkMessagebox(title=tr("common.error"), message=tr("matrix.preset.save_failed", details=str(e)), icon="cancel").wait_window()

    def _delete_active_tab(self):
        """アクティブなタブ（プロンプトセット）を削除する"""
        if not self._tabs:
            return
        # 確認ダイアログ
        try:
            tab_name = self._tabs[self._active_tab_index]['name'] if 0 <= self._active_tab_index < len(self._tabs) else ""
        except Exception:
            tab_name = ""
        if not messagebox.askyesno(tr("common.delete_confirm_title"), tr("matrix.tab.delete_confirm", name=tab_name)):
            return
        try:
            del self._tabs[self._active_tab_index]
        except Exception:
            return
        # 最低1つは保持：空ならデフォルト空タブを作成
        if not self._tabs:
            self._tabs = [{'name': tr('matrix.tab.default'), 'prompts_obj': {}, 'state': None}]
            self._active_tab_index = 0
        else:
            # 削除位置に応じてインデックス調整
            if self._active_tab_index >= len(self._tabs):
                self._active_tab_index = len(self._tabs) - 1
        # UI再構築と保存
        self._rebuild_tabs()
        # 新しいアクティブタブの内容を反映
        try:
            t = self._tabs[self._active_tab_index]
            prompts_obj = t.get('prompts_obj', {})
            self.prompts = {pid: (p.model_copy(deep=True) if hasattr(p, 'model_copy') else Prompt(**p.model_dump())) for pid, p in prompts_obj.items()} if prompts_obj else {}
        except Exception:
            self.prompts = {}
        self.checkbox_states = []
        self.results = []
        self._full_results = []
        self._row_summaries = []
        self._col_summaries = []
        self._update_ui()
        try:
            self._save_session()
        except Exception:
            pass

    def _choose_preset_dialog(self) -> Optional[tuple[str, dict]]:
        # Build list of presets by mtime desc
        presets = []
        for p in self._prompt_set_dir().glob('*.json'):
            try:
                presets.append((p, p.stat().st_mtime))
            except Exception:
                pass
        presets.sort(key=lambda x: x[1], reverse=True)
        names = [pp[0].stem for pp in presets]
        # 固定選択肢: デフォルト, 空のセット
        names.insert(0, tr("matrix.preset.empty"))
        names.insert(0, tr("matrix.tab.default"))

        # Simple chooser using CTkOptionMenu in a small dialog
        dlg = ctk.CTkToplevel(self, fg_color=styles.HISTORY_ITEM_FG_COLOR)
        dlg.title(tr("matrix.preset.add_from_title"))
        dlg.geometry("360x160")
        dlg.transient(self)
        dlg# removed: .grab_set()
        var = ctk.StringVar(value=tr("matrix.tab.default") if names else tr("matrix.tab.default"))
        ctk.CTkLabel(dlg, text=tr("matrix.preset.choose_label"), text_color=styles.HISTORY_ITEM_TEXT_COLOR).pack(padx=12, pady=(16,6))
        menu = ctk.CTkOptionMenu(dlg, values=names, variable=var)
        menu.pack(padx=12, pady=6)
        chosen: list = []
        def _ok():
            chosen.append(var.get())
            dlg.destroy()
        ctk.CTkButton(dlg, text=tr("common.ok"), command=_ok, fg_color=styles.DEFAULT_BUTTON_FG_COLOR, text_color=styles.DEFAULT_BUTTON_TEXT_COLOR).pack(pady=10)
        self.wait_window(dlg)
        if not chosen:
            return None
        sel = chosen[0]
        if sel == tr("matrix.preset.empty"):
            return (tr("matrix.tab.auto_name_fmt", n=len(self._tabs)+1), {})
        if sel == tr("matrix.tab.default"):
            # 初期プロンプト群のうち include_in_matrix=True のもの
            base_prompts = getattr(self, '_initial_prompts', self.prompts)
            default_prompts = {pid: p for pid, p in base_prompts.items() if getattr(p, 'include_in_matrix', False)}
            return (tr("matrix.tab.default"), self._serialize_prompts(default_prompts))
        # Load preset
        try:
            import json
            file = self._prompt_set_dir() / f"{sel}.json"
            data = json.loads(file.read_text(encoding='utf-8'))
            prompts = data.get('prompts', {})
            return (data.get('name', sel), prompts)
        except Exception as e:
            CTkMessagebox(title=tr("common.error"), message=tr("matrix.preset.save_failed", details=str(e)), icon="cancel").wait_window()
            return None

    def _prompt_text_input(self, title: str, label: str, default: str = "") -> Optional[str]:
        dlg = ctk.CTkToplevel(self, fg_color=styles.HISTORY_ITEM_FG_COLOR)
        dlg.title(title)
        dlg.geometry("360x140")
        dlg.transient(self)
        dlg# removed: .grab_set()
        ctk.CTkLabel(dlg, text=label, text_color=styles.HISTORY_ITEM_TEXT_COLOR).pack(padx=12, pady=(16,6))
        entry = ctk.CTkEntry(dlg, fg_color=styles.HISTORY_ITEM_FG_COLOR, text_color=styles.HISTORY_ITEM_TEXT_COLOR)
        entry.pack(padx=12, pady=6, fill='x')
        try:
            entry.insert(0, default)
        except Exception:
            pass
        value: list[str] = []
        def _ok():
            value.append(entry.get().strip())
            dlg.destroy()
        ctk.CTkButton(dlg, text=tr("common.save"), command=_ok, fg_color=styles.DEFAULT_BUTTON_FG_COLOR, text_color=styles.DEFAULT_BUTTON_TEXT_COLOR).pack(pady=10)
        self.wait_window(dlg)
        return value[0] if value else None

    def _clear_active_set(self):
        # Clear prompts/results for active tab only; inputs remain shared
        try:
            self._tabs[self._active_tab_index]['prompts_obj'] = {}
            self.prompts = {}
        except Exception:
            self.prompts = {}
        self.checkbox_states = []
        self.results = []
        self._full_results = []
        self._row_summaries = []
        self._col_summaries = []
        self._result_textboxes = []
        self._cell_style = []
        self._update_ui()

    def _open_summary_settings(self):
        # 統一されたプロンプト管理画面を開く。入力グラブ等を解放し、アプリ画面を前面・フォーカスにする
        try:
            # クリップボード履歴ポップアップ等が掴んでいる場合は閉じてグラブを解放
            try:
                if hasattr(self, '_history_popup') and self._history_popup and self._history_popup.winfo_exists():
                    try:
                        if self._history_popup.grab_current() == str(self._history_popup):
                            self._history_popup.grab_release()
                    except Exception:
                        pass
                    self._history_popup.destroy()
                    self._history_popup = None
            except Exception:
                pass

            # どこかでgrabされていれば強制解放
            try:
                cur = self.tk.call('grab', 'current')
                if cur:
                    try:
                        self.tk.call('grab', 'release', cur)
                    except Exception:
                        try:
                            self.grab_release()
                        except Exception:
                            pass
            except Exception:
                pass

            # このウィンドウが入力グラブしている場合は解放
            try:
                if self.grab_current() == str(self):
                    self.grab_release()
            except Exception:
                pass

            # 念のため最前面やトランジェントを解除し、このウィンドウを一旦隠す
            try:
                self.transient(None)
            except Exception:
                pass
            try:
                self.attributes("-topmost", False)
            except Exception:
                pass
            # 表示は維持しつつ操作を無効化（可能なら）
            try:
                pass # removed: self.attributes('-disabled', True)
            except Exception:
                pass

            # プロンプト管理を前面に
            if hasattr(self.agent, '_show_main_window'):
                self.agent._show_main_window()
            # マネージャが閉じられたら再度このウィンドウの操作を有効化
            try:
                self._watch_manager_to_reenable()
            except Exception:
                pass
        except Exception:
            pass

    def _open_set_manager(self):
        dlg = ctk.CTkToplevel(self, fg_color=styles.HISTORY_ITEM_FG_COLOR)
        dlg.title(tr("matrix.set.manager_title"))
        dlg.geometry("520x240")
        dlg.transient(self)
        dlg# removed: .grab_set()
        ctk.CTkLabel(dlg, text=tr("matrix.set.save_delete"), text_color=styles.HISTORY_ITEM_TEXT_COLOR).pack(padx=12, pady=(12,6))
        # Save-as row (unified with session manager style)
        row1 = ctk.CTkFrame(dlg, fg_color="transparent")
        row1.pack(fill='x', padx=12, pady=6)
        name_entry = ctk.CTkEntry(row1, placeholder_text=tr("matrix.set.placeholder"), fg_color=styles.HISTORY_ITEM_FG_COLOR, text_color=styles.HISTORY_ITEM_TEXT_COLOR)
        name_entry.pack(side='left', fill='x', expand=True)
        def do_save():
            name = name_entry.get().strip()
            if not name:
                CTkMessagebox(title=tr("common.warning"), message=tr("matrix.set.name_required"), icon="warning").wait_window()
                return
            try:
                import json
                serialized = self._serialize_prompts(self.prompts)
                # 更新: タブ表示名とプロンプトオブジェクト
                self._tabs[self._active_tab_index]['name'] = name
                self._tabs[self._active_tab_index]['prompts_obj'] = {pid: (p.model_copy(deep=True) if hasattr(p, 'model_copy') else Prompt(**p.model_dump())) for pid, p in self.prompts.items()}
                file = self._prompt_set_dir() / f"{name}.json"
                file.write_text(json.dumps({'name': name, 'prompts': serialized}, ensure_ascii=False, indent=2), encoding='utf-8')
                self._rebuild_tabs()
                CTkMessagebox(title=tr("common.success"), message=tr("matrix.set.saved"), icon="info").wait_window()
                dlg.destroy()
            except Exception as e:
                CTkMessagebox(title=tr("common.error"), message=tr("matrix.set.save_failed", details=str(e)), icon="cancel").wait_window()
        ctk.CTkButton(row1, text=tr("common.save"), command=do_save, fg_color=styles.DEFAULT_BUTTON_FG_COLOR, text_color=styles.DEFAULT_BUTTON_TEXT_COLOR).pack(side='left', padx=(6,0))
        # 削除UI
        from pathlib import Path as _Path
        options = []
        try:
            options = [p.stem for p in self._prompt_set_dir().glob('*.json')]
        except Exception:
            options = []
        var = ctk.StringVar(value=options[0] if options else "")
        row2 = ctk.CTkFrame(dlg, fg_color="transparent")
        row2.pack(fill='x', padx=12, pady=6)
        ctk.CTkLabel(row2, text=tr("matrix.set.delete_target"), width=80, anchor='w', text_color=styles.HISTORY_ITEM_TEXT_COLOR).pack(side='left')
        menu = ctk.CTkOptionMenu(row2, values=options or [""], variable=var)
        menu.pack(side='left', padx=(6,6))
        def do_delete():
            name = var.get().strip()
            if not name:
                return
            try:
                f = self._prompt_set_dir() / f"{name}.json"
                if f.exists():
                    f.unlink()
                    CTkMessagebox(title=tr("common.success"), message=tr("matrix.set.deleted"), icon="info").wait_window()
                    dlg.destroy()
            except Exception as e:
                CTkMessagebox(title=tr("common.error"), message=tr("matrix.set.delete_failed", details=str(e)), icon="cancel").wait_window()
        ctk.CTkButton(row2, text=tr("common.delete"), command=do_delete, fg_color=styles.DELETE_BUTTON_COLOR, text_color=styles.DEFAULT_BUTTON_TEXT_COLOR, hover_color=styles.DELETE_BUTTON_HOVER_COLOR).pack(side='left', padx=(6,0))

    def _open_session_manager(self):
        # Release any stale grab before opening manager
        try:
            cur = self.tk.call('grab', 'current')
            if cur:
                self.tk.call('grab', 'release', cur)
        except Exception:
            pass
        dlg = ctk.CTkToplevel(self, fg_color=styles.HISTORY_ITEM_FG_COLOR)
        dlg.title(tr("matrix.session.manager_title"))
        # Widen width to avoid button cutoff; reduce height for tighter clearance
        dlg.geometry("680x200")
        dlg.transient(self)
        try:
            dlg.lift(self)
            dlg.attributes('-topmost', True)
            dlg.focus_force()
            dlg.after(200, lambda: dlg.attributes('-topmost', False))
        except Exception:
            pass
        dlg# removed: .grab_set()
        ctk.CTkLabel(dlg, text=tr("matrix.session.save_load_delete"), text_color=styles.HISTORY_ITEM_TEXT_COLOR, anchor='w', justify='left').pack(padx=12, pady=(8,4), fill='x')

        # Layout constants
        LABEL_W = 90
        LIST_W = 320
        BTN_W = 80

        # --- Sessions row ---
        row_s = ctk.CTkFrame(dlg, fg_color="transparent")
        row_s.pack(fill='x', padx=12, pady=(4,2))
        row_s.grid_columnconfigure(0, weight=0)
        row_s.grid_columnconfigure(1, weight=0)
        row_s.grid_columnconfigure(2, weight=0)
        row_s.grid_columnconfigure(3, weight=0)
        ctk.CTkLabel(row_s, text=tr("matrix.sessions.db_section"), width=LABEL_W, anchor='w', text_color=styles.HISTORY_ITEM_TEXT_COLOR).grid(row=0, column=0, sticky='w')
        try:
            sess_rows = db.list_matrix_sessions()
            sess_opts = [f"{r['name']} (#{r['id']})" for r in sess_rows]
        except Exception:
            sess_opts = []
        db_sess_var = ctk.StringVar(value=(sess_opts[0] if sess_opts else ""))
        db_sess_menu = ctk.CTkOptionMenu(row_s, values=sess_opts or [""], variable=db_sess_var, width=LIST_W)
        db_sess_menu.grid(row=0, column=1, padx=(6,6), sticky='w')

        def rename_session():
            try:
                sel = db_sess_var.get()
                if not sel:
                    return
                sid = int(sel.split('#')[-1].rstrip(')'))
                top = ctk.CTkToplevel(dlg)
                top.title(tr("common.edit"))
                try:
                    top.transient(dlg)
                except Exception:
                    pass
                try:
                    top.lift(); top.focus_force(); top.attributes('-topmost', True)
                except Exception:
                    pass
                e = ctk.CTkEntry(top)
                e.pack(padx=10, pady=10)
                e.insert(0, sel.split(' (#')[0])
                def ok():
                    new = e.get().strip()
                    if new:
                        db.rename_matrix_session(sid, new)
                    try:
                        top.grab_release()
                    except Exception:
                        pass
                    top.destroy()
                    # refresh session list
                    try:
                        new_rows = db.list_matrix_sessions()
                        new_opts = [f"{r['name']} (#{r['id']})" for r in new_rows]
                        db_sess_var.set(new_opts[0] if new_opts else "")
                        db_sess_menu.configure(values=new_opts or [""])
                        refresh_tabs_for_selected_session()
                    except Exception:
                        pass
                ctk.CTkButton(top, text=tr("common.save"), command=ok).pack(pady=(0,10))
            except Exception:
                pass

        def delete_session():
            try:
                sel = db_sess_var.get()
                if not sel:
                    return
                sid = int(sel.split('#')[-1].rstrip(')'))
                db.delete_matrix_session(sid)
                # refresh
                new_rows = db.list_matrix_sessions()
                new_opts = [f"{r['name']} (#{r['id']})" for r in new_rows]
                db_sess_var.set(new_opts[0] if new_opts else "")
                db_sess_menu.configure(values=new_opts or [""])
                refresh_tabs_for_selected_session()
            except Exception:
                pass

        ctk.CTkButton(row_s, text=tr("common.edit"), width=BTN_W, command=rename_session, fg_color=styles.DEFAULT_BUTTON_FG_COLOR, text_color=styles.DEFAULT_BUTTON_TEXT_COLOR).grid(row=0, column=2, padx=(6,6))
        ctk.CTkButton(row_s, text=tr("common.delete"), width=BTN_W, command=delete_session, fg_color=styles.DELETE_BUTTON_COLOR, text_color=styles.DEFAULT_BUTTON_TEXT_COLOR, hover_color=styles.DELETE_BUTTON_HOVER_COLOR).grid(row=0, column=3)

        # --- Tabs row ---
        row_t = ctk.CTkFrame(dlg, fg_color="transparent")
        row_t.pack(fill='x', padx=12, pady=(2,6))
        for i in range(4):
            row_t.grid_columnconfigure(i, weight=0)
        ctk.CTkLabel(row_t, text=tr("matrix.sessions.tabs"), width=LABEL_W, anchor='w', text_color=styles.HISTORY_ITEM_TEXT_COLOR).grid(row=0, column=0, sticky='w')
        db_tab_var = ctk.StringVar(value="")
        db_tab_menu = ctk.CTkOptionMenu(row_t, values=[""], variable=db_tab_var, width=LIST_W)
        db_tab_menu.grid(row=0, column=1, padx=(6,6), sticky='w')

        def rename_tab():
            try:
                sel = db_tab_var.get()
                if not sel:
                    return
                tid = int(sel.split('#')[-1].rstrip(')'))
                top = ctk.CTkToplevel(dlg)
                top.title(tr("common.edit"))
                try:
                    top.transient(dlg)
                except Exception:
                    pass
                try:
                    top.lift(); top.focus_force(); top.attributes('-topmost', True)
                except Exception:
                    pass
                e = ctk.CTkEntry(top)
                e.pack(padx=10, pady=10)
                e.insert(0, sel.split(' (#')[0])
                def ok():
                    new = e.get().strip()
                    if new:
                        db.rename_matrix_tab(tid, new)
                    try:
                        top.grab_release()
                    except Exception:
                        pass
                    top.destroy()
                    refresh_tabs_for_selected_session()
                ctk.CTkButton(top, text=tr("common.save"), command=ok).pack(pady=(0,10))
            except Exception:
                pass

        def delete_tab():
            try:
                sel = db_tab_var.get()
                if not sel:
                    return
                tid = int(sel.split('#')[-1].rstrip(')'))
                db.delete_matrix_tab(tid)
                refresh_tabs_for_selected_session()
            except Exception:
                pass

        ctk.CTkButton(row_t, text=tr("common.edit"), width=BTN_W, command=rename_tab, fg_color=styles.DEFAULT_BUTTON_FG_COLOR, text_color=styles.DEFAULT_BUTTON_TEXT_COLOR).grid(row=0, column=2, padx=(6,6))
        ctk.CTkButton(row_t, text=tr("common.delete"), width=BTN_W, command=delete_tab, fg_color=styles.DELETE_BUTTON_COLOR, text_color=styles.DEFAULT_BUTTON_TEXT_COLOR, hover_color=styles.DELETE_BUTTON_HOVER_COLOR).grid(row=0, column=3)

        # --- helpers ---
        def refresh_tabs_for_selected_session(*_):
            try:
                sel = db_sess_var.get()
                if not sel:
                    db_tab_var.set("")
                    db_tab_menu.configure(values=[""])
                    return
                sid = int(sel.split('#')[-1].rstrip(')'))
                tabs = db.list_matrix_tabs(sid)
                tab_opts = [f"{t['name']} (#{t['id']})" for t in tabs]
                db_tab_var.set(tab_opts[0] if tab_opts else "")
                db_tab_menu.configure(values=tab_opts or [""], variable=db_tab_var)
            except Exception:
                db_tab_var.set("")
                db_tab_menu.configure(values=[""])

        db_sess_menu.configure(command=lambda *_: refresh_tabs_for_selected_session())
        refresh_tabs_for_selected_session()

    def _open_session_loader(self):
        """Lightweight loader for sessions (file and DB) with clear guidance."""
        # Release any stale grab before opening loader
        try:
            cur = self.tk.call('grab', 'current')
            if cur:
                self.tk.call('grab', 'release', cur)
        except Exception:
            pass
        dlg = ctk.CTkToplevel(self, fg_color=styles.HISTORY_ITEM_FG_COLOR)
        dlg.title(tr("matrix.toolbar.load"))
        dlg.geometry("520x300")
        dlg.transient(self)
        try:
            dlg.lift(self)
            dlg.attributes('-topmost', True)
            dlg.focus_force()
            dlg.after(200, lambda: dlg.attributes('-topmost', False))
        except Exception:
            pass
        dlg# removed: .grab_set()

        wrap_w = 480
        ctk.CTkLabel(dlg, text=tr("matrix.toolbar.load_hint"), text_color=styles.HISTORY_ITEM_TEXT_COLOR, anchor='w', justify='left', wraplength=wrap_w).pack(padx=12, pady=(12,6), fill='x')
        ctk.CTkLabel(dlg, text=tr("matrix.loader.help_file"), text_color=styles.HISTORY_ITEM_TEXT_COLOR, anchor='w', justify='left', wraplength=wrap_w).pack(padx=12, pady=(0,2), fill='x')
        ctk.CTkLabel(dlg, text=tr("matrix.loader.help_db"), text_color=styles.HISTORY_ITEM_TEXT_COLOR, anchor='w', justify='left', wraplength=wrap_w).pack(padx=12, pady=(0,8), fill='x')


        # DB sessions
        db_frame = ctk.CTkFrame(dlg, fg_color="transparent")
        db_frame.pack(fill='x', padx=12, pady=(6,6))
        db_frame.grid_columnconfigure(1, weight=1)
        ctk.CTkLabel(db_frame, text=tr("matrix.sessions.db"), width=80, anchor='w', text_color=styles.HISTORY_ITEM_TEXT_COLOR).grid(row=0, column=0, sticky='w')
        try:
            sess_rows = db.list_matrix_sessions()
            sess_opts = [f"{r['name']} (#{r['id']})" for r in sess_rows]
        except Exception:
            sess_opts = []
        db_sess_var = ctk.StringVar(value=sess_opts[0] if sess_opts else "")
        db_sess_menu = ctk.CTkOptionMenu(db_frame, values=sess_opts or [""], variable=db_sess_var)
        db_sess_menu.grid(row=0, column=1, sticky='ew', padx=(6,0))

        ctk.CTkLabel(db_frame, text=tr("matrix.sessions.tab"), width=80, anchor='w', text_color=styles.HISTORY_ITEM_TEXT_COLOR).grid(row=1, column=0, sticky='w', pady=(6,0))
        db_tab_var = ctk.StringVar(value="")
        db_tab_menu = ctk.CTkOptionMenu(db_frame, values=[""], variable=db_tab_var)
        db_tab_menu.grid(row=1, column=1, sticky='ew', padx=(6,0), pady=(6,0))

        def refresh_db_tabs(*_):
            sel = db_sess_var.get()
            if not sel:
                db_tab_menu.configure(values=[""])
                db_tab_var.set("")
                return
            try:
                sid = int(sel.split('#')[-1].rstrip(')'))
                tabs = db.list_matrix_tabs(sid)
                tab_opts = [f"{t['name']} (#{t['id']})" for t in tabs]
                db_tab_var.set(tab_opts[0] if tab_opts else "")
                db_tab_menu.configure(values=tab_opts or [""])
            except Exception:
                db_tab_var.set("")
                db_tab_menu.configure(values=[""])

        db_sess_menu.configure(command=lambda *_: refresh_db_tabs())
        refresh_db_tabs()

        # Footer actions
        footer = ctk.CTkFrame(dlg, fg_color="transparent")
        footer.pack(fill='x', padx=12, pady=(8,12))
        footer.grid_columnconfigure((0,1,2), weight=1)


        def load_db_tab():
            sel_s = db_sess_var.get()
            sel_t = db_tab_var.get()
            if not sel_s or not sel_t:
                return
            try:
                tid = int(sel_t.split('#')[-1].rstrip(')'))
                try:
                    dlg.grab_release()
                except Exception:
                    pass
                dlg.destroy()
                # Load safely in next loop turn
                self._load_tab_async_safely(tid)
                self._current_session_name = f"DB {sel_s} / Tab {tid}"
                self._session_label_var.set(f"{tr('matrix.toolbar.session_label')} {self._current_session_name}")
                # Avoid modal success box to reduce focus issues
            except Exception as e:
                try:
                    CTkMessagebox(title=tr("common.error"), message=tr("matrix.session.load_failed", details=str(e)), icon="cancel")
                except Exception:
                    pass

        ctk.CTkButton(footer, text=tr("matrix.toolbar.load"), command=load_db_tab, fg_color=styles.DEFAULT_BUTTON_FG_COLOR, text_color=styles.DEFAULT_BUTTON_TEXT_COLOR).grid(row=0, column=0, padx=(0,6), sticky='ew')
        ctk.CTkButton(footer, text=tr("common.cancel"), command=dlg.destroy, fg_color=styles.CANCEL_BUTTON_COLOR, text_color=styles.CANCEL_BUTTON_TEXT_COLOR).grid(row=0, column=1, sticky='ew')

    def _watch_manager_to_reenable(self):
        try:
            # 管理画面（root）が非表示または最小化されたら再有効化
            if not self.parent_app.winfo_viewable():
                try:
                    self.attributes('-disabled', False)
                except Exception:
                    pass
                try:
                    self.lift()
                    self.focus_force()
                except Exception:
                    pass
                return
        except Exception:
            return
        # まだ開いている場合は定期的に監視
        try:
            self.after(300, self._watch_manager_to_reenable)
        except Exception:
            pass

    def _create_main_grid_frame(self):
        """マトリクスグリッドを配置するためのスクロール可能フレームを作成する"""
        self.canvas_frame = ctk.CTkFrame(self, fg_color=styles.MATRIX_CANVAS_BACKGROUND_COLOR)
        self.canvas_frame.pack(fill="both", expand=True, padx=10, pady=(0,5))

        current_fg_color = self.canvas_frame.cget("fg_color")
        appearance_mode_index = 0 if ctk.get_appearance_mode() == "Light" else 1
        canvas_bg_color = current_fg_color[appearance_mode_index] if isinstance(current_fg_color, tuple) else current_fg_color
        self.canvas = ctk.CTkCanvas(self.canvas_frame, highlightthickness=0, bg=canvas_bg_color)
        self.canvas.pack(side="left", fill="both", expand=True)

        self.v_scrollbar = ctk.CTkScrollbar(self.canvas_frame, orientation="vertical", command=self.canvas.yview)
        self.v_scrollbar.pack(side="right", fill="y")
        self.canvas.configure(yscrollcommand=self.v_scrollbar.set)

        self.h_scrollbar = ctk.CTkScrollbar(self, orientation="horizontal", command=self.canvas.xview)
        self.h_scrollbar.pack(fill="x", side="bottom", padx=10, pady=(0, 10))
        self.canvas.configure(xscrollcommand=self.h_scrollbar.set)

        # タブごとの内容フレーム（キャッシュ）
        self.scrollable_content_frame = ctk.CTkFrame(self.canvas, fg_color="transparent")
        self._window_id = self.canvas.create_window((0, 0), window=self.scrollable_content_frame, anchor="nw")
        # タブにフレームを格納し、切替時はフレームを差し替える
        try:
            if 0 <= self._active_tab_index < len(self._tabs):
                self._tabs[self._active_tab_index]['frame'] = self.scrollable_content_frame
        except Exception:
            pass

        self.scrollable_content_frame.bind("<Configure>", self._on_frame_configure)
        self.canvas.bind("<Configure>", self._on_canvas_configure)

        self.scrollable_content_frame.grid_columnconfigure(0, weight=0)
        self.scrollable_content_frame.grid_rowconfigure(0, weight=0)

        self.run_button_frame = ctk.CTkFrame(self, fg_color=styles.MATRIX_TOP_BG_COLOR)
        self.run_button_frame.pack(fill="x", padx=10, pady=10, side="bottom")
        self.run_button_frame.grid_columnconfigure((0, 1, 2, 3, 4, 5), weight=1)

        # Order: 実行, フロー実行, 行まとめ, 列まとめ, 行列まとめ, エクセル出力
        ctk.CTkButton(self.run_button_frame, text=tr("matrix.run"), command=self._run_batch_processing, fg_color=styles.DEFAULT_BUTTON_FG_COLOR, text_color=styles.DEFAULT_BUTTON_TEXT_COLOR).grid(row=0, column=0, padx=5, pady=5, sticky="ew")
        
        self.flow_run_button = ctk.CTkButton(self.run_button_frame, text=tr("matrix.run_flow"), command=self._run_flow_processing, fg_color=styles.DEFAULT_BUTTON_FG_COLOR, text_color=styles.DEFAULT_BUTTON_TEXT_COLOR)
        self.flow_run_button.grid(row=0, column=1, padx=5, pady=5, sticky="ew")

        self.summarize_row_button = ctk.CTkButton(self.run_button_frame, text=tr("matrix.run_row_summary"), command=self._summarize_rows, state="disabled", fg_color=styles.DEFAULT_BUTTON_FG_COLOR, text_color=styles.DEFAULT_BUTTON_TEXT_COLOR)
        self.summarize_row_button.grid(row=0, column=2, padx=5, pady=5, sticky="ew")

        self.summarize_col_button = ctk.CTkButton(self.run_button_frame, text=tr("matrix.run_col_summary"), command=self._summarize_columns, state="disabled", fg_color=styles.DEFAULT_BUTTON_FG_COLOR, text_color=styles.DEFAULT_BUTTON_TEXT_COLOR)
        self.summarize_col_button.grid(row=0, column=3, padx=5, pady=5, sticky="ew")

        self.summarize_matrix_button = ctk.CTkButton(self.run_button_frame, text=tr("matrix.matrix_summary"), command=self._summarize_matrix, state="disabled", fg_color=styles.DEFAULT_BUTTON_FG_COLOR, text_color=styles.DEFAULT_BUTTON_TEXT_COLOR)
        self.summarize_matrix_button.grid(row=0, column=4, padx=5, pady=5, sticky="ew")

        self.export_excel_button = ctk.CTkButton(self.run_button_frame, text="Excel", command=self._export_to_excel, state="disabled", fg_color=styles.DEFAULT_BUTTON_FG_COLOR, text_color=styles.DEFAULT_BUTTON_TEXT_COLOR)
        self.export_excel_button.grid(row=0, column=5, padx=5, pady=5, sticky="ew")

    def _on_frame_configure(self, event):
        self.canvas.configure(scrollregion=self.canvas.bbox("all"))
        self.tooltip_window = None

    def _on_canvas_configure(self, event):
        self.canvas.coords(self._window_id, 0, 0)
        self._on_frame_configure(event)

    def _update_input_row_display(self, row_idx: int):
        for widget in self.scrollable_content_frame.grid_slaves(row=row_idx + 1, column=0):
            widget.destroy()
        self._add_input_row_widgets(row_idx, self.input_data[row_idx])

    def _show_tooltip(self, text: str):
        try:
            # Debounce rapid enter/leave; schedule with slight delay to reduce flicker
            if hasattr(self, '_tooltip_after_id') and self._tooltip_after_id:
                try:
                    self.after_cancel(self._tooltip_after_id)  # type: ignore[arg-type]
                except Exception:
                    pass
                self._tooltip_after_id = None

            def _do_show():
                x = int(self.winfo_pointerx() + 16)
                y = int(self.winfo_pointery() + 12)
                if not getattr(self, 'tooltip_window', None) or not self.tooltip_window.winfo_exists():
                    self.tooltip_window = tk.Toplevel(self)
                    try:
                        self.tooltip_window.wm_overrideredirect(True)
                    except Exception:
                        pass
                    # Cache label for quick text updates
                    self._tooltip_label = tk.Label(
                        self.tooltip_window,
                        text=text,
                        justify='left',
                        background="#ffffe0",
                        relief='solid',
                        borderwidth=1,
                        font=("tahoma", "8", "normal")
                    )
                    self._tooltip_label.pack(ipadx=4, ipady=2)
                else:
                    try:
                        self._tooltip_label.configure(text=text)  # type: ignore[attr-defined]
                    except Exception:
                        pass
                # Position: must include leading '+'
                try:
                    self.tooltip_window.geometry(f"+{x}+{y}")
                except Exception:
                    # Fallback to wm_geometry with proper format
                    try:
                        self.tooltip_window.wm_geometry(f"+{x}+{y}")
                    except Exception:
                        pass
                try:
                    self.tooltip_window.attributes('-topmost', True)
                except Exception:
                    pass
                # Auto hide after 2 seconds to avoid lingering tooltips
                try:
                    if hasattr(self, '_tooltip_auto_hide_id') and self._tooltip_auto_hide_id:
                        self.after_cancel(self._tooltip_auto_hide_id)  # type: ignore[arg-type]
                    self._tooltip_auto_hide_id = self.after(2000, self._hide_tooltip)
                except Exception:
                    pass

            # Show after 120ms to avoid flicker on fast hover
            self._tooltip_after_id = self.after(120, _do_show)
        except Exception:
            pass

    def _hide_tooltip(self):
        try:
            if hasattr(self, '_tooltip_after_id') and self._tooltip_after_id:
                try:
                    self.after_cancel(self._tooltip_after_id)  # type: ignore[arg-type]
                except Exception:
                    pass
                self._tooltip_after_id = None
            if getattr(self, 'tooltip_window', None) and self.tooltip_window.winfo_exists():
                self.tooltip_window.destroy()
            try:
                if hasattr(self, '_tooltip_auto_hide_id') and self._tooltip_auto_hide_id:
                    self.after_cancel(self._tooltip_auto_hide_id)  # type: ignore[arg-type]
                    self._tooltip_auto_hide_id = None
            except Exception:
                pass
        except Exception:
            pass
        self.tooltip_window = None

    def _update_row_summary_column(self):
        """行まとめ列の表示/非表示、および内容の更新を行う"""
        from gemclip.features.matrix.view import update_row_summary_column
        return update_row_summary_column(self)

    def _update_column_summary_row(self):
        """列まとめ行の表示/非表示、および内容の更新を行う"""
        from gemclip.features.matrix.view import update_column_summary_row
        return update_column_summary_row(self)

    def _update_ui(self):
        self.scrollable_content_frame.update_idletasks()
        if self._is_closing or not self.winfo_exists():
            return

        for widget in self.scrollable_content_frame.winfo_children():
            widget.destroy()

        configure_scrollable_grid(self)

        for col_idx, (prompt_id, prompt_config) in enumerate(self.prompts.items()):
            self._add_prompt_header_widgets(col_idx, prompt_id, prompt_config)

        for row_idx, input_item in enumerate(self.input_data):
            self._add_input_row_widgets(row_idx, input_item)

        self._update_row_summary_column()
        self._update_column_summary_row()
        # アクティブタブに現在の変数参照を保存（高速切替用）
        save_active_tab_vars(self)

    def _add_prompt_header_widgets(self, col_idx: int, prompt_id: str, prompt_config: Prompt):
        header_frame = ctk.CTkFrame(self.scrollable_content_frame, border_width=1, border_color=styles.MATRIX_HEADER_BORDER_COLOR, width=styles.MATRIX_CELL_WIDTH, height=styles.MATRIX_RESULT_CELL_HEIGHT)
        header_frame.grid(row=0, column=col_idx + 1, padx=5, pady=5, sticky="nsew")
        header_frame.grid_propagate(False)
        header_frame.grid_columnconfigure(0, weight=1)
        header_frame.grid_rowconfigure(1, weight=1)

        # Track for drag-and-drop reordering
        setattr(header_frame, "_prompt_id", prompt_id)
        while len(self._col_header_frames) <= col_idx:
            self._col_header_frames.append(None)  # type: ignore
        self._col_header_frames[col_idx] = header_frame

        # Bind drag events to the header frame
        header_frame.bind("<ButtonPress-1>", self._on_col_press)
        header_frame.bind("<B1-Motion>", self._on_col_motion)
        header_frame.bind("<ButtonRelease-1>", self._on_col_release)

        col_header_inner_frame = ctk.CTkFrame(header_frame)
        col_header_inner_frame.pack(fill="x", padx=2, pady=2)
        col_header_inner_frame.grid_columnconfigure(0, weight=1)
        col_header_inner_frame.grid_columnconfigure(1, weight=0)

        col_letter = chr(ord('A') + col_idx)
        col_num_label = ctk.CTkLabel(col_header_inner_frame, text=f"{col_letter}", font=styles.MATRIX_FONT_BOLD, anchor="center")
        col_num_label.grid(row=0, column=0, sticky="ew")
        # Bind drag events to child widgets as well so drags starting on them work
        col_num_label.bind("<ButtonPress-1>", self._on_col_press)
        col_num_label.bind("<B1-Motion>", self._on_col_motion)
        col_num_label.bind("<ButtonRelease-1>", self._on_col_release)

        delete_col_icon = None
        try:
            icon_path = Path(DELETE_ICON_FILE)
            if icon_path.exists():
                from PIL import Image
                icon_img = Image.open(icon_path)
                size = (16, 16)
                icon_img.thumbnail(size)
                delete_col_icon = ctk.CTkImage(light_image=icon_img, dark_image=icon_img, size=size)
        except Exception:
            delete_col_icon = None

        delete_col_button = ctk.CTkButton(col_header_inner_frame, text="" if delete_col_icon else tr("common.delete"), image=delete_col_icon, width=24, height=24, fg_color=styles.MATRIX_DELETE_BUTTON_COLOR, hover_color=styles.MATRIX_DELETE_BUTTON_HOVER_COLOR, command=lambda c=col_idx: self._delete_column(c))
        delete_col_button.grid(row=0, column=1, padx=(5, 0), sticky="e")
        delete_col_button.bind("<ButtonPress-1>", self._on_col_press)
        delete_col_button.bind("<B1-Motion>", self._on_col_motion)
        delete_col_button.bind("<ButtonRelease-1>", self._on_col_release)
        
        prompt_name_entry = ctk.CTkEntry(header_frame, placeholder_text=tr("prompt.header_name"))
        prompt_name_entry.insert(0, prompt_config.name)
        prompt_name_entry.configure(state="readonly")
        # Open editor only on click release when not dragging
        # Single-click to open editor (debounced after loads)
        prompt_name_entry.bind("<ButtonRelease-1>", lambda e=None, p_id=prompt_id: self._open_editor_if_not_drag(p_id))
        # Also allow drag start from the entry area
        prompt_name_entry.bind("<ButtonPress-1>", self._on_col_press)
        prompt_name_entry.bind("<B1-Motion>", self._on_col_motion)
        prompt_name_entry.bind("<ButtonRelease-1>", self._on_col_release)
        prompt_name_entry.pack(fill="x", padx=2, pady=2)

        inner_h = max(20, styles.MATRIX_RESULT_CELL_HEIGHT - 10)
        system_prompt_textbox = ctk.CTkTextbox(header_frame, height=inner_h, wrap="word")
        setup_textbox_right_click_menu(system_prompt_textbox)
        system_prompt_textbox.insert("1.0", prompt_config.system_prompt)
        try:
            system_prompt_textbox.tag_configure("left", justify="left")
            system_prompt_textbox.tag_add("left", "1.0", "end")
        except Exception:
            pass
        system_prompt_textbox.configure(state="disabled")
        system_prompt_textbox.bind("<ButtonRelease-1>", lambda e=None, p_id=prompt_id: self._open_editor_if_not_drag(p_id))
        system_prompt_textbox.bind("<ButtonPress-1>", self._on_col_press)
        system_prompt_textbox.bind("<B1-Motion>", self._on_col_motion)
        system_prompt_textbox.bind("<ButtonRelease-1>", self._on_col_release)
        system_prompt_textbox.pack(fill="both", expand=True, padx=2, pady=2)

    # --- Column drag-and-drop handlers (similar to prompt manager row DnD) ---
    def _on_col_press(self, event):
        try:
            from gemclip.features.matrix.view import compute_col_drop_index
            idx = compute_col_drop_index(self, event.x_root)
            if not (0 <= idx < len(self._col_header_frames)):
                return
            target_frame = self._col_header_frames[idx]
            if target_frame is None:
                return
            self._col_drag_data = {"frame": target_frame, "index": idx, "current_index": idx, "moved": False, "start_x": event.x_root}
            # Highlight dragged column header
            if self._col_drag_active_frame and self._col_drag_active_frame.winfo_exists():
                try:
                    self._col_drag_active_frame.configure(fg_color=styles.HISTORY_ITEM_FG_COLOR)
                except Exception:
                    pass
            self._col_drag_active_frame = target_frame
            try:
                self._col_drag_active_frame.configure(fg_color=styles.DRAG_ACTIVE_ROW_COLOR)
            except Exception:
                pass
        except Exception:
            return

    def _on_col_motion(self, event):
        if not self._col_drag_data:
            return
        try:
            # Mark as moved when exceeding small threshold
            try:
                if abs(int(event.x_root) - int(self._col_drag_data.get("start_x", event.x_root))) > 3:
                    self._col_drag_data["moved"] = True
            except Exception:
                self._col_drag_data["moved"] = True
            from gemclip.features.matrix.view import compute_col_drop_index, draw_col_drop_indicator
            new_index = compute_col_drop_index(self, event.x_root)
            current_index = self._col_drag_data.get("current_index", 0)
            if 0 <= new_index < len(self._col_header_frames):
                # Draw a white boundary indicator line between columns
                draw_col_drop_indicator(self, event.x_root)
            if new_index != current_index and 0 <= new_index < len(self._col_header_frames):
                self._col_drag_data["current_index"] = new_index
        except Exception:
            return

    def _on_col_release(self, event):
        if not self._col_drag_data:
            return
        try:
            # Compute final index and reorder prompt mapping
            from gemclip.features.matrix.view import compute_col_drop_index
            drop_index = compute_col_drop_index(self, event.x_root)
            start_index = self._col_drag_data.get("index", 0)
            if 0 <= drop_index < len(self._col_header_frames) and drop_index != start_index:
                # Compute old/new id order
                old_ids = list(self.prompts.keys())
                moving_id = old_ids.pop(start_index)
                new_ids = old_ids.copy()
                new_ids.insert(drop_index, moving_id)

                # Reorder per-column state arrays to match new order
                try:
                    id_to_old_index = {pid: i for i, pid in enumerate(list(self.prompts.keys()))}
                    # checkbox, results, full_results
                    for r in range(len(self.input_data)):
                        if r < len(self.checkbox_states):
                            old_row = self.checkbox_states[r]
                            new_row = []
                            for pid in new_ids:
                                oi = id_to_old_index.get(pid, None)
                                if oi is None or oi >= len(old_row):
                                    new_row.append(ctk.BooleanVar(value=False))
                                else:
                                    new_row.append(old_row[oi])
                            self.checkbox_states[r] = new_row
                        if r < len(self.results):
                            old_row_r = self.results[r]
                            new_row_r = []
                            for pid in new_ids:
                                oi = id_to_old_index.get(pid, None)
                                if oi is None or oi >= len(old_row_r):
                                    new_row_r.append(ctk.StringVar(value=""))
                                else:
                                    new_row_r.append(old_row_r[oi])
                            self.results[r] = new_row_r
                        if r < len(self._full_results):
                            old_row_f = self._full_results[r]
                            new_row_f = []
                            for pid in new_ids:
                                oi = id_to_old_index.get(pid, None)
                                if oi is None or oi >= len(old_row_f):
                                    new_row_f.append("")
                                else:
                                    new_row_f.append(old_row_f[oi])
                            self._full_results[r] = new_row_f
                    # per-column summaries
                    if self._col_summaries:
                        old_cols = self._col_summaries
                        new_cols = []
                        for pid in new_ids:
                            oi = id_to_old_index.get(pid, None)
                            if oi is None or oi >= len(old_cols):
                                new_cols.append(ctk.StringVar(value=""))
                            else:
                                new_cols.append(old_cols[oi])
                        self._col_summaries = new_cols
                    # widths
                    if self._column_widths:
                        old_w = self._column_widths
                        # Ensure length equals number of prompts
                        while len(old_w) < len(new_ids):
                            old_w.append(styles.MATRIX_CELL_WIDTH)
                        new_w = []
                        for pid in new_ids:
                            oi = id_to_old_index.get(pid, 0)
                            new_w.append(old_w[oi])
                        self._column_widths = new_w
                except Exception:
                    pass

                # Rebuild dict in new order
                new_prompts = {pid: self.prompts[pid] for pid in new_ids}
                self.prompts = new_prompts
                # Rebuild UI to reflect new column order
                self._update_ui()
            # Reset highlights
            for fr in self._col_header_frames:
                if fr is None or not fr.winfo_exists():
                    continue
                try:
                    fr.configure(fg_color=styles.HISTORY_ITEM_FG_COLOR)
                except Exception:
                    pass
        except Exception:
            pass
        finally:
            self._col_drag_data = {}
            self._col_drag_active_frame = None
            # Remove drop indicator line
            # Remove drop indicator (line/frame)
            try:
                if self._col_drop_line_id is not None:
                    self.canvas.delete(self._col_drop_line_id)
            except Exception:
                pass
            self._col_drop_line_id = None
            try:
                if self._col_drop_indicator_widget is not None and self._col_drop_indicator_widget.winfo_exists():
                    self._col_drop_indicator_widget.destroy()
            except Exception:
                pass
            self._col_drop_indicator_widget = None

    def _delete_column(self, col_idx: int):
        try:
            from gemclip.features.matrix.colops import delete_column
            return delete_column(self, col_idx)
        except Exception:
            # Fallback: no-op if helper import fails; avoids hard crash
            pass

    def _compute_col_drop_index(self, x_root: int) -> int:
        from gemclip.features.matrix.view import compute_col_drop_index
        return compute_col_drop_index(self, x_root)

    def _draw_col_drop_indicator(self, x_root: int) -> None:
        from gemclip.features.matrix.view import draw_col_drop_indicator
        return draw_col_drop_indicator(self, x_root)

    def _open_editor_if_not_drag(self, prompt_id: str):
        # Only open editor when no drag occurred recently
        try:
            if self._col_drag_data and self._col_drag_data.get("moved"):
                return
        except Exception:
            pass
        # Debounce shortly after a load to avoid accidental open triggered by focus/geometry changes
        try:
            import time as _t
            if getattr(self, '_just_loaded_at', None):
                if _t.time() - float(self._just_loaded_at) < 0.4:
                    return
        except Exception:
            pass
        self._open_prompt_editor(prompt_id)

    def _add_input_row_widgets(self, row_idx: int, input_item: Dict[str, Any]):
        from gemclip.features.matrix.view import add_input_row_widgets
        return add_input_row_widgets(self, row_idx, input_item)

    def _create_result_cell(self, row_idx: int, col_idx: int):
        """指定された行と列に結果表示用のセルウィジェットを作成する"""
        while len(self.checkbox_states) <= row_idx:
            self.checkbox_states.append([])
        while len(self.checkbox_states[row_idx]) <= col_idx:
            self.checkbox_states[row_idx].append(ctk.BooleanVar(value=False))
        
        while len(self.results) <= row_idx:
            self.results.append([])
        while len(self.results[row_idx]) <= col_idx:
            self.results[row_idx].append(ctk.StringVar(value=""))
        
        while len(self._full_results) <= row_idx:
            self._full_results.append([])
        while len(self._full_results[row_idx]) <= col_idx:
            self._full_results[row_idx].append("")

        cell_frame = ctk.CTkFrame(self.scrollable_content_frame, border_width=1, border_color=styles.MATRIX_CELL_BORDER_COLOR, width=styles.MATRIX_CELL_WIDTH, height=styles.MATRIX_RESULT_CELL_HEIGHT)
        cell_frame.grid(row=row_idx + 1, column=col_idx + 1, padx=5, pady=5, sticky="nsew")
        cell_frame.grid_propagate(False)
        cell_frame.grid_columnconfigure(1, weight=1)

        checkbox = ctk.CTkCheckBox(
            cell_frame,
            text="",
            variable=self.checkbox_states[row_idx][col_idx],
            width=15,
            command=lambda r=row_idx, c=col_idx: self._on_cell_checkbox_toggled(r, c)
        )
        checkbox.grid(row=0, column=0, padx=0, pady=2, sticky="w")

        inner_h = max(20, styles.MATRIX_RESULT_CELL_HEIGHT - 6)
        result_textbox = ctk.CTkTextbox(cell_frame, wrap="word", height=inner_h, font=styles.MATRIX_RESULT_FONT, fg_color=styles.HISTORY_ITEM_FG_COLOR, text_color=styles.HISTORY_ITEM_TEXT_COLOR)
        setup_textbox_right_click_menu(result_textbox)
        result_textbox.grid(row=0, column=1, padx=(0,2), pady=2, sticky="nsew")
        result_textbox.insert("1.0", self.results[row_idx][col_idx].get())
        result_textbox.configure(state="disabled")
        result_textbox.bind("<Button-1>", lambda event, r=row_idx, c=col_idx: self._show_full_result_popup(r, c))
        self.results[row_idx][col_idx].trace_add("write", lambda *args, sv=self.results[row_idx][col_idx], tb=result_textbox: self._update_textbox_from_stringvar(sv, tb))

        # Track textbox reference and style matrix
        while len(self._result_textboxes) <= row_idx:
            self._result_textboxes.append([])
        while len(self._result_textboxes[row_idx]) <= col_idx:
            self._result_textboxes[row_idx].append(None)
        self._result_textboxes[row_idx][col_idx] = result_textbox

        while len(self._cell_style) <= row_idx:
            self._cell_style.append([])
        while len(self._cell_style[row_idx]) <= col_idx:
            self._cell_style[row_idx].append("normal")

    def _on_cell_checkbox_toggled(self, r_idx: int, c_idx: int) -> None:
        """Persist checkbox state for a cell into the DB (matrix_cell)."""
        try:
            self._ensure_db_session_tab()
            if self._db_tab_id is None:
                return
            # Ensure related input/prompt ids
            input_id = self._ensure_db_input_id(r_idx)
            prompt_key = list(self.prompts.keys())[c_idx] if c_idx < len(self.prompts) else None
            prompt_cfg = self.prompts.get(prompt_key) if prompt_key else None
            prompt_id = self._ensure_db_prompt_id(c_idx, prompt_cfg) if prompt_cfg else None
            if not input_id or not prompt_id:
                return
            checked = False
            if 0 <= r_idx < len(self.checkbox_states) and 0 <= c_idx < len(self.checkbox_states[r_idx]):
                checked = bool(self.checkbox_states[r_idx][c_idx].get())
            db.upsert_matrix_cell(self._db_tab_id, input_id, prompt_id, checked)
        except Exception:
            # Non-fatal; ignore persistence errors here
            pass

    def _ensure_checkbox_matrix(self) -> None:
        """Ensure checkbox_states is sized to match current inputs x prompts."""
        try:
            num_rows = len(self.input_data)
            num_cols = len(self.prompts)
            while len(self.checkbox_states) < num_rows:
                self.checkbox_states.append([])
            for r in range(num_rows):
                while len(self.checkbox_states[r]) < num_cols:
                    self.checkbox_states[r].append(ctk.BooleanVar(value=False))
        except Exception:
            pass

    def _add_input_row(self):
        # Ensure any visible tooltip is hidden before modifying layout
        try:
            self._hide_tooltip()
        except Exception:
            pass
        try:
            self.configure(cursor='watch')
            self.update_idletasks()
        except Exception:
            pass

        new_row_idx = len(self.input_data)
        self.input_data.append({"type": "text", "data": ""})
        if self._row_summaries:
            self._row_summaries.append(ctk.StringVar(value=""))

        self._add_input_row_widgets(new_row_idx, self.input_data[new_row_idx])
        
        self.after(10, lambda: self.canvas.configure(scrollregion=self.canvas.bbox("all")))

        try:
            self.configure(cursor='')
        except Exception:
            pass

    def _add_prompt_column(self):
        # Ensure any visible tooltip is hidden before modifying layout
        try:
            self._hide_tooltip()
        except Exception:
            pass
        try:
            self.configure(cursor='watch')
            self.update_idletasks()
        except Exception:
            pass

        new_col_idx = len(self.prompts)
        new_prompt_id = f"prompt_{new_col_idx + 1}"
        new_prompt_name = tr("prompt.default_name_fmt", n=new_col_idx + 1)
        new_prompt_config = Prompt(name=new_prompt_name, model="gemini-2.5-flash", system_prompt=tr("prompt.new_placeholder"))
        
        self.prompts[new_prompt_id] = new_prompt_config
        try:
            # update active tab prompt objects snapshot
            self._tabs[self._active_tab_index]['prompts_obj'] = {pid: (p.model_copy(deep=True) if hasattr(p, 'model_copy') else Prompt(**p.model_dump())) for pid, p in self.prompts.items()}
        except Exception:
            pass
        self._column_widths.append(styles.MATRIX_CELL_WIDTH)

        if self._col_summaries:
            self._col_summaries.append(ctk.StringVar(value=""))

        self._add_prompt_header_widgets(new_col_idx, new_prompt_id, new_prompt_config)
        for row_idx in range(len(self.input_data)):
            self._create_result_cell(row_idx, new_col_idx)

        self.after(10, lambda: self.canvas.configure(scrollregion=self.canvas.bbox("all")))

        try:
            self.configure(cursor='')
        except Exception:
            pass

    def _clear_all(self):
        if not messagebox.askyesno(tr("matrix.clear_confirm_title"), tr("matrix.clear_confirm_message")):
            return

        self.input_data = [{"type": "text", "data": ""}]
        self.prompts = {}
        self.checkbox_states = []
        self.results = []
        self._full_results = []
        self._row_summaries = []
        self._col_summaries = []
        self._row_heights = []
        self._column_widths = []
        self.total_tasks = 0
        self.completed_tasks = 0

        if self.summarize_row_button:
            self.summarize_row_button.configure(state="disabled")
        if self.summarize_col_button:
            self.summarize_col_button.configure(state="disabled")
        if self.summarize_matrix_button:
            self.summarize_matrix_button.configure(state="disabled")
        if self.export_excel_button:
            self.export_excel_button.configure(state="disabled")

        try:
            self.progress_label.configure(text=tr("matrix.progress_fmt", done=0, total=0))
        except Exception:
            pass

        self._update_ui()

    def _run_batch_processing(self):
        # Ensure checkbox matrix is sized to current grid
        try:
            self._ensure_checkbox_matrix()
        except Exception:
            pass
        checked_tasks = []
        for r_idx, row_input in enumerate(self.input_data):
            for c_idx, prompt_id in enumerate(self.prompts.keys()):
                if self.checkbox_states[r_idx][c_idx].get():
                    checked_tasks.append((r_idx, c_idx, row_input, prompt_id))
        
        if not checked_tasks:
            messagebox.showinfo(tr("matrix.run_title"), tr("matrix.no_checked_combinations"))
            return

        self.total_tasks = len(checked_tasks)
        self.completed_tasks = 0
        self._update_progress_label()

        num_inputs = len(self.input_data)
        num_prompts = len(self.prompts)

        while len(self.results) < num_inputs:
            self.results.append([])
        for r_idx in range(num_inputs):
            while len(self.results[r_idx]) < num_prompts:
                self.results[r_idx].append(ctk.StringVar(value=""))

        while len(self._full_results) < num_inputs:
            self._full_results.append([])
        for r_idx in range(num_inputs):
            while len(self._full_results[r_idx]) < num_prompts:
                self._full_results[r_idx].append("")

        for r_idx, c_idx, _, _ in checked_tasks:
            self.results[r_idx][c_idx].set(tr("common.processing"))
            # Normal run uses default color
            self._set_cell_style(r_idx, c_idx, "normal")

        asyncio.run_coroutine_threadsafe(self._execute_llm_tasks(checked_tasks), self.worker_loop)

    def _set_cell_style(self, r_idx: int, c_idx: int, style: str):
        try:
            while len(self._cell_style) <= r_idx:
                self._cell_style.append([])
            while len(self._cell_style[r_idx]) <= c_idx:
                self._cell_style[r_idx].append("normal")
            self._cell_style[r_idx][c_idx] = style
            tb = None
            if 0 <= r_idx < len(self._result_textboxes) and 0 <= c_idx < len(self._result_textboxes[r_idx]):
                tb = self._result_textboxes[r_idx][c_idx]
            if tb and tb.winfo_exists():
                tb.configure(state="normal")
                if style == "flow":
                    tb.configure(text_color=styles.FLOW_RESULT_TEXT_COLOR)
                else:
                    tb.configure(text_color=styles.HISTORY_ITEM_TEXT_COLOR)
                tb.configure(state="disabled")
        except Exception:
            pass

    def _update_cell_on_main_thread(self, r_idx: int, c_idx: int, text_content: str, is_final: bool = False):
        if self._is_closing or not self.winfo_exists():
            return
        current_text = ""
        if 0 <= r_idx < len(self.results) and 0 <= c_idx < len(self.results[r_idx]):
            current_text = self.results[r_idx][c_idx].get()
        else:
            return

        new_text = text_content if is_final else current_text + text_content
        
        try:
            self.results[r_idx][c_idx].set(new_text)
        except tk.TclError:
            pass

        if is_final:
            if 0 <= r_idx < len(self._full_results) and 0 <= c_idx < len(self._full_results[r_idx]):
                self._full_results[r_idx][c_idx] = new_text
            else:
                print(f"ERROR: _update_cell_on_main_thread - _full_results のインデックス ({r_idx}, {c_idx}) が範囲外です。最終結果の保存をスキップします。")
            
            # Persist matrix result for flow or batch (if not already saved in async path)
            try:
                input_id = self._ensure_db_input_id(r_idx)
                # derive prompt config by column index
                prompt_key = list(self.prompts.keys())[c_idx] if c_idx < len(self.prompts) else None
                prompt_cfg = self.prompts.get(prompt_key) if prompt_key else None
                prompt_id = self._ensure_db_prompt_id(c_idx, prompt_cfg) if prompt_cfg else None
                self._ensure_db_session_tab()
                if input_id and prompt_id and self._db_tab_id is not None:
                    db.add_matrix_result(self._db_tab_id, input_id, prompt_id, run_id=None, output_id=None, final_text=new_text)
            except Exception:
                pass

            with self.progress_lock:
                self.completed_tasks += 1
                self._update_progress_label()

    async def _execute_llm_tasks(self, tasks_to_run: List[tuple]):
        self.processing_tasks = []
        for r_idx, c_idx, row_input, prompt_id in tasks_to_run:
            prompt_config = self.prompts.get(prompt_id)
            if prompt_config:
                task = asyncio.create_task(self._process_single_cell(r_idx, c_idx, row_input, prompt_config))
                self.processing_tasks.append(task)
            else:
                print(f"ERROR: _execute_llm_tasks - prompt_id '{prompt_id}' not found.")
                error_msg = tr("matrix.error_no_prompt_config")
                self.after(0, self._update_cell_on_main_thread, r_idx, c_idx, error_msg, True)

        await asyncio.gather(*self.processing_tasks)
        
        def show_completion_notification():
            if self.summarize_row_button:
                self.summarize_row_button.configure(state="normal")
            if self.summarize_col_button:
                self.summarize_col_button.configure(state="normal")
            if self.export_excel_button:
                self.export_excel_button.configure(state="normal")
            if self.summarize_matrix_button:
                self.summarize_matrix_button.configure(state="normal")
            for r_idx, c_idx, _, _ in tasks_to_run:
                if 0 <= r_idx < len(self.checkbox_states) and 0 <= c_idx < len(self.checkbox_states[r_idx]):
                    self.checkbox_states[r_idx][c_idx].set(False)
        
        self.after(0, show_completion_notification)

    def _confirm_flow(self, plans: Dict[int, List[int]]) -> bool:
        # Build message: steps per row and warnings
        lines = []
        total_steps = 0
        for r_idx, cols in plans.items():
            if not cols:
                continue
            letters = [chr(ord('A') + c) for c in cols]
            flow_str = " → ".join(letters)
            lines.append(f"{tr('action.input').rstrip(':')}{r_idx+1}: {len(cols)} {tr('matrix.flow.steps_label')} ({flow_str})")
            total_steps += len(cols)
        if total_steps == 0:
            messagebox.showinfo(tr("matrix.flow.running_title"), tr("matrix.no_checked_combinations"))
            return False
        # Overwrite notice
        overwrite = False
        for r_idx, cols in plans.items():
            for c_idx in cols:
                if 0 <= r_idx < len(self._full_results) and 0 <= c_idx < len(self._full_results[r_idx]):
                    if self._full_results[r_idx][c_idx]:
                        overwrite = True
                        break
            if overwrite:
                break
        msg = "\n".join(lines)
        if overwrite:
            msg += f"\n\n{tr('matrix.flow.overwrite_note')}"
        msg += f"\n\n{tr('matrix.flow.max_steps_label')}: {self.max_flow_steps}"
        res = messagebox.askokcancel(tr("matrix.flow.confirm_title"), msg)
        return bool(res)

    def _run_flow_processing(self):
        # Build per-row plans of selected columns, limited and sorted by column index (A..)
        try:
            self._ensure_checkbox_matrix()
        except Exception:
            pass
        plans: Dict[int, List[int]] = {}
        for r_idx, row_input in enumerate(self.input_data):
            sel_cols = [c_idx for c_idx, _ in enumerate(self.prompts.keys()) if self.checkbox_states[r_idx][c_idx].get()]
            sel_cols.sort()
            if sel_cols:
                try:
                    self.max_flow_steps = int(getattr(self.agent.config, 'max_flow_steps', self.max_flow_steps))
                except Exception:
                    pass
                plans[r_idx] = sel_cols[: int(self.max_flow_steps)]

        if not self._confirm_flow(plans):
            return

        # Initialize UI states
        total_steps = sum(len(cols) for cols in plans.values())
        self.total_tasks = total_steps
        self.completed_tasks = 0
        self._update_progress_label()

        # Mark target cells as processing and flow-styled
        for r_idx, cols in plans.items():
            for c_idx in cols:
                self.results[r_idx][c_idx].set(tr("common.processing"))
                self._set_cell_style(r_idx, c_idx, "flow")

        # Launch per-row flows concurrently
        self._flow_cancel_requested = False
        try:
            self.flow_run_button.configure(state="disabled")
        except Exception:
            pass
        self._show_flow_progress_dialog()
        asyncio.run_coroutine_threadsafe(self._execute_flow_tasks(plans), self.worker_loop)

    async def _execute_flow_for_row(self, r_idx: int, cols: List[int]):
        # Conversation history as alternating user/model messages (dicts)
        conv: List[Dict[str, Any]] = []
        # Prepare initial user parts
        input_item = self.input_data[r_idx]
        initial_parts: List[Any] = []
        if input_item["type"] == "text":
            initial_parts = [{"text": input_item["data"]}]
        elif input_item["type"] in ("image", "image_compressed"):
            img_b64 = input_item["data"]
            if input_item["type"] == "image_compressed":
                try:
                    import zlib
                    img_b64 = base64.b64encode(zlib.decompress(base64.b64decode(img_b64))).decode("utf-8")
                except Exception:
                    pass
            initial_parts = [create_image_part(img_b64)]
        elif input_item["type"] == "file":
            file_path = input_item["data"]
            try:
                mime_type, _ = mimetypes.guess_type(file_path)
                if not mime_type:
                    mime_type = "application/octet-stream"
                uploaded_file = await asyncio.to_thread(genai.upload_file, path=file_path, mime_type=mime_type)
                initial_parts = [uploaded_file]
            except Exception as e:
                err = tr("matrix.error_prefix") + tr("notify.file_upload_failed", details=str(e))
                self.after(0, self._update_cell_on_main_thread, r_idx, cols[0], err, True)
                return
        else:
            initial_parts = [{"text": ""}]

        current_parts = initial_parts
        # Sequentially run per selected column
        for step_idx, c_idx in enumerate(cols):
            if self._flow_cancel_requested:
                break
            # Build user message for current step: prepend instruction as prefix (text) or as separate part
            prompt_id = list(self.prompts.keys())[c_idx]
            prompt_config = self.prompts[prompt_id]
            combined_parts: List[Any] = []
            try:
                # If current parts are purely text, combine into one text blob with instruction prefix
                if all(isinstance(p, dict) and 'text' in p for p in current_parts):
                    joined = "\n\n".join(str(p.get('text', '')) for p in current_parts)
                    prefix = str(getattr(prompt_config, 'system_prompt', '') or '')
                    combined_text = f"{prefix}\n\n---\n\n{joined}" if prefix else joined
                    combined_parts = [{"text": combined_text}]
                else:
                    # Non-text (image/file) inputs: include instruction as separate text part before input
                    instr = str(getattr(prompt_config, 'system_prompt', '') or '')
                    if instr:
                        combined_parts.append({"text": instr})
                    combined_parts.extend(list(current_parts))
            except Exception:
                combined_parts = list(current_parts) if current_parts else [{"text": ""}]

            try:
                conv.append({"role": "user", "parts": combined_parts})
            except Exception:
                conv.append({"role": "user", "parts": [{"text": str(combined_parts)}]})
            try:
                async with self.semaphore:
                    gemini_model = GenerativeModel(prompt_config.model, system_instruction=prompt_config.system_prompt)
                    # Detect tools setting
                    has_url_text = any(isinstance(p, dict) and "text" in p and isinstance(p["text"], str) and p["text"].strip().startswith(("http://", "https://")) for p in combined_parts)
                    tools_list = [{"google_search": {}}] if getattr(prompt_config, 'enable_web', False) or has_url_text else None
                    generate_content_config = types.GenerationConfig(
                        temperature=prompt_config.parameters.temperature,
                        top_p=prompt_config.parameters.top_p,
                        top_k=prompt_config.parameters.top_k,
                        max_output_tokens=prompt_config.parameters.max_output_tokens,
                        stop_sequences=prompt_config.parameters.stop_sequences
                    )

                    def _gen_sync(config, contents, tools=None):
                        if tools is not None:
                            return gemini_model.generate_content(contents=contents, generation_config=config, tools=tools)
                        return gemini_model.generate_content(contents=contents, generation_config=config)

                    # Call model with conversation so far; fallback on tool errors
                    try:
                        response = await asyncio.to_thread(_gen_sync, generate_content_config, conv, tools_list)
                    except Exception:
                        if tools_list:
                            try:
                                alt_tools = [{"google_search_retrieval": {}}]
                                response = await asyncio.to_thread(_gen_sync, generate_content_config, conv, alt_tools)
                            except Exception:
                                response = await asyncio.to_thread(_gen_sync, generate_content_config, conv, None)
                        else:
                            raise

                    # Extract text
                    def _extract_text(resp) -> str:
                        try:
                            if getattr(resp, 'candidates', None):
                                cand = resp.candidates[0]
                                content = getattr(cand, 'content', None)
                                parts = getattr(content, 'parts', None) if content else None
                                if parts:
                                    return "".join(p.text for p in parts if hasattr(p, 'text'))
                            return getattr(resp, 'text', '') or ''
                        except Exception:
                            return ''
                    out_text = _extract_text(response)
                    if not out_text:
                        out_text = tr("matrix.response_empty")
            except Exception as e:
                out_text = tr("matrix.error_prefix") + str(e)

            # Update cell with result and style, and uncheck the box
            self.after(0, self._update_cell_on_main_thread, r_idx, c_idx, out_text, True)
            self._set_cell_style(r_idx, c_idx, "flow")
            try:
                self.after(0, lambda rr=r_idx, cc=c_idx: self.checkbox_states[rr][cc].set(False))
            except Exception:
                pass
            # Append model message and set next input as its text
            try:
                conv.append({"role": "model", "parts": [{"text": out_text}]})
            except Exception:
                conv.append({"role": "model", "parts": [{"text": str(out_text)}]})
            current_parts = [{"text": out_text}]

        # After row flow, uncheck relevant checkboxes
        def _clear_checks():
            for c_idx in cols:
                try:
                    self.checkbox_states[r_idx][c_idx].set(False)
                except Exception:
                    pass
        self.after(0, _clear_checks)

        # Enable summary/export buttons after flows (all tasks completion is handled globally too)
    async def _execute_flow_tasks(self, plans: Dict[int, List[int]]):
        self._flow_tasks = []
        for r_idx, cols in plans.items():
            task = asyncio.create_task(self._execute_flow_for_row(r_idx, cols))
            self._flow_tasks.append(task)
        try:
            await asyncio.gather(*self._flow_tasks)
        except asyncio.CancelledError:
            pass
        def _enable_actions():
            try:
                if self.summarize_row_button:
                    self.summarize_row_button.configure(state="normal")
                if self.summarize_col_button:
                    self.summarize_col_button.configure(state="normal")
                if self.summarize_matrix_button:
                    self.summarize_matrix_button.configure(state="normal")
                if self.export_excel_button:
                    self.export_excel_button.configure(state="normal")
                self.flow_run_button.configure(state="normal")
                self._close_flow_progress_dialog()
            except Exception:
                pass
        self.after(0, _enable_actions)

    async def _process_single_cell(self, r_idx: int, c_idx: int, input_item: Dict[str, Any], prompt_config: Prompt):
        full_result = ""
        try:
            async with self.semaphore:
                gemini_model = GenerativeModel(prompt_config.model, system_instruction=prompt_config.system_prompt)
                contents_to_send = []

                if input_item["type"] == "text":
                    contents_to_send.append(input_item["data"])
                elif input_item["type"] in ("image", "image_compressed"):
                    img_b64 = input_item["data"]
                    if input_item["type"] == "image_compressed":
                        try:
                            import zlib
                            decompressed = zlib.decompress(base64.b64decode(img_b64))
                            img_b64 = base64.b64encode(decompressed).decode('utf-8')
                        except Exception:
                            pass
                    image_part = create_image_part(img_b64)
                    contents_to_send.append(image_part)
                elif input_item["type"] == "file":
                    file_path = input_item["data"]
                    try:
                        mime_type, _ = mimetypes.guess_type(file_path)
                        if not mime_type:
                            mime_type = "application/octet-stream"
                        uploaded_file = await asyncio.to_thread(genai.upload_file, path=file_path, mime_type=mime_type)
                        contents_to_send.append(uploaded_file)
                    except Exception as e:
                        raise RuntimeError(tr("notify.file_upload_failed", details=str(e)))
                else:
                    raise ValueError(f"Unsupported input type: {input_item['type']}")

                has_url_text = any(isinstance(p, str) and p.strip().startswith(("http://", "https://")) for p in contents_to_send)
                tools_list = [{"google_search": {}}] if getattr(prompt_config, 'enable_web', False) or has_url_text else None

                try:
                    generate_content_config = types.GenerationConfig(temperature=prompt_config.parameters.temperature, top_p=prompt_config.parameters.top_p, top_k=prompt_config.parameters.top_k, max_output_tokens=prompt_config.parameters.max_output_tokens, stop_sequences=prompt_config.parameters.stop_sequences, tools=tools_list)
                except TypeError:
                    generate_content_config = types.GenerationConfig(temperature=prompt_config.parameters.temperature, top_p=prompt_config.parameters.top_p, top_k=prompt_config.parameters.top_k, max_output_tokens=prompt_config.parameters.max_output_tokens, stop_sequences=prompt_config.parameters.stop_sequences)

                def _gen_sync(config):
                    return gemini_model.generate_content(contents=contents_to_send, generation_config=config)

                # Create DB run for this cell
                run_id = None
                try:
                    run_id = db.create_run('matrix', prompt_config.model, {
                        'temperature': prompt_config.parameters.temperature,
                        'top_p': prompt_config.parameters.top_p,
                        'top_k': prompt_config.parameters.top_k,
                        'max_output_tokens': prompt_config.parameters.max_output_tokens,
                        'stop_sequences': prompt_config.parameters.stop_sequences,
                    }, {
                        'name': prompt_config.name,
                        'model': prompt_config.model,
                        'system_prompt': prompt_config.system_prompt,
                    })
                    # Save inputs as run_input
                    try:
                        cid = None
                        if input_item['type'] == 'text':
                            cid = db.save_clipboard_item(input_item['data'], source='matrix_input')
                        elif input_item['type'] in ('image','image_compressed'):
                            b64 = input_item['data']
                            if input_item['type'] == 'image_compressed':
                                try:
                                    import zlib, base64 as _b64
                                    raw = zlib.decompress(_b64.b64decode(b64))
                                    b64 = _b64.b64encode(raw).decode('utf-8')
                                except Exception:
                                    pass
                            cid = db.save_clipboard_item({'type':'image','data':b64}, source='matrix_input')
                        elif input_item['type'] == 'file':
                            cid = db.save_clipboard_item({'type':'file','data':input_item['data']}, source='matrix_input')
                        db.add_run_input(run_id, cid)
                    except Exception:
                        pass
                except Exception:
                    run_id = None

                try:
                    response = await asyncio.to_thread(_gen_sync, generate_content_config)
                except Exception:
                    if tools_list:
                        try:
                            alt_tools = [{"google_search_retrieval": {}}]
                            alt_config = types.GenerationConfig(temperature=generate_content_config.temperature, top_p=generate_content_config.top_p, top_k=generate_content_config.top_k, max_output_tokens=generate_content_config.max_output_tokens, stop_sequences=generate_content_config.stop_sequences, tools=alt_tools)
                            response = await asyncio.to_thread(_gen_sync, alt_config)
                        except Exception:
                            no_tool_config = types.GenerationConfig(temperature=generate_content_config.temperature, top_p=generate_content_config.top_p, top_k=generate_content_config.top_k, max_output_tokens=generate_content_config.max_output_tokens, stop_sequences=generate_content_config.stop_sequences)
                            response = await asyncio.to_thread(_gen_sync, no_tool_config)
                    else:
                        raise
                
                def _extract_text(resp) -> str:
                    try:
                        if getattr(resp, 'candidates', None):
                            cand = resp.candidates[0]
                            content = getattr(cand, 'content', None)
                            parts = getattr(content, 'parts', None) if content else None
                            if parts:
                                return "".join(p.text for p in parts if hasattr(p, 'text'))
                        return getattr(resp, 'text', '') or ''
                    except Exception:
                        return ''

                if response.prompt_feedback and response.prompt_feedback.block_reason:
                    full_result = tr("safety.request_blocked_message")
                    self.after(0, lambda: self.notification_callback(tr("safety.request_blocked_title"), full_result, level="error"))
                elif not response.candidates:
                    full_result = tr("matrix.no_response")
                    self.after(0, lambda: self.notification_callback(tr("common.info"), full_result, level="error"))
                else:
                    extracted = _extract_text(response)
                    full_result = extracted if extracted else (tr("matrix.response_empty") + f" finish_reason={getattr(response.candidates[0], 'finish_reason', None)}")

                # Save to DB: run_output + matrix_result
                try:
                    input_id = self._ensure_db_input_id(r_idx)
                    prompt_id = self._ensure_db_prompt_id(c_idx, prompt_config)
                    out_id = None
                    if run_id is not None:
                        out_id = db.add_run_output(run_id, prompt_id=None, input_id=None, content_text=full_result)
                        db.finish_run(run_id, status='success')
                    if input_id and prompt_id:
                        self._ensure_db_session_tab()
                        if self._db_tab_id is not None:
                            db.add_matrix_result(self._db_tab_id, input_id, prompt_id, run_id, out_id, full_result)
                except Exception:
                    pass

        except Exception as e:
            full_result = tr("matrix.error_prefix") + str(e)
            self.after(0, lambda err=e: self.notification_callback(tr("matrix.processing_error_title"), tr("matrix.cell_error_fmt", row=r_idx+1, col=c_idx+1, details=str(err)), "error"))
            traceback.print_exc()
        finally:
            self.after(0, self._update_cell_on_main_thread, r_idx, c_idx, full_result, True)

    async def _summarize_content_with_llm(self, content_list: List[str], summary_type: str, r_idx: Optional[int] = None, c_idx: Optional[int] = None) -> str:
        combined_content = "\n\n".join(content_list)
        summary_prompt_text = f"以下の{summary_type}の情報を要約してください。重要なポイントを簡潔にまとめてください。\n\n{combined_content}"
        
        cfg_prompt: Optional[Prompt] = None
        try:
            if r_idx is not None:
                cfg_prompt = getattr(self.agent.config, 'matrix_row_summary_prompt', None)
            elif c_idx is not None:
                cfg_prompt = getattr(self.agent.config, 'matrix_col_summary_prompt', None)
            else:
                cfg_prompt = getattr(self.agent.config, 'matrix_matrix_summary_prompt', None)
        except Exception:
            cfg_prompt = None
        
        summary_prompt_config = cfg_prompt or Prompt(name=f"{summary_type}要約", model="gemini-2.5-flash-lite", system_prompt="与えられた情報を簡潔に要約してください。" )
        
        full_summary_result = ""
        try:
            generation_config = genai.types.GenerationConfig(temperature=summary_prompt_config.parameters.temperature, top_p=summary_prompt_config.parameters.top_p, top_k=summary_prompt_config.parameters.top_k, max_output_tokens=summary_prompt_config.parameters.max_output_tokens, stop_sequences=summary_prompt_config.parameters.stop_sequences)
            gemini_model = GenerativeModel(summary_prompt_config.model, system_instruction=summary_prompt_config.system_prompt)
            response = await asyncio.to_thread(gemini_model.generate_content, contents=[summary_prompt_text], generation_config=generation_config)
            
            def _extract_text(resp) -> str:
                try:
                    if getattr(resp, 'candidates', None):
                        cand = resp.candidates[0]
                        content = getattr(cand, 'content', None)
                        parts = getattr(content, 'parts', None) if content else None
                        if parts:
                            return "".join(p.text for p in parts if hasattr(p, 'text'))
                    return getattr(resp, 'text', '') or ''
                except Exception:
                    return ''

            if response.prompt_feedback and response.prompt_feedback.block_reason:
                full_summary_result = tr("safety.request_blocked_message")
                self.after(0, lambda: self.notification_callback(tr("safety.request_blocked_title"), full_summary_result, level="error"))
            elif not response.candidates:
                full_summary_result = tr("matrix.final_summary.none")
                self.after(0, lambda: self.notification_callback(tr("common.info"), full_summary_result, level="error"))
            else:
                extracted = _extract_text(response)
                full_summary_result = extracted if extracted else tr("matrix.response_empty")
            
        except Exception as e:
            full_summary_result = tr("matrix.final_summary.error_fmt", details=str(e))
            self.after(0, lambda err=e: self.notification_callback(tr("matrix.final_summary.error_title"), tr("matrix.final_summary.error_fmt", details=str(err)), "error"))
            traceback.print_exc()
        
        return full_summary_result

    async def _summarize_rows_async(self):

        # initialize row summaries
        self._row_summaries = [ctk.StringVar(value=tr("common.processing")) for _ in range(len(self.input_data))]
        self.after(0, self._update_row_summary_column)

        summary_tasks = []
        for r_idx in range(len(self.input_data)):
            row_results = [self._full_results[r_idx][c_idx] for c_idx in range(len(self.prompts))]
            valid_results = [res for res in row_results if res and res != tr("common.processing") and not res.startswith(tr("matrix.error_prefix").strip())]
            
            if valid_results:
                task = asyncio.create_task(self._summarize_content_with_llm(valid_results, f"{tr('matrix.row_summary_header')} {r_idx+1}", r_idx=r_idx))
                summary_tasks.append((r_idx, task))
            else:
                # Update on UI thread to avoid Tk threading issues
                self.after(0, lambda r=r_idx: self._row_summaries[r].set(tr("matrix.summary.target_none")))

        results = await asyncio.gather(*[task for _, task in summary_tasks])

        for i, (r_idx, _) in enumerate(summary_tasks):
            text = results[i]
            self.after(0, lambda r=r_idx, s=text: self._row_summaries[r].set(s))
            # Save to DB
            try:
                input_id = self._ensure_db_input_id(r_idx)
                self._ensure_db_session_tab()
                if input_id and self._db_tab_id is not None:
                    db.add_matrix_row_summary(self._db_tab_id, input_id, text)
            except Exception:
                pass
        
        self.after(0, self._update_row_summary_column)
        # try:
        #     CTkMessagebox(title=tr("matrix.row_summary_header"), message=tr("matrix.summary.row_done"), icon="info").wait_window()
        # except Exception:
        #     pass

    async def _summarize_columns_async(self):

        self._col_summaries = [ctk.StringVar(value=tr("common.processing")) for _ in range(len(self.prompts))]
        self.after(0, self._update_column_summary_row)

        summary_tasks = []
        for c_idx in range(len(self.prompts)):
            col_results = [self._full_results[r_idx][c_idx] for r_idx in range(len(self.input_data))]
            valid_results = [res for res in col_results if res and res != tr("common.processing") and not res.startswith(tr("matrix.error_prefix").strip())]

            if valid_results:
                task = asyncio.create_task(self._summarize_content_with_llm(valid_results, f"{tr('matrix.col_summary_header')} {chr(ord('A') + c_idx)}", c_idx=c_idx))
                summary_tasks.append((c_idx, task))
            else:
                self.after(0, lambda c=c_idx: self._col_summaries[c].set(tr("matrix.summary.target_none")))

        results = await asyncio.gather(*[task for _, task in summary_tasks])

        for i, (c_idx, _) in enumerate(summary_tasks):
            text = results[i]
            self.after(0, lambda c=c_idx, s=text: self._col_summaries[c].set(s))
            # Save to DB
            try:
                # Map prompt id
                prompt_key = list(self.prompts.keys())[c_idx] if c_idx < len(self.prompts) else None
                prompt_cfg = self.prompts.get(prompt_key) if prompt_key else None
                pid = self._ensure_db_prompt_id(c_idx, prompt_cfg) if prompt_cfg else None
                self._ensure_db_session_tab()
                if pid and self._db_tab_id is not None:
                    db.add_matrix_col_summary(self._db_tab_id, pid, text)
            except Exception:
                pass
        
        self.after(0, self._update_column_summary_row)
        # try:
        #     CTkMessagebox(title=tr("matrix.col_summary_header"), message=tr("matrix.summary.col_done"), icon="info").wait_window()
        # except Exception:
        #     pass

    def _summarize_rows(self):
        asyncio.run_coroutine_threadsafe(self._summarize_rows_async(), self.worker_loop)

    def _summarize_columns(self):
        asyncio.run_coroutine_threadsafe(self._summarize_columns_async(), self.worker_loop)

    def _summarize_matrix(self):
        asyncio.run_coroutine_threadsafe(self._summarize_matrix_async(), self.worker_loop)

    async def _summarize_matrix_async(self):
        if not self._row_summaries or any(s.get() in ["", tr("common.processing")] for s in self._row_summaries):
            await self._summarize_rows_async()
            await asyncio.sleep(0.1)

        if not self._col_summaries or any(s.get() in ["", tr("common.processing")] for s in self._col_summaries):
            await self._summarize_columns_async()
            await asyncio.sleep(0.1)

        # まとめテキスト生成（エラーのみ除外。対象なしでもテキストとして許容）
        # UIスレッド反映のタイミング差を避けるため、可能ならDBから直近のサマリーを取得
        row_summary_texts: List[str] = []
        col_summary_texts: List[str] = []
        try:
            self._ensure_db_session_tab()
            if self._db_tab_id is not None:
                rs = db.get_matrix_row_summaries(self._db_tab_id)
                if rs:
                    # rs: List[(row_index, text)] → ビュー用に整形
                    for r_idx, text in rs:
                        if text and "エラー" not in text:
                            row_summary_texts.append(f"【行 {r_idx+1} のまとめ】\n{text}")
                cs = db.get_matrix_col_summaries(self._db_tab_id)
                if cs:
                    for c_idx, text in cs:
                        if text and "エラー" not in text:
                            col_summary_texts.append(f"【列 {chr(ord('A') + c_idx)} のまとめ】\n{text}")
        except Exception:
            # DB取得に失敗した場合はUI変数から取得
            pass
        # DBから取得できない/空の場合はUI変数から再構築
        if not row_summary_texts:
            row_summary_texts = [f"【行 {i+1} のまとめ】\n{s.get()}" for i, s in enumerate(self._row_summaries) if s.get() and "エラー" not in s.get()]
        if not col_summary_texts:
            col_summary_texts = [f"【列 {chr(ord('A') + i)} のまとめ】\n{s.get()}" for i, s in enumerate(self._col_summaries) if s.get() and "エラー" not in s.get()]

        if not row_summary_texts and not col_summary_texts:
            try:
                CTkMessagebox(title=tr("matrix.matrix_summary"), message=tr("matrix.final_summary.none"), icon="warning").wait_window()
            except Exception:
                pass
            self.after(0, lambda: self._update_matrix_summary_cell(""))
            return

        combined_summaries = "\n\n".join(row_summary_texts + col_summary_texts)
        final_summary_prompt = f"以下の各行・各列の要約情報を基に、全体を俯瞰した総合的な結論や洞察を導き出してください。\n\n---\n\n{combined_summaries}"

        final_summary = await self._summarize_content_with_llm([final_summary_prompt], tr("matrix.matrix_summary"))

        if "エラー" not in final_summary:
            pyperclip.copy(final_summary)
            self.after(0, lambda: self._show_final_summary_popup(final_summary))
            self.after(0, lambda: self._update_matrix_summary_cell(final_summary))
            # Save final summary
            try:
                self._ensure_db_session_tab()
                if self._db_tab_id is not None:
                    db.add_matrix_final_summary(self._db_tab_id, final_summary)
            except Exception:
                pass
            try:
                CTkMessagebox(title=tr("matrix.matrix_summary"), message=tr("matrix.final_summary.copied"), icon="info").wait_window()
            except Exception:
                pass
        else:
            try:
                CTkMessagebox(title=tr("matrix.final_summary.error_title"), message=tr("matrix.final_summary.error_fmt", details=final_summary), icon="cancel").wait_window()
            except Exception:
                pass

    def _update_matrix_summary_cell(self, summary_text: str):
        from gemclip.features.matrix.view import update_matrix_summary_cell
        return update_matrix_summary_cell(self, summary_text)

    def _show_final_summary_popup(self, summary_text: str):
        from gemclip.ui.summary_popups import show_final_summary_popup
        return show_final_summary_popup(self, summary_text)

    # moved to utils.truncate_result

    def _update_textbox_from_stringvar(self, string_var: ctk.StringVar, textbox: ctk.CTkTextbox):
        if self._is_closing or not self.winfo_exists():
            return
        try:
            textbox.configure(state="normal")
            textbox.delete("1.0", "end")
            textbox.insert("1.0", string_var.get())
            # Apply style-based text color if tracked
            try:
                # Find indices of this textbox if possible
                for r_idx, row in enumerate(self._result_textboxes):
                    for c_idx, tb in enumerate(row):
                        if tb is textbox:
                            style = None
                            try:
                                style = self._cell_style[r_idx][c_idx]
                            except Exception:
                                style = None
                            if style == "flow":
                                textbox.configure(text_color=styles.FLOW_RESULT_TEXT_COLOR)
                            else:
                                textbox.configure(text_color=styles.HISTORY_ITEM_TEXT_COLOR)
                            raise StopIteration
            except StopIteration:
                pass
            textbox.configure(state="disabled")
        except tk.TclError:
            pass

    def _show_full_result_popup(self, r_idx: int, c_idx: int):
        from gemclip.ui.summary_popups import show_full_result_popup
        return show_full_result_popup(self, r_idx, c_idx)

    def _show_full_row_summary_popup(self, r_idx: int):
        from gemclip.ui.summary_popups import show_full_row_summary_popup
        return show_full_row_summary_popup(self, r_idx)

    def _show_full_col_summary_popup(self, c_idx: int):
        from gemclip.ui.summary_popups import show_full_col_summary_popup
        return show_full_col_summary_popup(self, c_idx)

    def _save_full_result_and_close_popup(self, popup: ctk.CTkToplevel, textbox: ctk.CTkTextbox, r_idx: int, c_idx: int):
        edited_content = textbox.get("1.0", "end-1c")
        self._full_results[r_idx][c_idx] = edited_content
        self.results[r_idx][c_idx].set(truncate_result(edited_content))
        popup.destroy()

    def _save_full_row_summary_and_close_popup(self, popup: ctk.CTkToplevel, textbox: ctk.CTkTextbox, r_idx: int):
        edited = textbox.get("1.0", "end-1c")
        try:
            self._row_summaries[r_idx].set(edited)
        except Exception:
            pass
        popup.destroy()

    def _save_full_col_summary_and_close_popup(self, popup: ctk.CTkToplevel, textbox: ctk.CTkTextbox, c_idx: int):
        edited_content = textbox.get("1.0", "end-1c")
        self._col_summaries[c_idx].set(edited_content)
        popup.destroy()

    def _delete_row(self, row_idx: int):
        if not messagebox.askyesno(tr("matrix.delete_row_title"), f"{tr('matrix.delete_row_confirm_fmt', row=row_idx + 1)}\n{tr('common.cannot_undo')}"):
            return
        try:
            for widget in list(self.scrollable_content_frame.grid_slaves(row=row_idx + 1)):
                widget.destroy()
        except Exception:
            pass
        def finalize_delete():
            try:
                if 0 <= row_idx < len(self.input_data):
                    self.input_data.pop(row_idx)
                    self.checkbox_states.pop(row_idx)
                    self.results.pop(row_idx)
                    self._full_results.pop(row_idx)
                    if self._row_summaries and 0 <= row_idx < len(self._row_summaries):
                        self._row_summaries.pop(row_idx)
                    if self._row_heights and 0 <= row_idx + 1 < len(self._row_heights):
                        self._row_heights.pop(row_idx + 1)
            except Exception:
                pass
            if not self.input_data:
                self._clear_all()
            self._update_ui()
        self.after(10, finalize_delete)

    def _open_prompt_editor(self, prompt_id: str):
        from gemclip.ui import PromptEditorDialog
        
        current_prompt = self.prompts.get(prompt_id)
        if not current_prompt:
            return

        dlg = PromptEditorDialog(self, title=tr("prompt.edit_title_fmt", name=current_prompt.name), prompt=current_prompt)
        result_prompt = dlg.get_result()

        if result_prompt:
            self.prompts[prompt_id] = result_prompt
            try:
                self._tabs[self._active_tab_index]['prompts_obj'] = {pid: (p.model_copy(deep=True) if hasattr(p, 'model_copy') else Prompt(**p.model_dump())) for pid, p in self.prompts.items()}
            except Exception:
                pass
            self._update_prompt_header_display(prompt_id)

    def _update_prompt_header_display(self, prompt_id: str):
        try:
            col_idx = list(self.prompts.keys()).index(prompt_id)
        except ValueError:
            return

        for widget in self.scrollable_content_frame.grid_slaves(row=0, column=col_idx + 1):
            widget.destroy()
        
        prompt_config = self.prompts[prompt_id]
        self._add_prompt_header_widgets(col_idx, prompt_id, prompt_config)

    def _delete_column(self, col_idx: int):
        col_letter = chr(ord('A') + col_idx)
        if not messagebox.askyesno(tr("matrix.delete_col_title"), f"{tr('matrix.delete_col_confirm_fmt', col=col_letter)}\n{tr('common.cannot_undo')}"):
            return
        try:
            for widget in list(self.scrollable_content_frame.grid_slaves(column=col_idx + 1)):
                widget.destroy()
        except Exception:
            pass
        def finalize_delete():
            try:
                prompt_keys = list(self.prompts.keys())
                if 0 <= col_idx < len(prompt_keys):
                    del self.prompts[prompt_keys[col_idx]]
                for r_idx in range(len(self.input_data)):
                    if r_idx < len(self.checkbox_states) and 0 <= col_idx < len(self.checkbox_states[r_idx]):
                        self.checkbox_states[r_idx].pop(col_idx)
                    if r_idx < len(self.results) and 0 <= col_idx < len(self.results[r_idx]):
                        self.results[r_idx].pop(col_idx)
                    if r_idx < len(self._full_results) and 0 <= col_idx < len(self._full_results[r_idx]):
                        self._full_results[r_idx].pop(col_idx)
                if self._col_summaries and 0 <= col_idx < len(self._col_summaries):
                    self._col_summaries.pop(col_idx)
                if self._column_widths and 0 <= col_idx + 1 < len(self._column_widths):
                    self._column_widths.pop(col_idx + 1)
                try:
                    self._tabs[self._active_tab_index]['prompts_obj'] = {pid: (p.model_copy(deep=True) if hasattr(p, 'model_copy') else Prompt(**p.model_dump())) for pid, p in self.prompts.items()}
                except Exception:
                    pass
            except Exception:
                pass
            if not self.prompts:
                self._clear_all()
            self._update_ui()
        self.after(10, finalize_delete)

    def _select_input_source(self, row_idx: int):
        from gemclip.ui.input_source import select_input_source
        return select_input_source(self, row_idx)
    
    def _show_image_preview(self, row_idx: int):
        from gemclip.ui.image_preview import show_image_preview
        return show_image_preview(self, row_idx)

    def _update_progress_label(self):
        if self._is_closing or not self.winfo_exists():
            return
        try:
            self.progress_label.configure(text=tr("matrix.progress_fmt", done=self.completed_tasks, total=self.total_tasks))
        except tk.TclError:
            pass

    def _cancel_flow_processing(self):
        # Signal cancellation and attempt to cancel tasks
        self._flow_cancel_requested = True
        try:
            for t in list(self._flow_tasks):
                try:
                    if not t.done():
                        t.cancel()
                except Exception:
                    pass
            # Update progress dialog state
            try:
                if hasattr(self, '_flow_dialog_label') and self._flow_dialog_label:
                    self._flow_dialog_label.configure(text=tr("matrix.flow.stopping_message"))
            except Exception:
                pass
        except Exception:
            pass

    # --- Flow progress dialog ---
    def _show_flow_progress_dialog(self):
        from gemclip.ui.flow_dialogs import show_flow_progress_dialog
        return show_flow_progress_dialog(self, self._cancel_flow_processing)

    def _close_flow_progress_dialog(self):
        from gemclip.ui.flow_dialogs import close_flow_progress_dialog
        return close_flow_progress_dialog(self)

    def _start_cursor_monitoring(self):
        from gemclip.features.matrix.cursor import start_cursor_monitoring
        return start_cursor_monitoring(self)

    def _update_cursor_direct(self, x, y):
        from gemclip.features.matrix.cursor import update_cursor_direct
        return update_cursor_direct(self, x, y)

    def _export_to_excel(self):
        from gemclip.features.matrix.export import export_to_excel
        return export_to_excel(self)

    def _show_clipboard_history_popup(self, row_idx: int):
        # Fetch recent history from DB (not just in-memory) for selection
        history_for_popup = []
        try:
            try:
                page_limit = int(getattr(self.agent, 'max_history_size', getattr(self.agent.config, 'max_history_size', 20)))
            except Exception:
                page_limit = 20
            rows = db.get_clipboard_items(limit=page_limit, offset=0, q=None)
        except Exception:
            rows = []
        for r in rows:
            try:
                t = r['type']
                if t == 'text':
                    history_for_popup.append({"type": "text", "data": r['text'] or ''})
                elif t == 'image':
                    b = r['image_blob']
                    b64 = base64.b64encode(b).decode('utf-8') if b else ''
                    history_for_popup.append({"type": "image", "data": b64})
                elif t == 'file':
                    history_for_popup.append({"type": "file", "data": r['file_path'] or ''})
            except Exception:
                continue
        def on_select(selected_item: Dict[str, Any]):
            self._set_input_data_from_history(row_idx, selected_item)
        if self._history_popup and self._history_popup.winfo_exists():
            try:
                self._history_popup.destroy()
            except Exception:
                pass
        def _on_popup_destroy():
            self._history_popup = None
        from gemclip.ui.history_selector import ClipboardHistorySelectorPopup as _HistoryPopup
        self._history_popup = _HistoryPopup(parent_app=self, clipboard_history=history_for_popup, on_select_callback=on_select, on_destroy_callback=_on_popup_destroy, page_limit=page_limit, initial_offset=len(rows))
        self._history_popup.show_at_cursor()

    def _open_history_edit_dialog(self, row_idx: int):
        try:
            if not (0 <= row_idx < len(self.input_data)):
                return
            item = self.input_data[row_idx]
            if not isinstance(item, dict) or item.get('type') != 'text':
                return
            dlg = HistoryEditDialog(self.parent_app, initial_value=item.get('data', ''))
            dlg.show()
            new_text = dlg.get_input()
            if new_text is not None:
                self.input_data[row_idx] = {"type": "text", "data": new_text}
                self._set_input_data_from_history(row_idx, self.input_data[row_idx])
        except Exception:
            pass

    def _set_input_data_from_history(self, row_idx: int, selected_item: Dict[str, Any]):
        if not (0 <= row_idx < len(self.input_data)):
            self.notification_callback(tr("common.error"), tr("matrix.invalid_row_index"), "error")
            return
        self.input_data[row_idx] = selected_item
        try:
            self._update_input_row_display(row_idx)
        except Exception as e:
            print(f"ERROR: _set_input_data_from_history - UIの更新に失敗: {e}")
            traceback.print_exc()
            self._update_ui()
