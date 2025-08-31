# agent.py
import asyncio
import base64
import queue
import sys
import threading
import time
import tkinter as tk
from tkinter import filedialog
import customtkinter as ctk
from io import BytesIO
import hashlib
import json
from pathlib import Path
from typing import Dict, Literal, Optional, List, Any, Callable
import traceback # 追加

import keyring
import pyperclip
# winsound is Windows-only; import lazily/optionally
try:
    import winsound  # type: ignore
except Exception:  # pragma: no cover - non-Windows
    winsound = None  # type: ignore
from PIL import Image, ImageGrab
from google.api_core import exceptions
from pystray import Icon, Menu, MenuItem
from google.generativeai import types
import google.generativeai as genai

from gemclip.core import BaseAgent, LlmAgent, create_image_part, Prompt, PromptParameters
from config_manager import load_config, save_config
from gemclip.core import API_SERVICE_ID, APP_NAME, COMPLETION_SOUND_FILE, ICON_FILE
from gemclip.features.matrix import MatrixBatchProcessorWindow
from gemclip.ui import (
    ActionSelectorWindow,
    ResizableInputDialog,
    SettingsWindow,
    NotificationService,
)
from gemclip.infra.hotkeys import WindowsHotkeyManager
from i18n import tr
from gemclip.infra import db  # SQLite storage for history/results

class ClipboardToolAgent(BaseAgent):
    def __init__(self, name: str = "ClipboardToolAgent", description: str = "クリップボード操作とLLM処理を行うエージェント"):
        super().__init__(name, description)
        self.config = load_config()
        if not self.config:
            sys.exit(1)
        
        # API価格情報を読み込む
        self.api_price_info = self._load_api_price_info()

        self.api_key = self._get_api_key()
        # genai.configure を使用してAPIキーを設定
        if self.api_key:
            genai.configure(api_key=self.api_key)

        self.task_queue = queue.Queue()
        self.loop = None
        self.worker_thread = threading.Thread(target=self._async_worker, daemon=True)
        self._worker_running = True
        self._loop_ready_event = threading.Event()
        self.worker_thread.start()
        self._loop_ready_event.wait(timeout=5)
        if not self._loop_ready_event.is_set():
            sys.exit(1)

        self.app: Optional[ctk.CTk] = None
        self.matrix_batch_processor_window: Optional[MatrixBatchProcessorWindow] = None
        self._notification_service: Optional[NotificationService] = None
        self._current_action_selector_window: Optional[ActionSelectorWindow] = None
        self._settings_window: Optional[SettingsWindow] = None

        self.clipboard_history = []
        self.max_history_size = self.config.max_history_size
        self._clipboard_monitor_thread: Optional[threading.Thread] = None
        self._clipboard_monitor_running = False
        self._on_history_updated_callback: Optional[Callable[[List[str]], None]] = None

        # Hotkey manager (Windows API)
        self._hotkey_manager = WindowsHotkeyManager(dispatch=self._dispatch_ui)
        self._register_hotkey()

    def _is_image_output_capable(self, model_id: str) -> bool:
        """Check models.json for image-output capability of a model id."""
        try:
            import json
            from pathlib import Path
            models_path = Path(__file__).resolve().parent / "models.json"
            if not models_path.exists():
                models_path = Path("models.json")
            if models_path.exists():
                with open(models_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                for m in data:
                    if m.get("id") == model_id:
                        caps = m.get("capabilities") or {}
                        return bool(caps.get("output_image"))
        except Exception:
            pass
        # Fallback hardcode for known image-output model id
        return model_id == "gemini-2.5-flash-image-preview"

    def _copy_image_bytes_to_clipboard(self, png_bytes: bytes) -> None:
        """Copy PNG bytes to Windows clipboard as a DIB (CF_DIB).

        Converts PNG to BMP in-memory and strips the BITMAPFILEHEADER (first 14 bytes)
        to produce a CF_DIB payload.
        """
        try:
            from io import BytesIO
            from PIL import Image
            import ctypes
            from ctypes import wintypes

            with Image.open(BytesIO(png_bytes)) as im:
                if im.mode not in ("RGB", "RGBA"):
                    im = im.convert("RGB")
                # Always save as BMP; CF_DIB expects a DIB without the 14-byte file header
                with BytesIO() as bmp_buffer:
                    im.save(bmp_buffer, format="BMP")
                    bmp_data = bmp_buffer.getvalue()
            # Strip BITMAPFILEHEADER (14 bytes)
            dib_data = bmp_data[14:]

            CF_DIB = 8
            GMEM_MOVEABLE = 0x0002

            user32 = ctypes.windll.user32
            kernel32 = ctypes.windll.kernel32
            memcpy = ctypes.cdll.msvcrt.memcpy

            if not user32.OpenClipboard(None):
                raise RuntimeError("OpenClipboard failed")
            try:
                if not user32.EmptyClipboard():
                    raise RuntimeError("EmptyClipboard failed")
                h_global = kernel32.GlobalAlloc(GMEM_MOVEABLE, len(dib_data))
                if not h_global:
                    raise RuntimeError("GlobalAlloc failed")
                p_global = kernel32.GlobalLock(h_global)
                if not p_global:
                    kernel32.GlobalFree(h_global)
                    raise RuntimeError("GlobalLock failed")
                try:
                    memcpy(p_global, dib_data, len(dib_data))
                finally:
                    kernel32.GlobalUnlock(h_global)

                if not user32.SetClipboardData(CF_DIB, h_global):
                    kernel32.GlobalFree(h_global)
                    raise RuntimeError("SetClipboardData failed")
                # Ownership of h_global is transferred to the system on success
            finally:
                user32.CloseClipboard()
        except Exception as e:
            print(f"ERROR: Failed to set image to clipboard: {e}")
            raise

    def _guess_mime_type(self, file_path: str) -> str:
        """Return a stable mime-type for known text/code formats to help Gemini parse files."""
        try:
            import mimetypes
            ext = str(Path(file_path).suffix).lower()
            if ext == '.py':
                return 'text/x-python'
            if ext in ('.txt',):
                return 'text/plain'
            if ext in ('.md', '.markdown'):
                return 'text/markdown'
            if ext in ('.json',):
                return 'application/json'
            mime_type, _ = mimetypes.guess_type(file_path)
            return mime_type or 'application/octet-stream'
        except Exception:
            return 'application/octet-stream'

    def _load_api_price_info(self) -> Dict:
        """api_price.jsonファイルを読み込む"""
        try:
            price_file_path = Path("api_price.json")
            if price_file_path.exists():
                with open(price_file_path, 'r', encoding='utf-8') as f:
                    return json.load(f)
            else:
                print(f"WARNING: {price_file_path} が見つかりません。デフォルトの価格情報を使用します。")
                return {}
        except Exception as e:
            print(f"ERROR: API価格情報の読み込みに失敗しました: {e}")
            return {}

    def _get_model_pricing(self, model_name: str, input_token_count: int = 0) -> tuple:
        """モデル名と入力トークン数に基づいて価格情報を取得する"""
        if not self.api_price_info:
            return 0.0, 0.0
            
        # モデル名の完全一致を試す
        model_info = self.api_price_info.get(model_name)
        if model_info:
            # 階層情報がある場合
            if "tiers" in model_info:
                # トークン数に応じた価格を取得
                for tier in sorted(model_info["tiers"], key=lambda x: x.get("threshold_tokens", 0), reverse=True):
                    if input_token_count <= tier.get("threshold_tokens", 0) or tier.get("threshold_tokens", 0) == -1:
                        return (
                            tier.get("input_cost_per_thousand_tokens", 0.0),
                            tier.get("output_cost_per_thousand_tokens", 0.0)
                        )
                # デフォルト価格を使用
                default = model_info.get("default", {})
                return (
                    default.get("input_cost_per_thousand_tokens", 0.0),
                    default.get("output_cost_per_thousand_tokens", 0.0)
                )
            # 階層情報がない場合（フラットな構造）
            elif "input_cost_per_thousand_tokens" in model_info:
                return (
                    model_info.get("input_cost_per_thousand_tokens", 0.0),
                    model_info.get("output_cost_per_thousand_tokens", 0.0)
                )
        
        # 部分一致で検索
        for key, model_info in self.api_price_info.items():
            if key in model_name:
                if "tiers" in model_info:
                    for tier in sorted(model_info["tiers"], key=lambda x: x.get("threshold_tokens", 0), reverse=True):
                        if input_token_count <= tier.get("threshold_tokens", 0) or tier.get("threshold_tokens", 0) == -1:
                            return (
                                tier.get("input_cost_per_thousand_tokens", 0.0),
                                tier.get("output_cost_per_thousand_tokens", 0.0)
                            )
                    default = model_info.get("default", {})
                    return (
                        default.get("input_cost_per_thousand_tokens", 0.0),
                        default.get("output_cost_per_thousand_tokens", 0.0)
                    )
                elif "input_cost_per_thousand_tokens" in model_info:
                    return (
                        model_info.get("input_cost_per_thousand_tokens", 0.0),
                        model_info.get("output_cost_per_thousand_tokens", 0.0)
                    )
        
        # 見つからない場合はデフォルト値
        return 0.0, 0.0

    def _register_hotkey(self):
        """Windows専用: ホットキーを WindowsHotkeyManager で登録。"""
        specs: List[tuple[str, Callable]] = []
        if getattr(self.config, 'hotkey_prompt_list', None):
            specs.append((self.config.hotkey_prompt_list, self._show_action_selector_gui))
        if getattr(self.config, 'hotkey_refine', None):
            specs.append((self.config.hotkey_refine, self.handle_refine))
        if getattr(self.config, 'hotkey_matrix', None):
            specs.append((self.config.hotkey_matrix, self.show_matrix_batch_processor_window))
        if getattr(self.config, 'hotkey_free_input', None):
            specs.append((self.config.hotkey_free_input, self.handle_free_input))
        try:
            self._hotkey_manager.register_hotkeys(specs)
        except Exception as e:
            print(f"WARNING: ホットキーの登録に失敗しました: {e}")

    def _dispatch_ui(self, callback: Callable) -> None:
        try:
            if self.app:
                self.app.after(0, callback)
            else:
                callback()
        except Exception:
            try:
                callback()
            except Exception:
                pass

    def update_hotkey(self, target: str, new_hotkey: Optional[str]):
        """Update a specific hotkey and re-register all.

        target: 'prompt_list' | 'refine' | 'matrix' | 'free_input'
        new_hotkey: hotkey string like 'ctrl+shift+g', or None/"" to disable
        """
        try:
            # Normalize empty to None
            new_val = new_hotkey if new_hotkey else None
            if target == 'prompt_list':
                setattr(self.config, 'hotkey_prompt_list', new_val)
                # For backward compatibility, also mirror to deprecated field
                self.config.hotkey = None
            elif target == 'refine':
                setattr(self.config, 'hotkey_refine', new_val)
            elif target == 'matrix':
                setattr(self.config, 'hotkey_matrix', new_val)
            elif target == 'free_input':
                setattr(self.config, 'hotkey_free_input', new_val)
            else:
                raise ValueError(f"Unknown hotkey target: {target}")
            # Re-register
            self._register_hotkey()
            save_config(self.config)
            print(f"INFO: {target} ホットキーを '{new_val}' に更新しました。")
            return True
        except Exception as e:
            print(f"ERROR: ホットキーの更新に失敗しました: {e}")
            return False

    def _parse_hotkey_to_win(self, hotkey_str: str) -> Optional[tuple]:
        """
        Parse a human-readable hotkey string (e.g., 'ctrl+shift+g') into a tuple of
        (modifier_flags, virtual_key_code) suitable for the Windows RegisterHotKey API.

        Returns None if parsing fails.
        """
        if not hotkey_str:
            return None
        mods = 0
        parts = [p.strip() for p in hotkey_str.lower().split('+')]
        key_part = parts[-1]
        for mod in parts[:-1]:
            if mod == 'ctrl':
                mods |= 0x0002  # MOD_CONTROL
            elif mod == 'shift':
                mods |= 0x0004  # MOD_SHIFT
            elif mod == 'alt':
                mods |= 0x0001  # MOD_ALT
            elif mod == 'win':
                mods |= 0x0008  # MOD_WIN
        # Determine the virtual-key code
        if len(key_part) == 1:
            vk = ord(key_part.upper())
        elif key_part.startswith('f') and key_part[1:].isdigit():
            fn = int(key_part[1:])
            vk = 0x70 + (fn - 1)  # F1 -> 0x70
        else:
            return None
        return mods, vk

    # 旧 Windows ホットキー実装（未使用）は削除予定でしたが、段階的移行のためプレースホルダに置換しました。

    # 旧 Windows ホットキー解除実装は未使用のため削除しました。

    # 旧 Windows メッセージループ実装は未使用のため削除しました。

    # ----------------------------------------------------------------------
    # New Windows hotkey implementation using its own message loop thread
    # ----------------------------------------------------------------------
    # 旧 Windows Hotkey v2 実装は未使用のため削除しました。

    # 旧 Windows Hotkey v2 解除実装は未使用のため削除しました。

    # 旧 Windows Hotkey v2 メッセージループ実装は未使用のため削除しました。

    def set_ui_elements(self, app: ctk.CTk, on_history_updated_callback: Optional[Callable[[List[str]], None]] = None):
        self.app = app
        # Notification service after app is set
        self._notification_service = NotificationService(self.app)
        # 監視は常に開始する（履歴ボタン等で履歴を利用するため）
        if on_history_updated_callback:
            self._on_history_updated_callback = on_history_updated_callback
        # Start clipboard history service (facade)
        try:
            from gemclip.infra.history import HistoryService
            self._history_service = HistoryService(
                max_size=self.max_history_size,
                on_updated=lambda items: self._on_history_updated(items),
            )
            self._history_service.start()
        except Exception:
            # Fallback to legacy monitor if service fails to start
            self._start_clipboard_monitor()

        # 追加指示用の直近結果・設定の初期化
        if not hasattr(self, 'last_result_text'):
            self.last_result_text = None
        if not hasattr(self, 'last_prompt_config'):
            self.last_prompt_config = None
        if not hasattr(self, 'last_generation_params'):
            self.last_generation_params = {}

    def _update_notification_message(self, chunk: str):
        if self._notification_service:
            self._notification_service.update_message(chunk)

    def _show_notification_ui(self, title: str, message: str, level: Literal["info", "warning", "error", "success"] = "info", duration_ms: Optional[int] = 3000):
        if self._notification_service:
            self._notification_service.show(title, message, level, duration_ms)

    def show_matrix_batch_processor_window(self, icon=None, item=None):
        if self.app:
            self.app.after(0, self._show_matrix_batch_processor_gui)

    def _show_matrix_batch_processor_gui(self):
        if self.matrix_batch_processor_window and self.matrix_batch_processor_window.winfo_exists():
            self.matrix_batch_processor_window.destroy()
            self.matrix_batch_processor_window = None

        # マトリクスに含めるフラグが立っているプロンプトのみをデフォルトで表示する
        # すべてのプロンプトをコピーして使用すると行列UIでの編集が設定ファイルに影響しない
        filtered_prompts = {pid: prompt for pid, prompt in self.config.prompts.items() if getattr(prompt, "include_in_matrix", False)}
        self.matrix_batch_processor_window = MatrixBatchProcessorWindow(
            prompts=filtered_prompts,
            on_processing_completed=self._on_batch_processing_completed,
            llm_agent_factory=self._create_llm_agent_for_matrix,
            notification_callback=self._show_notification_ui,
            worker_loop=self.loop,
            parent_app=self.app,
            agent=self
        )
        self.matrix_batch_processor_window.deiconify()
        self.matrix_batch_processor_window.lift()
        try:
            self.matrix_batch_processor_window.grab_set()
            self.app.wait_window(self.matrix_batch_processor_window)
        finally:
            if self.matrix_batch_processor_window and self.matrix_batch_processor_window.winfo_exists():
                try:
                    self.matrix_batch_processor_window.grab_release()
                except tk.TclError as e:
                    print(f"WARNING: _show_matrix_batch_processor_gui - grab_release中にTclErrorが発生しました: {e}")

    def _create_llm_agent_for_matrix(self, name: str, prompt_config: Prompt) -> LlmAgent:
        return LlmAgent(name=name, prompt_config=prompt_config)

    def _on_batch_processing_completed(self, result: str):
        pass

    def notify_prompts_changed(self):
        """Notify open Matrix window to refresh its prompt set from current config."""
        try:
            if self.app and self.matrix_batch_processor_window and self.matrix_batch_processor_window.winfo_exists():
                # Reflect latest prompts into matrix window on UI thread
                self.app.after(0, lambda: self.matrix_batch_processor_window.on_prompts_updated(self.config.prompts))
        except Exception:
            pass

    def _async_worker(self):
        self.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.loop)
        self.loop.set_debug(False) # 内部ログを抑制するためデバッグモードを無効化
        self._loop_ready_event.set()

        def _run_task_from_queue():
            try:
                task_info = self.task_queue.get_nowait()
                if task_info is None:
                    self.loop.call_soon_threadsafe(self.loop.stop)
                    return

                async def _process_task_internal():
                    try:
                        await self.run_async(**task_info)
                    except Exception as e:
                        error_message = tr("notify.agent_run_error", details=str(e))
                        if self.app:
                            self.app.after(0, lambda msg=error_message: self._show_notification_ui(tr("common.error"), msg, "error"))
                    finally:
                        self.task_queue.task_done()
                
                try:
                    self.loop.create_task(_process_task_internal())
                except Exception as e:
                    error_message = tr("notify.task_create_unexpected", details=str(e))
                    if self.app:
                        self.app.after(0, lambda msg=error_message: self._show_notification_ui(tr("common.error"), msg, "error"))
                    self.task_queue.task_done()
            except queue.Empty:
                pass
            except Exception as e:
                if self.app:
                    self.app.after(0, lambda e=e: self._show_notification_ui(tr("common.error"), tr("notify.worker_queue_unexpected", details=str(e)), "error"))
            finally:
                if self._worker_running:
                    self.loop.call_later(0.1, _run_task_from_queue)

        self.loop.call_soon_threadsafe(_run_task_from_queue)
        try:
            self.loop.run_forever()
        finally:
            if self.loop and not self.loop.is_closed():
                self.loop.close()

    def _copy_to_clipboard_and_notify(self, processed_text: str, prompt_config: Prompt, cost_message: str = ""):
        try:
            pyperclip.copy(processed_text)
            time.sleep(0.05)
            pasted_text = pyperclip.paste()
            if pasted_text != processed_text:
                 self.app.clipboard_clear()
                 self.app.clipboard_append(processed_text)
                 self.app.update()
        except Exception as e:
            print(f"ERROR: Clipboard operation failed: {e}")

        if not self.app:
            return

        threading.Thread(target=self._play_completion_sound, daemon=True).start()
        self._show_notification_ui(tr("notify.done_title"), tr("notify.copied_fmt", name=prompt_config.name, cost=cost_message), level="success")

    def _get_api_key(self) -> Optional[str]:
        return keyring.get_password(API_SERVICE_ID, "api_key")

    async def _generate_with_image_output_model(
        self,
        *,
        run_id: Optional[int],
        final_prompt_name: str,
        final_model_name: str,
        final_system_prompt: str,
        final_temperature: float,
        user_parts: List[Any],
        prompt_id: Optional[str],
    ) -> str:
        """Generate with google.genai for models that support image output and copy image to clipboard.

        Returns accumulated text (if any).
        """
        # Import the new client lazily to avoid conflicts with google.generativeai.types alias
        try:
            from google import genai as g2
            from google.genai import types as g2types
        except Exception as e:
            self._show_notification_ui(tr("common.error_title"), f"google.genai not available: {e}", level="error")
            raise

        client = g2.Client(api_key=self.api_key)

        # Map existing parts (dicts) to google.genai parts
        g2_parts: List[Any] = []
        for p in user_parts or []:
            try:
                if isinstance(p, dict) and isinstance(p.get("text"), str):
                    txt = p.get("text")
                    if txt and txt.strip():
                        g2_parts.append(g2types.Part.from_text(text=txt))
                elif isinstance(p, dict) and isinstance(p.get("inline_data"), dict):
                    inline = p["inline_data"]
                    mime = inline.get("mime_type") or "image/png"
                    data = inline.get("data")
                    if isinstance(data, str):
                        import base64 as _b64
                        data = _b64.b64decode(data)
                    if data:
                        g2_parts.append(g2types.Part.from_bytes(mime_type=mime, data=data))
                # Note: file refs from google.generativeai are not supported in google.genai path
            except Exception:
                pass

        # Ensure there is at least some instruction text
        if final_system_prompt and (not any(isinstance(pp, g2types.Part) and getattr(pp, "text", None) for pp in g2_parts)):
            g2_parts.insert(0, g2types.Part.from_text(text=final_system_prompt))

        contents = [
            g2types.Content(role="user", parts=g2_parts if g2_parts else [g2types.Part.from_text(text=final_system_prompt or "")])
        ]

        generate_content_config = g2types.GenerateContentConfig(
            response_modalities=["IMAGE", "TEXT"],
            temperature=final_temperature,
        )

        full_text = ""
        last_image_png: Optional[bytes] = None

        try:
            responses = client.models.generate_content_stream(
                model=final_model_name,
                contents=contents,
                config=generate_content_config,
            )
            for chunk in responses:
                # Text
                try:
                    if getattr(chunk, "text", None):
                        full_text += chunk.text
                        self.app.after(0, lambda c=chunk.text: self._update_notification_message(c))
                except Exception:
                    pass
                # Image inline_data
                try:
                    cands = getattr(chunk, "candidates", None)
                    if cands and cands[0].content and cands[0].content.parts:
                        for part in cands[0].content.parts:
                            inline = getattr(part, "inline_data", None)
                            if inline and getattr(inline, "data", None):
                                # Assume PNG bytes
                                last_image_png = inline.data
                except Exception:
                    pass
        except Exception as e:
            self._show_notification_ui(tr("notify.api_error_title"), tr("notify.unexpected_error", details=str(e)), level="error")
            raise

        # If image produced, copy to clipboard
        if last_image_png:
            try:
                self._copy_image_bytes_to_clipboard(last_image_png)
            except Exception:
                pass

        # Build prompt snapshot for DB
        final_prompt_config = Prompt(
            name=final_prompt_name,
            model=final_model_name,
            system_prompt=final_system_prompt,
            parameters=PromptParameters(
                temperature=final_temperature,
            ),
            enable_web=False,
        )

        cost_message = tr("pricing.unavailable_suffix")

        # Notify and persist
        if last_image_png:
            self._show_notification_ui(tr("notify.done_title"), tr("notify.copied_fmt", name=final_prompt_name, cost=cost_message), level="success")
            # Persist run output (image in blob)
            try:
                if run_id is not None:
                    out_id = db.add_run_output(run_id, prompt_id=prompt_id, input_id=None, content_text=full_text or None, error_json=None, content_blob=last_image_png)
                    db.mark_output_copied(out_id)
                    try:
                        db.finish_run(run_id, status="success")
                    except Exception:
                        pass
            except Exception:
                pass
            # Also keep as last_result_text for refine to work minimally
            self.last_result_text = full_text or "[image]"
            self.last_prompt_config = final_prompt_config
        elif full_text:
            # Fallback to text behavior
            self._copy_to_clipboard_and_notify(full_text, final_prompt_config, cost_message)
            try:
                if run_id is not None:
                    out_id = db.add_run_output(run_id, prompt_id=prompt_id, input_id=None, content_text=full_text)
                    db.mark_output_copied(out_id)
                    try:
                        db.finish_run(run_id, status='success')
                    except Exception:
                        pass
            except Exception:
                pass
            self.last_result_text = full_text
            self.last_prompt_config = final_prompt_config

        return full_text

    async def _process_clipboard_content(self, file_paths: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        contents = []
        if file_paths:
            # genai.Client を使用せず、genai.upload_file を直接使用
            for file_path in file_paths:
                try:
                    mime_type = self._guess_mime_type(file_path)
                    uploaded_file = await asyncio.to_thread(genai.upload_file, path=file_path, mime_type=mime_type) # genai.upload_file を使用
                    contents.append({"type": "file", "file_ref": uploaded_file})
                except Exception as e:
                    error_message = tr("notify.file_upload_failed", details=str(e))
                    self._show_notification_ui(tr("notify.file_upload_error"), error_message, "error")
                    raise RuntimeError(error_message)
            return contents
        else:
            max_retries = 3
            for i in range(max_retries):
                try:
                    image = ImageGrab.grabclipboard()
                    if isinstance(image, Image.Image):
                        if image.mode != 'RGB':
                            image = image.convert('RGB')
                        buffered = BytesIO()
                        image.save(buffered, format="PNG")
                        image_data = base64.b64encode(buffered.getvalue()).decode("utf-8")
                        contents.append({"type": "image", "data": image_data})
                        return contents
                    else:
                        original_text = pyperclip.paste()
                        if not original_text:
                            raise ValueError(tr("notify.clipboard_empty"))
                        contents.append({"type": "text", "data": original_text})
                        return contents
                except Exception as e:
                    await asyncio.sleep(0.1 * (i + 1))
            self._show_notification_ui(tr("notify.clipboard_error"), tr("notify.clipboard_get_failed"), "error")
            raise RuntimeError(tr("notify.clipboard_get_failed"))

    async def run_async(self, prompt_id: Optional[str] = None, file_paths: Optional[List[str]] = None, system_prompt: Optional[str] = None, model: Optional[str] = None, temperature: Optional[float] = None, top_p: Optional[float] = None, top_k: Optional[int] = None, max_output_tokens: Optional[int] = None, stop_sequences: Optional[List[str]] = None, refine_instruction: Optional[str] = None) -> str:
        if not self.app:
            raise RuntimeError("UI application not initialized.")

        if not self.api_key:
            error_message = tr("notify.api_key_missing_message")
            self._show_notification_ui(tr("notify.api_key_missing_title"), error_message, level="error")
            raise RuntimeError(error_message)

        try:
            final_prompt_name = tr("free_input.manual_prompt")
            final_system_prompt = system_prompt
            final_model_name = model
            final_temperature = temperature

            if refine_instruction:
                # 追加指示: prompt_id が指定されていればその設定を、無ければ直近設定を使用
                if prompt_id:
                    prompt_config = self.config.prompts.get(prompt_id)
                    if not prompt_config:
                        raise ValueError(tr("notify.prompt_missing_fmt", id=prompt_id))
                else:
                    if not getattr(self, 'last_result_text', None) or not getattr(self, 'last_prompt_config', None):
                        raise ValueError(tr("notify.no_last_result"))
                    prompt_config = self.last_prompt_config
                final_prompt_name = f"{prompt_config.name}{tr('refine.suffix')}"
                final_system_prompt = prompt_config.system_prompt
                final_model_name = prompt_config.model
                final_temperature = prompt_config.parameters.temperature
            elif prompt_id:
                prompt_config = self.config.prompts.get(prompt_id)
                if not prompt_config:
                    raise ValueError(tr("notify.prompt_missing_fmt", id=prompt_id))
                final_prompt_name = prompt_config.name
                final_system_prompt = prompt_config.system_prompt
                final_model_name = prompt_config.model
                final_temperature = prompt_config.parameters.temperature
            elif not (final_system_prompt and final_model_name and final_temperature is not None):
                raise ValueError(tr("notify.no_content"))

            self._show_notification_ui(tr("notify.running_fmt", name=final_prompt_name), tr("notify.sending"), duration_ms=None)

            # --- DB run initialization ---
            run_id = None
            run_type = 'refine' if refine_instruction else 'single'
            try:
                prompt_snapshot = {
                    "name": final_prompt_name,
                    "model": final_model_name,
                    "system_prompt": final_system_prompt,
                }
                # parameters are set after config is built below; initialize with minimal
                run_id = db.create_run(
                    run_type=run_type,
                    api_model_id=final_model_name,
                    parameters={"temperature": final_temperature},
                    prompt_snapshot=prompt_snapshot,
                )
            except Exception:
                run_id = None

            contents_to_send = []  # legacy flat list; we will convert to message parts
            user_parts: List[Any] = []  # unified parts list to send as one user message
            # Be extra robust: honor file paths from argument or any staged temp paths
            effective_file_paths = file_paths if file_paths else getattr(self, '_temp_file_paths_for_processing', None)
            try:
                if effective_file_paths:
                    print(f"DEBUG: Effective file_paths count = {len(effective_file_paths)}")
                else:
                    print("DEBUG: Effective file_paths is empty/null")
            except Exception:
                pass
            # Manual free input mode: no prompt_id, no refine; only send the dialog text and explicit attachments
            manual_free_input = (not refine_instruction and not prompt_id)
            if refine_instruction:
                # Build as multiple text parts within a single user message
                user_parts = [
                    {"text": f"{tr('refine.prev_output_label')}\n{self.last_result_text}"},
                    {"text": f"{tr('refine.additional_input_label')}\n{refine_instruction}"},
                    {"text": tr("refine.requirements_text")},
                ]
                # Save inputs into DB (as text clipboard items)
                if run_id is not None:
                    try:
                        prev_id = db.save_clipboard_item(self.last_result_text, source="refine_prev_output")
                        rid = db.add_run_input(run_id, prev_id, is_refine_prev_output=True)
                        instr_id = db.save_clipboard_item(refine_instruction, source="refine_instruction")
                        db.add_run_input(run_id, instr_id, is_refine_prev_output=False)
                    except Exception:
                        pass
            else:
                if manual_free_input:
                    # Use the provided free-input text as the actual message content
                    if final_system_prompt:
                        user_parts.append({"text": final_system_prompt})
                        if run_id is not None:
                            try:
                                cid = db.save_clipboard_item(final_system_prompt, source="manual_input")
                                db.add_run_input(run_id, cid)
                            except Exception:
                                pass
                    # Do not auto-read clipboard; only attach explicitly provided files
                    processed_contents = []
                    if effective_file_paths:
                        processed_contents = await self._process_clipboard_content(effective_file_paths)
                    # Clear system instruction in manual mode to avoid duplicating the same text
                    # Use empty string instead of None to satisfy Prompt model validation later
                    final_system_prompt = ""
                else:
                    # Prompt selection mode: combine temp override (history-selected item) and any attachments
                    processed_contents = []
                    try:
                        if getattr(self, '_temp_input_for_processing', None) is not None:
                            processed_contents.append(getattr(self, '_temp_input_for_processing'))
                            print("DEBUG: Using temp override content from history selection.")
                            delattr(self, '_temp_input_for_processing')
                    except Exception:
                        pass
                    if effective_file_paths:
                        file_parts = await self._process_clipboard_content(effective_file_paths)
                        processed_contents.extend(file_parts)

                # Append processed contents (images/files/text) to user parts and record DB inputs
                for content_info in (processed_contents or []):
                    if content_info["type"] == "image":
                        image_part = create_image_part(content_info["data"]) # 共通関数を呼び出し
                        user_parts.append(image_part)
                        if run_id is not None:
                            try:
                                cid = db.save_clipboard_item({"type": "image", "data": content_info["data"]}, source="action_selector")
                                db.add_run_input(run_id, cid)
                            except Exception:
                                pass
                    elif content_info["type"] == "file":
                        try:
                            if "file_ref" in content_info and content_info["file_ref"] is not None:
                                user_parts.append(content_info["file_ref"])
                                file_ref_path = getattr(content_info["file_ref"], 'path', None)
                            else:
                                # Upload now if only path provided
                                p = content_info.get("data")
                                mime_type = self._guess_mime_type(p)
                                uploaded = await asyncio.to_thread(genai.upload_file, path=p, mime_type=mime_type)
                                user_parts.append(uploaded)
                                file_ref_path = p
                            if run_id is not None and file_ref_path:
                                try:
                                    cid = db.save_clipboard_item({"type": "file", "data": file_ref_path}, source="file_attach")
                                    db.add_run_input(run_id, cid)
                                except Exception:
                                    pass
                        except Exception:
                            pass
                    else: # テキストの場合
                        text_content = content_info['data']
                        user_parts.append({"text": text_content})
                        if run_id is not None:
                            try:
                                cid = db.save_clipboard_item(text_content, source="clipboard_text")
                                db.add_run_input(run_id, cid)
                            except Exception:
                                pass

            # Ensure at least one text part exists (Gemini expects a textual instruction alongside files/images)
            def _has_text_part(parts: List[Any]) -> bool:
                try:
                    return any(isinstance(p, dict) and isinstance(p.get('text'), str) and p.get('text').strip() for p in parts)
                except Exception:
                    return False
            if user_parts and not _has_text_part(user_parts):
                # For image-output capable model, prefer a transform instruction that yields an image.
                if self._is_image_output_capable(final_model_name):
                    try:
                        user_parts.insert(0, {"text": tr("free_input.attach_image_transform_default")})
                    except Exception:
                        user_parts.insert(0, {"text": "Convert this image to a vivid color tone. Return the result as an image (PNG)."})
                else:
                    try:
                        user_parts.insert(0, {"text": tr("free_input.attach_prompt_default")})
                    except Exception:
                        user_parts.insert(0, {"text": "Please analyze the attached file(s)."})

            # For backward compatibility, keep a flat list (not used for send anymore)
            contents_to_send = user_parts[:] if user_parts else contents_to_send

            # Web検索ツールの有効化判定（SDK差異に備えつつ試す）
            tools_list = None
            try:
                enable_web_flag = False
                cfg = None
                if refine_instruction:
                    cfg = self.last_prompt_config
                elif prompt_id:
                    cfg = self.config.prompts.get(prompt_id)
                if cfg:
                    enable_web_flag = bool(getattr(cfg, 'enable_web', False))
                def _parts_have_url(parts: List[Any]) -> bool:
                    try:
                        for p in parts:
                            if isinstance(p, dict) and 'text' in p and isinstance(p['text'], str):
                                if p['text'].strip().startswith(("http://", "https://")):
                                    return True
                        return False
                    except Exception:
                        return False
                has_text_url = _parts_have_url(user_parts)
                if enable_web_flag or has_text_url:
                    tools_list = [{"google_search": {}}]
            except Exception:
                tools_list = None

            # Respect selected model (gemini-2.5-flash-lite also supports attachments); do not auto-upgrade model.

            # If model supports image output using the new google.genai client, branch here
            if self._is_image_output_capable(final_model_name):
                # Handle image-output path separately and return
                full_text_image = await self._generate_with_image_output_model(
                    run_id=run_id,
                    final_prompt_name=final_prompt_name,
                    final_model_name=final_model_name,
                    final_system_prompt=final_system_prompt,
                    final_temperature=final_temperature,
                    user_parts=user_parts,
                    prompt_id=prompt_id,
                )
                return full_text_image

            # Prefer setting tools on the model instance (SDKs may not accept tools in GenerationConfig)
            try:
                _kwargs = {}
                if final_system_prompt:
                    _kwargs["system_instruction"] = final_system_prompt
                if tools_list:
                    _kwargs["tools"] = tools_list
                model_instance = genai.GenerativeModel(final_model_name, **_kwargs)
            except TypeError:
                # Fallback: try without tools, then with only model
                if final_system_prompt:
                    try:
                        model_instance = genai.GenerativeModel(final_model_name, system_instruction=final_system_prompt)
                    except Exception:
                        model_instance = genai.GenerativeModel(final_model_name)
                else:
                    model_instance = genai.GenerativeModel(final_model_name)

            # GenerationConfig を構築（安全設定はデフォルト、ツールは有効なら付与）
            # Build GenerationConfig without tools by default; tools were set on the model when possible
            generate_content_config = types.GenerationConfig(
                temperature=final_temperature,
                top_p=top_p if prompt_id is None else (prompt_config.parameters.top_p if prompt_config and prompt_config.parameters else None),
                top_k=top_k if prompt_id is None else (prompt_config.parameters.top_k if prompt_config and prompt_config.parameters else None),
                max_output_tokens=max_output_tokens if prompt_id is None else (prompt_config.parameters.max_output_tokens if prompt_config and prompt_config.parameters else None),
                stop_sequences=stop_sequences if prompt_id is None else (prompt_config.parameters.stop_sequences if prompt_config and prompt_config.parameters else None),
            )

            input_token_count = 0
            try:
                # model_instance.count_tokens を使用（単一のユーザーメッセージ形式に統一）
                conv_payload = ([{"role": "user", "parts": user_parts}] if user_parts else contents_to_send)
                # Debug summary of payload parts
                try:
                    text_cnt = 0
                    file_cnt = 0
                    img_cnt = 0
                    for p in (user_parts or []):
                        if isinstance(p, dict) and 'text' in p:
                            if isinstance(p.get('text'), str) and p.get('text').strip():
                                text_cnt += 1
                        elif isinstance(p, dict) and 'inline_data' in p:
                            img_cnt += 1
                        else:
                            # Likely a FileRef or unknown part
                            file_cnt += 1
                    print(f"DEBUG: Parts summary - text={text_cnt}, files={file_cnt}, images={img_cnt}")
                except Exception:
                    pass
                count_tokens_response = await asyncio.to_thread(
                    model_instance.count_tokens,
                    contents=conv_payload,
                )
                input_token_count = count_tokens_response.total_tokens
                print(f"DEBUG: Input token count: {input_token_count}")
            except Exception as e:
                print(f"WARNING: Failed to count input tokens: {e}")

            full_response_text = ""
            # 生成（Web検索ツールがエラーならフォールバック）。単一のユーザーメッセージで送信
            def _gen(stream_flag: bool, config):
                payload = ([{"role": "user", "parts": user_parts}] if user_parts else contents_to_send)
                return model_instance.generate_content(
                    contents=payload,
                    stream=stream_flag,
                    generation_config=config,
                )

            try:
                responses = _gen(True, generate_content_config)
            except Exception:
                # Fallback: recreate model with alternative tool field if supported
                if tools_list:
                    try:
                        _kwargs = {"tools": [{"google_search_retrieval": {}}]}
                        if final_system_prompt:
                            _kwargs["system_instruction"] = final_system_prompt
                        model_instance = genai.GenerativeModel(final_model_name, **_kwargs)
                    except Exception:
                        pass
                responses = _gen(True, generate_content_config)

            for chunk in responses:
                # chunk.text を参照する際に ValueError を吐くことがあるため安全に取得する
                text = None
                try:
                    text = chunk.text
                except Exception:
                    # `chunk.text` が取得できない場合は候補が安全性によりブロックされているとみなす
                    pass

                if text:
                    full_response_text += text
                    # クロージャ内で chunk.text を再度評価しないよう text を閉じ込める
                    self.app.after(0, lambda c=text: self._update_notification_message(c))
                elif chunk.prompt_feedback and chunk.prompt_feedback.block_reason:
                    # Prompt was blocked due to safety settings
                    full_response_text = tr("safety.request_blocked_message")
                    self.app.after(0, lambda: self._show_notification_ui(tr("safety.request_blocked_title"), full_response_text, level="error"))
                    break  # Stop processing further chunks
                elif chunk.candidates and (not chunk.candidates[0].content.parts or chunk.candidates[0].finish_reason):
                    # Candidate was blocked or finished due to safety settings or other reasons
                    full_response_text = tr("safety.response_blocked_message")
                    self.app.after(0, lambda: self._show_notification_ui(tr("safety.response_blocked_title"), full_response_text, level="error"))
                    break  # Stop processing further chunks

            output_token_count = (await asyncio.to_thread(
                model_instance.count_tokens,
                contents=[full_response_text],
            )).total_tokens
            print(f"DEBUG: Output token count: {output_token_count}")

            # 価格情報を取得（推定コストの表示用）
            input_cost_per_thousand_tokens, output_cost_per_thousand_tokens = self._get_model_pricing(final_model_name, input_token_count)

            estimated_cost = (input_token_count / 1000) * input_cost_per_thousand_tokens + \
                             (output_token_count / 1000) * output_cost_per_thousand_tokens

            cost_message_suffix = ""
            if input_cost_per_thousand_tokens == 0.0 and output_cost_per_thousand_tokens == 0.0:
                cost_message_suffix = tr("pricing.unavailable_suffix")

            cost_message = (f"{tr('pricing.estimated_cost_prefix')}{estimated_cost:.6f}{cost_message_suffix}"
                            if (input_token_count or output_token_count) else cost_message_suffix)

            if full_response_text:
                final_prompt_config = Prompt(
                    name=final_prompt_name,
                    model=final_model_name,
                    system_prompt=final_system_prompt,
                    parameters=PromptParameters(
                        temperature=final_temperature,
                        top_p=generate_content_config.top_p,
                        top_k=generate_content_config.top_k,
                        max_output_tokens=generate_content_config.max_output_tokens,
                        stop_sequences=generate_content_config.stop_sequences,
                    ),
                    enable_web=bool(tools_list)
                )
                self._copy_to_clipboard_and_notify(full_response_text, final_prompt_config, cost_message)
                # Save run output and mark copied
                try:
                    if run_id is not None:
                        out_id = db.add_run_output(run_id, prompt_id=prompt_id, input_id=None, content_text=full_response_text)
                        db.mark_output_copied(out_id)
                        # Update final parameters and finish run
                        params = {
                            "temperature": final_temperature,
                            "top_p": generate_content_config.top_p,
                            "top_k": generate_content_config.top_k,
                            "max_output_tokens": generate_content_config.max_output_tokens,
                            "stop_sequences": generate_content_config.stop_sequences,
                        }
                        # Update finish with tokens/cost
                        try:
                            db.finish_run(run_id, status="success", input_tokens=input_token_count, output_tokens=output_token_count, cost_usd=estimated_cost)
                        except Exception:
                            db.finish_run(run_id, status="success")
                except Exception:
                    pass
                # 直近結果を保持（追加指示用）
                self.last_result_text = full_response_text
                self.last_prompt_config = final_prompt_config
                self.last_generation_params = {
                    "temperature": final_temperature,
                    "top_p": generate_content_config.top_p,
                    "top_k": generate_content_config.top_k,
                    "max_output_tokens": generate_content_config.max_output_tokens,
                    "stop_sequences": generate_content_config.stop_sequences,
                }
            else:
                if self._current_notification_popup_window and self._current_notification_popup_window.winfo_exists():
                    self.app.after(0, self._current_notification_popup_window.destroy)

            return full_response_text

        except exceptions.GoogleAPICallError as e:
            error_message = tr("notify.api_error_message", code=e.code, message=e.message)
            self._show_notification_ui(tr("notify.api_error_title"), error_message, level="error")
            try:
                if 'run_id' in locals() and run_id is not None:
                    db.finish_run(run_id, status="error")
            except Exception:
                pass
            raise RuntimeError(error_message)
        except Exception as e:
            error_message = tr("notify.unexpected_error", details=str(e))
            if "finish_reason: SAFETY" in str(e) or (hasattr(e, '__cause__') and e.__cause__ and "finish_reason: SAFETY" in str(e.__cause__)):
                error_message = tr("safety.request_blocked_message")
                self._show_notification_ui(tr("safety.request_blocked_title"), error_message, level="error")
            else:
                self._show_notification_ui(tr("common.error"), error_message, level="error")
            traceback.print_exc() # スタックトレースを出力
            try:
                if 'run_id' in locals() and run_id is not None:
                    db.finish_run(run_id, status="error")
            except Exception:
                pass
            raise
        finally:
            # Clear staged temp file paths after use to avoid unintended reuse
            try:
                if hasattr(self, '_temp_file_paths_for_processing'):
                    delattr(self, '_temp_file_paths_for_processing')
            except Exception:
                pass

    def _play_completion_sound(self):
        """Play a short completion sound, if supported on this platform.

        - On Windows, use winsound to play the bundled file or a system alias.
        - On other platforms, silently skip (no external deps introduced).
        """
        sound_file = Path(COMPLETION_SOUND_FILE)
        try:
            if winsound is not None:
                if sound_file.exists():
                    winsound.PlaySound(str(sound_file.resolve()), winsound.SND_FILENAME | winsound.SND_ASYNC)
                else:
                    winsound.PlaySound("SystemHand", winsound.SND_ALIAS | winsound.SND_ASYNC)
        except Exception as e:
            print(f"ERROR: Sound playback failed: {e}")

    def add_prompt(self, prompt_id: str, prompt: Prompt):
        if prompt_id in self.config.prompts:
            raise ValueError(tr("prompt.id_exists", id=prompt_id))
        self.config.prompts[prompt_id] = prompt

    def update_prompt(self, prompt_id: str, updated_prompt: Prompt):
        if prompt_id not in self.config.prompts:
            raise ValueError(tr("prompt.id_missing", id=prompt_id))
        self.config.prompts[prompt_id] = updated_prompt

    def delete_prompt(self, prompt_id: str):
        if prompt_id not in self.config.prompts:
            raise ValueError(tr("prompt.id_missing", id=prompt_id))
        del self.config.prompts[prompt_id]

    def _on_prompt_selected(self, prompt_id: str, file_paths: Optional[List[str]] = None):
        self._run_process_in_thread(prompt_id=prompt_id, file_paths=file_paths)

    def _run_process_in_thread(self, **kwargs):
        try:
            self.task_queue.put(kwargs)
        except Exception as e:
            self._show_notification_ui(tr("common.error"), tr("notify.task_enqueue_failed", details=str(e)), level="error")

    def _show_action_selector_gui(self, *args, **kwargs):
        """
        ActionSelectorWindowを表示する。
        pystrayからの呼び出しと、内部からの呼び出し(file_paths付き)の両方に対応する。
        """
        try:
            print("DEBUG: _show_action_selector_gui invoked.")
        except Exception:
            pass
        file_paths = kwargs.get('file_paths')

        # 既存のウィンドウがあれば、安全に破棄する
        if self._current_action_selector_window and self._current_action_selector_window.winfo_exists():
            self._current_action_selector_window.destroy()
            self._current_action_selector_window = None

        # カーソル位置を取得
        cursor_x = self.app.winfo_pointerx()
        cursor_y = self.app.winfo_pointery()

        # 新しいウィンドウを作成して表示する
        self.app.after(50, lambda: self._create_and_show_action_selector(file_paths=file_paths, cursor_pos=(cursor_x, cursor_y)))

    def _create_and_show_action_selector(self, file_paths: Optional[List[str]] = None, cursor_pos: Optional[tuple] = None):
        """ActionSelectorWindowを作成して表示するヘルパーメソッド"""
        # 既に別のインスタンスが存在していたら何もしない（念のため）
        if self._current_action_selector_window and self._current_action_selector_window.winfo_exists():
            return

        if self.app:
            self._current_action_selector_window = ActionSelectorWindow(
                prompts=self.config.prompts,
                on_prompt_selected_callback=self._on_prompt_selected,
                agent=self,
                file_paths=file_paths,
                on_destroy_callback=lambda: setattr(self, '_current_action_selector_window', None)
            )
            self._current_action_selector_window.show_at_cursor(cursor_pos=cursor_pos)

    def _show_main_window(self, icon=None, item=None):
        if self.app:
            try:
                # Release any existing Tk grab globally to avoid blocked interactions
                try:
                    cur = self.app.tk.call('grab', 'current')
                    if cur:
                        try:
                            self.app.nametowidget(cur).grab_release()
                        except Exception:
                            # Fallback: release grab at root level if possible
                            self.app.grab_release()
                except Exception:
                    pass
                self.app.deiconify()
                # Temporarily set topmost to reliably lift above other toplevels (e.g., matrix window)
                try:
                    self.app.attributes("-topmost", True)
                except Exception:
                    pass
                self.app.lift()
                try:
                    self.app.focus_force()
                except Exception:
                    pass
                # Drop topmost shortly after so normal z-order behavior resumes
                self.app.after(250, lambda: self._unset_topmost_safe())
            except Exception:
                pass

    def _unset_topmost_safe(self):
        try:
            if self.app and self.app.winfo_exists():
                self.app.attributes("-topmost", False)
        except Exception:
            pass

    def show_settings_window(self, icon=None, item=None):
        print("DEBUG: show_settings_window called.")
        if self.app:
            if self._settings_window and self._settings_window.winfo_exists():
                print("DEBUG: show_settings_window - Existing settings window found, destroying it.")
                self._settings_window.destroy()
                self._settings_window = None
            self._settings_window = SettingsWindow(parent_app=self.app, agent=self)
            self._settings_window.deiconify()
            self._settings_window.lift()
            try:
                print(f"DEBUG: show_settings_window - Calling grab_set. Current grab: {self.app.grab_current()}")
                self._settings_window.grab_set()
                print(f"DEBUG: show_settings_window - grab_set called. New grab: {self.app.grab_current()}")
                self.app.wait_window(self._settings_window)
                print("DEBUG: show_settings_window - Settings window closed.")
            finally:
                if self._settings_window and self._settings_window.winfo_exists():
                    try:
                        print(f"DEBUG: show_settings_window - Calling grab_release. Current grab: {self.app.grab_current()}")
                        self._settings_window.grab_release()
                        print("DEBUG: show_settings_window - grab_release called successfully.")
                    except tk.TclError as e:
                        print(f"WARNING: show_settings_window - grab_release中にTclErrorが発生しました: {e}")
                self._settings_window = None
        else:
            print("ERROR: show_settings_window - self.app is not initialized.")

    def handle_free_input(self):
        # ActionSelectorWindowが存在し、grab_setされている場合、grab_releaseを呼び出す
        if self._current_action_selector_window and self._current_action_selector_window.winfo_exists():
            if self._current_action_selector_window.grab_current() == str(self._current_action_selector_window):
                try:
                    self._current_action_selector_window.grab_release()
                    print("DEBUG: ActionSelectorWindow grab_release called before opening ResizableInputDialog.")
                except tk.TclError as e:
                    print(f"WARNING: handle_free_input - ActionSelectorWindow grab_release中にTclErrorが発生しました: {e}")

        dialog = ResizableInputDialog(parent_app=self.app, text=tr("free_input.prompt_label"), title=tr("free_input.title"), agent=self, enable_history=True)
        dialog.show() # ここでshow()メソッドを呼び出す
        prompt_text = dialog.get_input()
        if prompt_text:
            # Read model/temperature from the dialog if available
            model_name = "gemini-2.5-flash-lite"
            try:
                if hasattr(dialog, 'model_variable'):
                    display_val = dialog.model_variable.get()
                    # Labels look like: "gemini-2.5-flash-lite (高速、低精度)" -> pick the first token
                    model_name = display_val.split(" ")[0] if display_val else model_name
            except Exception:
                pass
            temperature_val = 1.0
            try:
                if hasattr(dialog, 'temperature_slider'):
                    temperature_val = float(dialog.temperature_slider.get())
            except Exception:
                pass

            # Check if there's a temporary file path from a previous file attachment
            file_paths_to_process = getattr(self, '_temp_file_paths_for_processing', None)
            self._run_process_in_thread(system_prompt=prompt_text, model=model_name, temperature=temperature_val, file_paths=file_paths_to_process)
            # Clear the temporary file path after use
            if hasattr(self, '_temp_file_paths_for_processing'):
                del self._temp_file_paths_for_processing

    def _on_prompt_selected(self, prompt_id: str, file_paths: Optional[List[str]] = None):
        # もし一時ファイルパスが設定されていればそれを使用し、そうでなければNone
        final_file_paths = file_paths if file_paths else getattr(self, '_temp_file_paths_for_processing', None)
        
        self._run_process_in_thread(prompt_id=prompt_id, file_paths=final_file_paths)
        
        # 処理後、一時ファイルパスをクリア
        if hasattr(self, '_temp_file_paths_for_processing'):
            del self._temp_file_paths_for_processing

    def handle_refine(self, icon=None, item=None):
        if not getattr(self, 'last_result_text', None):
            self._show_notification_ui(tr("refine.title"), tr("notify.no_last_result"), level="warning")
            return
        dialog = ResizableInputDialog(parent_app=self.app, text=tr("refine.prompt_label"), title=tr("refine.title"))
        dialog.show()
        instruction = dialog.get_input()
        if instruction:
            self._run_process_in_thread(refine_instruction=instruction)

    def handle_file_attach(self):
        file_paths = filedialog.askopenfilenames() # 複数ファイル選択を許可
        if file_paths:
            # 絶対パスに正規化して保存
            self._temp_file_paths_for_processing = [str(Path(p).resolve()) for p in file_paths]
            
            # 選択されたファイル名を通知として表示
            # file_names = [Path(p).name for p in file_paths]
            # notification_message = "以下のファイルを添付しました:\n" + "\n".join(file_names)
            # self._show_notification_ui("ファイル添付", notification_message, level="info", duration_ms=5000)

            # ファイル選択後、プロンプト選択画面を再表示
            self._show_action_selector_gui(file_paths=list(file_paths))
        # else:
        #     self._show_notification_ui("ファイル添付", "ファイルが選択されませんでした。", level="info", duration_ms=2000)

    def quit_app(self, icon=None, item=None):
        # Unregister Windows hotkeys (if any) and stop the listener thread
        try:
            if hasattr(self, '_hotkey_manager') and self._hotkey_manager:
                self._hotkey_manager.unregister_all()
        except Exception:
            pass
        # Remove all keyboard hotkeys to clean up fallback registrations
        try:
            keyboard.unhook_all()
        except Exception:
            pass
        self._worker_running = False
        if self._clipboard_monitor_thread and self._clipboard_monitor_thread.is_alive():
            self._clipboard_monitor_running = False
            self._clipboard_monitor_thread.join(timeout=1.0)
        if self.task_queue:
            self.task_queue.put(None)
        if self.worker_thread and self.worker_thread.is_alive():
            self.worker_thread.join(timeout=1.0)
        
        if self._current_action_selector_window and self._current_action_selector_window.winfo_exists():
            self._current_action_selector_window.destroy()

        if self.matrix_batch_processor_window and self.matrix_batch_processor_window.winfo_exists():
            self.matrix_batch_processor_window.destroy()

        if self._settings_window and self._settings_window.winfo_exists():
            self._settings_window.destroy()

        if self.app:
            self.app.quit()
            self.app.destroy()
        if hasattr(self, 'tray_icon') and self.tray_icon:
            self.tray_icon.stop()
        sys.exit(0)

    def _start_clipboard_monitor(self):
        if not self._clipboard_monitor_running:
            self._clipboard_monitor_running = True
            self._clipboard_monitor_thread = threading.Thread(target=self._clipboard_monitor, daemon=True)
            self._clipboard_monitor_thread.start()

    def stop_clipboard_monitor(self):
        """Stop the clipboard monitoring service gracefully."""
        try:
            if hasattr(self, '_history_service') and self._history_service:
                self._history_service.stop()
        except Exception:
            pass
        # Stop legacy thread if running
        if self._clipboard_monitor_running:
            self._clipboard_monitor_running = False
            if self._clipboard_monitor_thread and self._clipboard_monitor_thread.is_alive():
                try:
                    self._clipboard_monitor_thread.join(timeout=1.0)
                except Exception:
                    pass

    def _clipboard_monitor(self):
        """テキストだけでなく、画像やファイルの履歴も収集する。"""
        last_signature: Optional[str] = None

        while self._clipboard_monitor_running:
            try:
                items_to_add = None
                signature = None

                # 1) 画像/ファイルのクリップボードを優先チェック
                try:
                    clip_obj = ImageGrab.grabclipboard()
                except Exception:
                    clip_obj = None

                if isinstance(clip_obj, Image.Image):
                    image = clip_obj
                    if image.mode != 'RGB':
                        image = image.convert('RGB')
                    # Convert image to bytes and compress to reduce memory footprint. Use zlib
                    # to compress the raw PNG bytes before Base64 encoding. This makes
                    # history storage more compact for large images.
                    with BytesIO() as buffer:
                        image.save(buffer, format='PNG')
                        image_bytes = buffer.getvalue()
                    try:
                        import zlib
                        compressed = zlib.compress(image_bytes)
                        encoded = base64.b64encode(compressed).decode('utf-8')
                        items_to_add = [{"type": "image_compressed", "data": encoded}]
                    except Exception:
                        # Fallback to storing uncompressed Base64 if compression fails
                        encoded = base64.b64encode(image_bytes).decode('utf-8')
                        items_to_add = [{"type": "image", "data": encoded}]
                    signature = "img:" + hashlib.sha1(image_bytes).hexdigest()
                elif isinstance(clip_obj, list):
                    file_paths = [p for p in clip_obj if isinstance(p, str)]
                    if file_paths:
                        items_to_add = [{"type": "file", "data": p} for p in file_paths]
                        signature = "files:" + "|".join(file_paths)

                # 2) テキストのチェック（上で何も取得できなかった場合）
                if items_to_add is None:
                    try:
                        text_content = pyperclip.paste()
                    except Exception:
                        text_content = ""
                    if text_content:
                        items_to_add = [{"type": "text", "data": text_content}]
                        signature = "text:" + hashlib.sha1(text_content.encode('utf-8')).hexdigest()

                # 3) 新規内容のみ履歴に追加
                if items_to_add and signature != last_signature:
                    for it in items_to_add:
                        self._add_to_history(it)
                    last_signature = signature

                time.sleep(0.5)
            except Exception:
                time.sleep(1)

    def _add_to_history(self, content: Any):
        # Delegate to service
        try:
            if hasattr(self, '_history_service') and self._history_service:
                self._history_service.add(content)
                return
        except Exception:
            pass
        # Legacy fallback
        try:
            self.clipboard_history.insert(0, content)
        except Exception:
            pass

    def _on_history_updated(self, items: List[Any]):
        # keep compatibility with existing consumers
        self.clipboard_history = items
        if self._on_history_updated_callback:
            if self.app:
                try:
                    self.app.after(0, lambda: self._on_history_updated_callback(self.clipboard_history))
                except Exception:
                    self._on_history_updated_callback(self.clipboard_history)
            else:
                self._on_history_updated_callback(self.clipboard_history)

    def create_tray_icon(self):
        image = Image.open(ICON_FILE)
        menu = (
            MenuItem(tr('tray.list'), self._show_action_selector_gui, default=True),
            MenuItem(tr('tray.matrix'), self.show_matrix_batch_processor_window),
            # MenuItem('追加指示…', self.handle_refine),
            MenuItem(tr('tray.manager'), self._show_main_window),
            MenuItem(tr('tray.settings'), self.show_settings_window),
            MenuItem(tr('tray.quit'), self.quit_app)
        )
        self.tray_icon = Icon(APP_NAME, image, APP_NAME, menu)
        return self.tray_icon

    def run(self):
        self.icon = self.create_tray_icon()
        self.icon.run()
