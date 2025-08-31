# Gem Clip Architecture Overview

This document provides a high-level overview of the Gem Clip project's architecture.

## Core Components

The application is primarily composed of two major parts: the **UI Layer** handled by `app.py`, and the **Backend Logic** managed by `agent.py`.

### 1. `app.py` - The UI Layer

-   **Framework:** Built with `customtkinter`, providing a modern look and feel.
-   **Responsibilities:**
    -   Manages the main application window, which serves as the "Prompt Manager".
    -   Handles UI for adding, editing, reordering, and deleting prompts.
    -   Launches other windows, such as the Settings window and the Matrix Batch Processor.
    -   Delegates all backend operations (like saving settings or running prompts) to the `ClipboardToolAgent`.

### 2. `agent.py` - The Backend Agent

-   **Class:** `ClipboardToolAgent` is the heart of the application.
-   **Responsibilities:**
    -   **System Tray Icon:** Manages the application's lifecycle via a `pystray` icon (run, show windows, quit).
    -   **API Communication:** Interfaces with the Google Gemini API to execute prompts. It handles prompt formatting, API calls, and processes responses.
    -   **Hotkey Management:** Uses the `pystray` and `keyboard` libraries to register and listen for global hotkeys (`Ctrl+Shift+C`, etc.) to trigger actions.
    -   **Clipboard Handling:** Monitors the clipboard for text and images using `pyperclip` and `Pillow`.
    -   **Configuration:** Works with `config_manager.py` to load and save settings, including prompts and API keys. API keys are securely stored using the `keyring` library.
    -   **Asynchronous Tasks:** Uses a `queue` and a separate `threading.Thread` with `asyncio` to run API calls in the background, preventing the UI from freezing.

### 3. `config_manager.py` - Configuration

-   Manages loading and saving the `config.json` file.
-   Creates a default configuration on the first run.
-   Handles schema normalization to ensure compatibility with future versions.

### 4. `gemclip/` - The Core Package

-   **`gemclip/__main__.py`:** The official entry point when running the app as a module (`python -m gemclip`). It handles command-line arguments.
-   **`gemclip/core.py`:** Contains core data structures like `Prompt` and `AppConfig`, shared across the application.
-   **`gemclip/features/`:** Houses logic for complex features. The `matrix/` subdirectory contains all components for the powerful Matrix Batch Processor.
-   **`gemclip/ui/`:** Contains reusable UI components and custom dialogs (`ActionSelectorWindow`, `SettingsWindow`, etc.).
-   **`gemclip/infra/`:** Manages infrastructure-level concerns like hotkey handling (`hotkeys.py`) and history management (`history.py`).

## Application Flow (Example: Executing a Prompt)

1.  The user presses the hotkey (e.g., `Ctrl+Shift+C`).
2.  `agent.py` (`WindowsHotkeyManager`) catches the hotkey press.
3.  The agent calls `_show_action_selector_gui()` to display the `ActionSelectorWindow` (from `gemclip/ui/action_selector.py`) at the cursor's position.
4.  The user selects a prompt from the list.
5.  The `ActionSelectorWindow` invokes the `_on_prompt_selected` callback in the agent, passing the chosen `prompt_id`.
6.  The agent gets the current clipboard content (text or image).
7.  The agent puts a new task into its `task_queue` with the prompt details and clipboard content.
8.  The background worker thread (`_async_worker`) picks up the task, formats the request, and sends it to the Gemini API.
9.  While the API call is in progress, a "Running..." notification is shown.
10. When the response is received, `agent.py` copies the result to the clipboard and shows a "Done" notification.

This architecture effectively separates the UI from the backend logic, allowing for non-blocking operations and a responsive user experience.