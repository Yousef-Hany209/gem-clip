"""Core facade for Gem Clip.

Re-exports core constants, paths utilities, logging setup, and common models.
This consolidates entry points for consumers without moving implementation yet.
"""

# Constants and model options
from constants import (
    APP_NAME,
    CONFIG_FILE,
    API_SERVICE_ID,
    ICON_FILE,
    COMPLETION_SOUND_FILE,
    DELETE_ICON_FILE,
    SUPPORTED_MODELS,
    model_id_to_label,
    model_label_to_id,
)

# Paths helpers
from paths import (
    get_base_dir,
    get_config_file_path,
    get_log_dir,
    get_data_dir,
)

# Logging
from logging_conf import setup_logging

# Common models
from common_models import (
    AppConfig,
    Prompt,
    PromptParameters,
    BaseAgent,
    LlmAgent,
    create_image_part,
)

__all__ = [
    # constants
    "APP_NAME",
    "CONFIG_FILE",
    "API_SERVICE_ID",
    "ICON_FILE",
    "COMPLETION_SOUND_FILE",
    "DELETE_ICON_FILE",
    "SUPPORTED_MODELS",
    "model_id_to_label",
    "model_label_to_id",
    # paths
    "get_base_dir",
    "get_config_file_path",
    "get_log_dir",
    "get_data_dir",
    # logging
    "setup_logging",
    # models/agents
    "AppConfig",
    "Prompt",
    "PromptParameters",
    "BaseAgent",
    "LlmAgent",
    "create_image_part",
]
