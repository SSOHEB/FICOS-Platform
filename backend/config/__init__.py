"""FICOS Configuration package."""
from .settings import (
    Settings,
    get_settings,
    BASE_DIR,
    CONFIGS_DIR,
    DATA_DIR,
    OUTPUTS_DIR,
)
from .canonical_config import CanonicalConfig

__all__ = [
    "Settings",
    "get_settings",
    "BASE_DIR",
    "CONFIGS_DIR",
    "DATA_DIR",
    "OUTPUTS_DIR",
    "CanonicalConfig",
]
