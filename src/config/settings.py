"""
FICOS — Environment-based Backend Settings
===========================================
Supports configuration via environment variables and .env file.
Zero hardcoded hostnames, ports, or absolute machine-specific file paths.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import List, Union
from functools import lru_cache
from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


def _find_project_root() -> Path:
    """Resolve project root directory."""
    return Path(__file__).resolve().parent.parent.parent


_PROJECT_ROOT = _find_project_root()


class Settings(BaseSettings):
    """
    FICOS Platform Settings.
    All attributes can be configured via environment variables (.env or deployment config).
    """

    # Server Networking
    host: str = Field(default="0.0.0.0", alias="HOST")
    port: int = Field(default=8000, alias="PORT")
    environment: str = Field(default="production", alias="ENVIRONMENT")
    model_version: str = Field(default="3.0.0", alias="MODEL_VERSION")

    # CORS Configuration
    cors_origins: Union[List[str], str] = Field(
        default=[
            "http://localhost:5173",
            "http://localhost:3000",
            "http://127.0.0.1:5173",
            "http://127.0.0.1:3000",
            "*"
        ],
        alias="CORS_ORIGINS"
    )

    # Core Directory Paths
    app_root: Path = Field(default=_PROJECT_ROOT)
    configs_dir: Path = Field(default=_PROJECT_ROOT / "configs", alias="FICOS_CONFIGS_DIR")
    data_dir: Path = Field(default=_PROJECT_ROOT / "data", alias="FICOS_DATA_DIR")
    outputs_dir: Path = Field(default=_PROJECT_ROOT / "outputs", alias="FICOS_OUTPUTS_DIR")
    models_dir: Path = Field(default=_PROJECT_ROOT / "models", alias="FICOS_MODELS_DIR")

    # Dataset & Model Artifact Paths
    dataset_a_path: Path = Field(
        default=_PROJECT_ROOT / "data" / "dataset_a_final_clean.csv",
        alias="FICOS_DATASET_A_PATH"
    )
    dataset_b_path: Path = Field(
        default=_PROJECT_ROOT / "data" / "dataset_b_mvp.xlsx",
        alias="FICOS_DATASET_B_PATH"
    )
    dataset_c_path: Path = Field(
        default=_PROJECT_ROOT / "data" / "dataset_c_disruptions.csv",
        alias="FICOS_DATASET_C_PATH"
    )
    modeling_dataset_path: Path = Field(
        default=_PROJECT_ROOT / "outputs" / "modeling_dataset.csv",
        alias="FICOS_MODELING_DATASET_PATH"
    )

    model_config = SettingsConfigDict(
        env_file=str(_PROJECT_ROOT / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False
    )

    @field_validator("cors_origins", mode="before")
    @classmethod
    def parse_cors_origins(cls, v: Union[List[str], str]) -> List[str]:
        if isinstance(v, str):
            # Split comma-separated list
            origins = [origin.strip() for origin in v.split(",") if origin.strip()]
            return origins if origins else ["*"]
        return v

    def get_cors_origins_list(self) -> List[str]:
        """Return CORS origins as a list of strings."""
        if isinstance(self.cors_origins, list):
            return self.cors_origins
        return [o.strip() for o in str(self.cors_origins).split(",") if o.strip()]


@lru_cache()
def get_settings() -> Settings:
    """Cached singleton instance of Settings."""
    return Settings()
