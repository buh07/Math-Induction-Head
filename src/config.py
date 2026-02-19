"""Configuration helpers for the rebooted experiments."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import yaml


def _coerce_path(path: Any) -> Path:
    if isinstance(path, Path):
        return path
    return Path(str(path))


def load_config_file(path: Any) -> Dict[str, Any]:
    """Load a YAML configuration file. Missing files yield an empty dict."""
    cfg_path = _coerce_path(path)
    if not cfg_path.exists():
        return {}
    with cfg_path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Expected mapping in {cfg_path}, found {type(data)}")
    return data


@dataclass
class ExperimentConfig:
    """Minimal experiment configuration."""

    model_name: str = "meta-llama/Llama-2-7b-hf"
    seed: int = 42
    problem_count: int = 50
    log_level: str = "INFO"
    devices: str = "5,6"

    @classmethod
    def from_dict(cls, data: Optional[Dict[str, Any]]) -> "ExperimentConfig":
        if data is None:
            data = {}
        filtered = {
            key: data[key]
            for key in ("model_name", "seed", "problem_count", "log_level", "devices")
            if key in data
        }
        return cls(**filtered)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "model_name": self.model_name,
            "seed": self.seed,
            "problem_count": self.problem_count,
            "log_level": self.log_level,
            "devices": self.devices,
        }
