"""Logging helpers and run manifests."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional


def _timestamp() -> str:
    return datetime.now(tz=timezone.utc).strftime("%Y%m%dT%H%M%SZ")


@dataclass
class RunLogger:
    name: str = "induction_head"
    level: int = logging.INFO

    def configure(self) -> logging.Logger:
        logger = logging.getLogger(self.name)
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        logger.setLevel(self.level)
        return logger


def create_run_manifest(
    output_dir: Path, config: Dict[str, Any], extras: Optional[Dict[str, Any]] = None
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "timestamp": _timestamp(),
        "config": config,
        "metadata": extras or {},
    }
    manifest_path = output_dir / "run_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest_path
