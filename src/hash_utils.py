"""Utility helpers for hashing datasets and configs."""

from __future__ import annotations

import hashlib
from typing import Iterable


def hash_strings(values: Iterable[str]) -> str:
    canonical = "\n".join(sorted(values))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()
