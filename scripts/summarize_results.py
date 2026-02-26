#!/usr/bin/env python3
from __future__ import annotations

import runpy
import sys
from pathlib import Path

TARGET = Path(__file__).resolve().parents[1] / "scripts/common/summarize_results.py"
print(f"[DEPRECATED ENTRYPOINT] Use scripts/common/summarize_results.py", file=sys.stderr)
runpy.run_path(str(TARGET), run_name="__main__")
