#!/usr/bin/env python3
from __future__ import annotations

import runpy
import sys
from pathlib import Path

TARGET = Path(__file__).resolve().parents[1] / "scripts/phase1/run_head_validity_suite.py"
print(f"[DEPRECATED ENTRYPOINT] Use scripts/phase1/run_head_validity_suite.py", file=sys.stderr)
runpy.run_path(str(TARGET), run_name="__main__")
