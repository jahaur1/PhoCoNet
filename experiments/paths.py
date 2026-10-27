"""Shared, version-isolated locations for all formal PhoCoNet experiments."""
from pathlib import Path
import os
import re

ROOT = Path(__file__).resolve().parents[1]
RUN_ID = os.environ.get("PHOCONET_RUN_ID", "")
if RUN_ID and not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", RUN_ID):
    raise ValueError("PHOCONET_RUN_ID must be one safe directory name")
RESULT_ROOT = ROOT / "result" / RUN_ID if RUN_ID else ROOT / "result"
