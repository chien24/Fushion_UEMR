"""The frozen parameters of the final run: ``configs/final_protocol.json`` (``docs/UEMR_FINAL_PROTOCOL.md``).

The constants of ``partitions_uemr`` are read from here (values of the YC2 runs that the HT-Step code paths still use).
``UEMR_PROTOCOL`` (env) points at another file.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

PATH = Path(os.environ.get("UEMR_PROTOCOL") or Path(__file__).resolve().parents[1] / "configs" / "final_protocol.json")
P = json.loads(PATH.read_text(encoding="utf-8"))
