"""The frozen parameters of the final run: ``configs/final_protocol.json`` (``docs/UEMR_FINAL_PROTOCOL.md``).

The constants of ``partitions_uemr`` (and so ``qualitative``) are read from here. The CLI defaults of
``uemr_fusion.train`` and the post-processing defaults of ``uniav_app`` (own runtime, imports nothing from
``omni_retrieval``) stay in their code and are checked against this file by the tests. ``UEMR_PROTOCOL`` (env)
points at another file.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

PATH = Path(os.environ.get("UEMR_PROTOCOL") or Path(__file__).resolve().parents[1] / "configs" / "final_protocol.json")
P = json.loads(PATH.read_text(encoding="utf-8"))
