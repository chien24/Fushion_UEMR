"""third_party/athena is byte-identical to VENDORED.md, and uniav_app never imports omni_retrieval."""

import hashlib
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
ATHENA = ROOT / "third_party" / "athena"


def test_vendored_sha256():
    rows = re.findall(r"^\| `([^`]+)` \| (\d+) \| `([0-9a-f]{64})` \|", (ATHENA / "VENDORED.md").read_text(encoding="utf-8"),
                      flags=re.M)
    assert len(rows) >= 22
    for rel, size, sha in rows:
        data = (ATHENA / rel).read_bytes()
        assert len(data) == int(size), rel
        assert hashlib.sha256(data).hexdigest() == sha, rel


def test_no_omni_import():
    for f in (ROOT / "uniav_app").rglob("*.py"):
        text = f.read_text(encoding="utf-8")
        assert not re.search(r"^\s*(from|import)\s+omni_retrieval", text, flags=re.M), f
