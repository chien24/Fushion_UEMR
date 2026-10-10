"""jsonl / csv helpers and the uniform-K baseline."""

import json

import pytest

from uniav_app.baselines import uniform_segments
from uniav_app.io_utils import append_jsonl, latest_by, read_jsonl, write_csv, write_jsonl


def test_jsonl_roundtrip_and_truncated_line(tmp_path):
    p = tmp_path / "a" / "x.jsonl"
    rows = [{"video_id": "v1", "x": 1}, {"video_id": "v2", "x": [1.5, "é"]}]
    assert write_jsonl(p, rows) == 2
    assert read_jsonl(p) == rows
    append_jsonl(p, [{"video_id": "v1", "x": 3}])
    with open(p, "a", encoding="utf-8") as f:
        f.write('{"video_id": "v3", "x"')        # cut by a disconnect
    got = read_jsonl(p)
    assert len(got) == 3
    assert latest_by(got)["v1"]["x"] == 3
    assert read_jsonl(tmp_path / "missing.jsonl") == []
    assert not (tmp_path / "a" / "x.jsonl.part").exists()


def test_write_csv(tmp_path):
    p = tmp_path / "t.csv"
    write_csv(p, [{"a": 1, "b": 0.1234567891}, {"a": 2, "c": [1, 2]}])
    lines = p.read_text(encoding="utf-8").splitlines()
    assert lines[0] == "a,b,c"
    assert lines[1] == "1,0.123457,"
    assert json.loads(lines[2].split(",", 2)[2].strip('"').replace('""', '"')) == [1, 2]


def test_uniform_segments():
    s = uniform_segments(100.0, 4)
    assert [(x["ts"], x["te"]) for x in s] == [(0, 25), (25, 50), (50, 75), (75, 100)]
    assert all(x["conf"] == 1.0 for x in s)
    assert [(x["ts"], x["te"]) for x in uniform_segments(10.0, 0)] == [(0, 10.0)]
    s3 = uniform_segments(10.0, 3)
    assert s3[-1]["te"] == 10.0 and s3[1]["ts"] == pytest.approx(10 / 3)
