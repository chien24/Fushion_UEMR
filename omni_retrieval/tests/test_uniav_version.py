"""UniAV output hash and install (docs/UEMR_VIEC_TIEP_THEO.md §A5), and the Config paths of a variant."""

import json

import numpy as np
import pytest

from omni_retrieval.config import Config
from omni_retrieval.uniav_version import fingerprint, hash_of, install


def _write(d, lines, z_videos):
    (d / "Z").mkdir(parents=True)
    (d / "segments_pred.jsonl").write_text("".join(json.dumps(r) + "\n" for r in lines))
    for v in z_videos:
        np.savez(d / "Z" / f"{v}.npz", z=np.full((4, 2), len(v), np.float16), t_sec=np.arange(4, dtype=np.float32))


LINES = [{"video_id": "g1", "ts": 0.0, "te": 5.0, "conf": 0.9}, {"video_id": "g1", "ts": 6.0, "te": 9.0, "conf": 0.5},
         {"video_id": "t1", "ts": 1.0, "te": 4.0, "conf": 0.7}]


def test_hash_ignores_how_the_output_is_split(tmp_path):
    one, a, b = tmp_path / "one", tmp_path / "a", tmp_path / "b"
    _write(one, LINES, ["g1", "t1"])
    _write(a, LINES[:2], ["g1"])
    _write(b, LINES[2:], ["t1"])
    fp = fingerprint([one])
    assert fp == fingerprint([a, b]) and (fp["videos"], fp["segments"], fp["z_files"]) == (2, 3, 2)
    changed = tmp_path / "changed"
    _write(changed, [dict(LINES[0], conf=0.8), *LINES[1:]], ["g1", "t1"])
    assert fingerprint([changed])["hash"] != fp["hash"]
    with pytest.raises(ValueError):                                     # a video delivered twice
        fingerprint([one, a])


def test_install_splits_and_is_idempotent(tmp_path):
    src = tmp_path / "delivered"
    _write(src, LINES, ["g1", "t1"])
    out = install(src, tmp_path / "uniav_cache", {"": ["g1"], "train": ["t1"]})
    h = out["hash8"]
    gal, tr = out["dirs"][""], out["dirs"]["train"]
    assert gal.name == f"ft_{h}" and tr.name == f"ft_train_{h}"
    assert [json.loads(l)["video_id"] for l in (gal / "segments_pred.jsonl").read_text().splitlines()] == ["g1", "g1"]
    assert sorted(p.name for p in (tr / "Z").iterdir()) == ["t1.npz"]
    assert hash_of(gal) == hash_of(tr) == h
    mtime = (gal / "SOURCE.json").stat().st_mtime_ns
    install(src, tmp_path / "uniav_cache", {"": ["g1"], "train": ["t1"]})
    assert (gal / "SOURCE.json").stat().st_mtime_ns == mtime            # same hash: kept
    with pytest.raises(ValueError):                                     # a listed video without output
        install(src, tmp_path / "uniav_cache2", {"": ["g1", "g2"]})


def test_config_variant_paths():
    old = Config(tag="ft", cache_root="/c")
    new = Config(tag="ft", cache_root="/c", variant="ab12cd34")
    assert (str(old.partitions_dir), str(old.out_dir)) == ("/c/common/partitions", "/c/ft")
    assert str(new.partitions_dir).replace("\\", "/") == "/c/common/partitions_ab12cd34"
    assert str(new.out_dir).replace("\\", "/") == "/c/ft/ab12cd34"
    assert new.av_store == old.av_store and new.subset_path == old.subset_path        # raw store shared
    assert new.segs_todo.name == "segs_todo_ft_ab12cd34.jsonl"
