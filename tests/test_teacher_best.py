"""Teacher set of UniAV from Omni best (docs/uemr_chot/UEMR_CHOT_01 §2, §4). CPU, synthetic data."""

import os
import sys

import numpy as np
import pytest

from uemr_htstep import teacher_best as tb

ANN = {"a": {"annotations": [{"segment": [1.0, 5.0], "id": 7, "partial": "Full Match"},
                             {"segment": [9.0, 9.0], "id": 8, "partial": "Partial Match"},      # end <= start
                             {"segment": [20.0, 300.0], "id": 9, "partial": "Partial Match"},   # clip window clamped to the mp4
                             {"segment": [30.0, 40.0], "id": 10, "partial": "Full Match"}]},    # clip not cut
       "b": {"annotations": [{"segment": [2.0, 4.0], "id": 11, "partial": "Full Match"}]},      # no clip, no mp4
       "c": {"annotations": [{"segment": [0.5, 3.5], "id": 12, "partial": "Full Match"}]}}      # cached (dev-1 store)
CLIPS = [{"video_id": "a", "segment": [1.0, 5.0], "window": [1.0, 5.0], "clip": "clips/a/0.npz"},
         {"video_id": "a", "segment": [20.0, 300.0], "window": [20.0, 250.0], "clip": "clips/a/1.npz"},
         {"video_id": "c", "segment": [0.5, 3.5], "window": [0.5, 3.5], "clip": "clips/c/0.npz"}]


def test_sources():
    segs = tb.assign(tb.gt_segments(ANN, ["a", "b", "c"]), CLIPS, {"c__0.50_3.50"}, {"a"})
    assert [s["id"] for s in segs] == ["a#a0", "a#a1", "a#a2", "a#a3", "b#a0", "c#a0"]
    assert [s["source"] for s in segs] == ["clip", "missing", "mp4", "mp4", "missing", "cache"]
    assert tb.record(segs[0])["clip"] == "clips/a/0.npz" and "video" not in tb.record(segs[0])
    assert tb.record(segs[2]) | {"conversations": 0} == {"id": "a#a2", "type": "retrieval", "conversations": 0, "video": "a.mp4",
                                                         "timestamps": [20.0, 300.0]}
    assert len(tb.record(segs[0])["conversations"]) == 1          # no caption turn


def test_write_is_what_hanli_reads(tmp_path):
    segs = tb.gt_segments(ANN, ["a", "b"])
    rng = np.random.default_rng(0)
    vec = {"a#a0": rng.standard_normal(3584) * 3, "a#a3": rng.standard_normal(3584)}
    rep = tb.write(segs, vec, tmp_path)
    assert set(rep["files"]) == {"a.npz"} and rep["segments"] == 2 and rep["missing"] == ["a#a1", "a#a2", "b#a0"]
    with np.load(tmp_path / "a.npz") as z:
        assert set(z.files) == {"emb", "seg_idx", "step_id", "segment", "partial"}        # the keys of the old teacher files
        assert z["emb"].dtype == np.float16 and z["emb"].shape == (2, 3584) and z["seg_idx"].tolist() == [0, 3]
        assert np.allclose(np.linalg.norm(z["emb"].astype(np.float32), axis=1), 1, atol=1e-3)
    repo = os.environ.get("UNIAV_REPO", "")
    if not repo:
        pytest.skip("set UNIAV_REPO to a checkout of HanLi-Retriever: the loader itself is not run")
    sys.path.insert(0, repo)
    from libs.datasets.helper import load_omni_segments
    # the dataset dropped annotation 1; slots = position among the kept ones
    vectors, ok = load_omni_segments(str(tmp_path / "a.npz"), {0: 0, 2: 1, 3: 2})
    assert ok[:3].tolist() == [True, False, True] and not ok[3:].any()
    assert np.allclose(vectors[2], vec["a#a3"] / np.linalg.norm(vec["a#a3"]), atol=2e-3)
    assert not load_omni_segments(str(tmp_path / "b.npz"), {0: 0})[1].any()             # a video without a file: no teacher
