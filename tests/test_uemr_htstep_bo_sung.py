"""Unit tests of the HT-Step add-on plan (docs/uemr/HTSTEP_BO_SUNG.md §10), CPU, synthetic data."""

import numpy as np
import pytest

from omni_retrieval.search import SegmentIndex
from uemr_htstep import bo_sung as bs
from uemr_htstep import metrics as hm
from uemr_htstep import score as sc

DUR = {"a": 100.0, "b": 37.5, "c": 12.0}
GT = {"a": [[5.0, 9.0], [20.0, 31.5], [60.0, 61.2]], "b": [[1.0, 4.0], [10.0, 30.0]], "c": [[0.0, 12.0]]}


def test_r1_is_the_normalised_mean():
    rng = np.random.default_rng(2)
    emb = rng.standard_normal((5, 8)).astype(np.float32)
    emb /= np.linalg.norm(emb, axis=1, keepdims=True)
    idx = SegmentIndex(emb, ["a", "a", "a", "b", "b"], [0, 1, 2, 0, 1], [1, 2, 3, 1, 2])
    r1 = idx.mean_pooled("r1")
    m = emb[:3].mean(0)
    assert np.allclose(r1.emb[0], m / np.linalg.norm(m), atol=1e-6) and r1.n_videos == 2


def test_zscore_uses_no_label_and_c1z_picks_by_fill():
    rng = np.random.default_rng(3)
    sb, ss = rng.normal(0.05, 0.04, (6, 4)), rng.normal(0.6, 0.06, (6, 4))
    zb, zs = bs.zstats(sb), bs.zstats(ss)
    assert zb == bs.zstats(sb[::-1])            # a permutation of the queries (their labels) changes nothing
    fill = np.array([False, True, False, True])
    out = bs.c1z(sb, ss, fill, zb, zs)
    assert np.allclose(out[:, 1], (ss[:, 1] - zs[0]) / zs[1]) and np.allclose(out[:, 0], (sb[:, 0] - zb[0]) / zb[1])


def _toy():
    rng = np.random.default_rng(4)
    vids = ["a", "b", "c"]
    emb = rng.standard_normal((7, 8)).astype(np.float32)
    emb /= np.linalg.norm(emb, axis=1, keepdims=True)
    idx = SegmentIndex(emb, ["a", "a", "a", "b", "b", "c", "c"], [0, 10, 20, 0, 10, 0, 5], [10, 20, 30, 10, 20, 5, 10])
    caps = [{"caption_id": "q0", "video_id": "a", "sentence": "x", "headline": "x", "partial": "Full Match", "gt_ts": 10.0, "gt_te": 20.0},
            {"caption_id": "q1", "video_id": "b", "sentence": "x", "headline": "x", "partial": "Partial Match", "gt_ts": 0.0, "gt_te": 10.0},
            {"caption_id": "q2", "video_id": "c", "sentence": "y", "headline": "y", "partial": "Full Match", "gt_ts": 0.0, "gt_te": 5.0}]
    q = rng.standard_normal((3, 8)).astype(np.float32)
    text = {"emb": q, "caption_id": np.array(["q0", "q1", "q2"]), "video_id": np.array(["a", "b", "c"]),
            "gt_ts": np.array([10.0, 0.0, 0.0]), "gt_te": np.array([20.0, 10.0, 5.0])}
    return vids, idx, caps, text


def test_boot_with_seeds_on_either_side():
    rng = np.random.default_rng(5)
    a, b = rng.integers(1, 4, (3, 50)), rng.integers(1, 4, 50)
    vids = np.repeat(np.arange(10), 5)
    ab, ba = sc.boot(a, b, vids), sc.boot(b, a, vids)
    assert ab["A - B"] == pytest.approx(-ba["A - B"]) and ab["ci_low"] == pytest.approx(-ba["ci_high"])
