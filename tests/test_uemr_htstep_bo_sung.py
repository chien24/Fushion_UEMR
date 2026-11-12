"""Unit tests of the HT-Step add-on plan (docs/uemr/HTSTEP_BO_SUNG.md §10), CPU, synthetic data."""

import numpy as np
import pytest

from omni_retrieval.search import SegmentIndex
from uemr_htstep import bo_sung as bs
from uemr_htstep._unused import bo_sung as un     # rows / columns the final plan does not run (UEMR_CHOT_00 §5); still tested
from uemr_htstep import metrics as hm
from uemr_htstep import score as sc

DUR = {"a": 100.0, "b": 37.5, "c": 12.0}
GT = {"a": [[5.0, 9.0], [20.0, 31.5], [60.0, 61.2]], "b": [[1.0, 4.0], [10.0, 30.0]], "c": [[0.0, 12.0]]}


def test_r4_k_dmin_and_tiling():
    k = {"a": 7, "b": 4, "c": 3}
    for seed in un.R4_SEEDS:
        part = un.r4(k, DUR, seed, 2.0)
        for v, segs in part.items():
            assert len(segs) == k[v]
            assert segs[0][0] == 0.0 and abs(segs[-1][1] - DUR[v]) < 1e-9
            assert all(abs(segs[i][1] - segs[i + 1][0]) < 0.011 for i in range(len(segs) - 1))
            assert min(te - ts for ts, te in segs) >= 2.0 - 0.011
    assert un.r4(k, DUR, 0, 2.0) != un.r4(k, DUR, 1, 2.0)


def test_r5_k_and_clips_inside():
    rng = np.random.default_rng(0)
    feats = {v: (rng.standard_normal((256, 16)).astype(np.float32), np.linspace(0, DUR[v], 256)) for v in DUR}
    k = {"a": 5, "b": 3, "c": 2}
    part, info = un.r5(feats, k, DUR, 0, 1.0)
    for v, segs in part.items():
        assert len(segs) == k[v]
        assert all(0 <= ts < te <= DUR[v] and abs((te - ts) - 2.0) < 0.011 for ts, te in segs)
    assert not info["not converged"]


def test_noise_hits_target_and_keeps_k():
    rng = np.random.default_rng(1)
    gt, dur = {}, {}
    for i in range(40):
        d = float(rng.uniform(60, 300))
        starts = np.sort(rng.uniform(0, d - 20, 5))
        gt[f"v{i}"] = [[round(float(s), 2), round(float(s + rng.uniform(2, 15)), 2)] for s in starts]
        dur[f"v{i}"] = d
    dirs = un.noise_dirs(gt, 0)
    for t in un.NOISE_TARGETS:
        a, got, part = un.calibrate(gt, dur, dirs, t, 0.1)
        assert abs(got - t) <= un.NOISE_TOL
        assert all(len(part[v]) == len(gt[v]) for v in gt)
        assert all(0 <= s[0] < s[1] <= dur[v] for v in part for s in part[v])
    assert un.paired_tiou(gt, gt) == pytest.approx(1.0)


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


def test_article_folds():
    art = {f"v{i}": i % 7 for i in range(50)}
    nq = {f"v{i}": 1 + i % 4 for i in range(50)}
    f = un.article_folds(art, nq, 0)
    assert set(f) == set(art)
    assert not ({art[v] for v, x in f.items() if x == "A"} & {art[v] for v, x in f.items() if x == "B"})
    tot = {x: sum(nq[v] for v, y in f.items() if y == x) for x in "AB"}
    assert abs(tot["A"] - tot["B"]) <= max(sum(nq[v] for v in art if art[v] == a) for a in range(7))


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


def test_global_bonus_gamma_zero_is_r8():
    vids, idx, caps, text = _toy()
    g = idx.mean_pooled("g")
    for m in hm.MODES:
        base = hm.text_to_video(idx, text, caps, m)
        zero = hm.text_to_video(idx, text, caps, m, bonus=un.global_bonus(text["emb"], idx, g, 0.0))
        assert np.array_equal(base["ranks"], zero["ranks"]) and np.array_equal(base["vcmr"], zero["vcmr"])


def test_boot_with_seeds_on_either_side():
    rng = np.random.default_rng(5)
    a, b = rng.integers(1, 4, (3, 50)), rng.integers(1, 4, 50)
    vids = np.repeat(np.arange(10), 5)
    ab, ba = sc.boot(a, b, vids), sc.boot(b, a, vids)
    assert ab["A - B"] == pytest.approx(-ba["A - B"]) and ab["ci_low"] == pytest.approx(-ba["ci_high"])


def test_context_fusion_split_masks_same_text_and_cross_fits():
    import torch
    from uemr_htstep._unused import cf

    vids, idx, caps, text = _toy()
    feats = {v: (np.random.default_rng(6).standard_normal((16, 12)).astype(np.float32), np.linspace(0, 30, 16).astype(np.float32)) for v in vids}
    dur = {"a": 30.0, "b": 20.0, "c": 10.0}
    g = idx.mean_pooled("g")
    ds = cf.make_split(idx, g, feats, dur, caps, text["emb"], vids, "all")
    # q0 ("x", video a) and q1 ("x", video b): each other's video is a right answer
    assert ds.same.tolist() == [[True, True, False], [True, True, False], [False, False, True]]
    hp = {"epochs": 1, "batch": 2, "lr": 1e-4, "wd": 0.01, "warmup": 0.05, "alpha_lr": 1e-3, "tau": 0.05, "tau_s": 0.05, "lam": 0.5,
          "d_h": 16, "heads": 2, "delta": 5.0, "dropout": 0.0, "ff": 32}
    folds = {"a": "A", "b": "B", "c": "B"}
    out, rep = cf.cross_fit(idx, g, feats, dur, caps, text["emb"], folds, hp, 0, "cpu", log=lambda *_: None)
    assert sorted(out.videos.tolist()) == vids
    assert out.emb.shape[0] == len(idx.emb) + len(vids)            # every segment + c_g of every video
    assert set(rep) == {"model A -> fold B", "model B -> fold A"}
    assert torch.isfinite(torch.tensor(out.emb)).all()
