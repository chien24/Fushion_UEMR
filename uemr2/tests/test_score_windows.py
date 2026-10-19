"""§3.4 tests: windows, seconds <-> grid, beta = 0 equals column (b), E3 with gamma -> -inf equals R6, the CV helpers."""

import numpy as np

from omni_retrieval.search import SegmentIndex, tiou_vec
from uemr2.score import cv_pick, e3, fold_split, layout, oof, ranks, video_max
from uemr2.windows import dense, gap_mask, outside_fraction
from uniav_app.timegrid import grid_to_sec, sec_to_grid


def _unit(x):
    return (x / np.linalg.norm(x, axis=-1, keepdims=True)).astype(np.float32)


def test_dense_covers_the_video_with_step_w_half():
    for dur in (3.0, 7.9, 10.0, 183.4):
        for w in (4, 8, 16):
            win = dense(dur, w)
            assert win[0, 0] == 0 and abs(win[-1, 1] - dur) < 1e-4 and (win >= 0).all() and (win <= dur + 1e-4).all()
            assert np.allclose(np.diff(win[:, 0]), w / 2)
            lens = win[:, 1] - win[:, 0]
            assert (lens <= w + 1e-5).all() and (lens > min(w / 2, dur) - 1e-5).all()
            assert (win[1:, 0] <= win[:-1, 1]).all()                  # no hole


def test_gap_windows_are_mostly_outside_r6():
    win = dense(60.0, 8)
    r6 = np.array([[5.0, 20.0], [30.0, 33.0]])
    m = gap_mask(win, r6)
    assert m.any() and not m.all()
    assert (outside_fraction(win[m], r6) >= 0.5).all() and (outside_fraction(win[~m], r6) < 0.5).all()
    assert np.allclose(outside_fraction([[0, 10]], [[2, 4], [3, 6]]), 0.6)
    assert gap_mask(win, np.zeros((0, 2))).all()
    # a 4 s window inside a 60 s R6 step: tIoU 0.07 but nothing outside -> not a gap window
    assert not gap_mask([[10.0, 14.0]], [[0.0, 60.0]]).any()
    # first registered definition (tIoU = 0) is the stricter subset
    v1 = gap_mask(win, r6, tiou_max=0.0)
    assert not (v1 & ~m).any()
    t = tiou_vec(win[v1, None, 0], win[v1, None, 1], r6[None, :, 0], r6[None, :, 1])
    assert (t == 0).all()


def test_seconds_grid_roundtrip_under_one_step():
    for n in (37, 256, 611):
        t = np.random.default_rng(n).uniform(0, n, 200)
        assert np.abs(grid_to_sec(sec_to_grid(t, n), n) - t).max() < n / 256 * 1e-6 + 1e-9


def _world(seed=0, v=12, d=16):
    rng = np.random.default_rng(seed)
    vids = np.array(sum([[f"v{i}"] * int(k) for i, k in enumerate(rng.integers(2, 6, v))], []))
    ts = rng.uniform(0, 50, len(vids)).astype(np.float32)
    te = ts + rng.uniform(1, 10, len(vids)).astype(np.float32)
    e, x = _unit(rng.normal(size=(len(vids), d))), _unit(rng.normal(size=(len(vids), d)))
    tq, tx = _unit(rng.normal(size=(40, d))), _unit(rng.normal(size=(40, d)))
    videos = [f"v{i}" for i in range(v)]
    pos = rng.integers(0, v, 40)
    return vids, ts, te, e, x, tq, tx, videos, pos


def test_beta_zero_is_column_b():
    vids, ts, te, e, x, tq, tx, videos, pos = _world()
    idx = SegmentIndex(e, vids, ts, te)
    ref, _ = idx.score_matrix(tq)                                  # column (b) as omni_retrieval computes it
    ref_rank = ranks(ref[:, [idx.video_index[v] for v in videos]], pos)
    pad = layout(vids, videos)
    s, _ = video_max([(tq, e, 0.3, 0.07, 1.0), (tx, x, 0.1, 0.05, 0.0)], pad)   # z-scored, beta = 0
    assert np.array_equal(ranks(s, pos), ref_rank)
    s1, _ = video_max([(tq, e, 0.3, 0.07, 1.0), (tx, x, 0.1, 0.05, 0.5)], pad)
    assert not np.array_equal(s1, s)


def test_e3_gamma_minus_inf_is_r6_and_argmax_rows():
    vids, ts, te, e, x, tq, tx, videos, pos = _world(1)
    pad = layout(vids, videos)
    s6, a6 = video_max([(tq, e, 0, 1, 1)], pad)
    sw, aw = video_max([(tx, x, 0, 1, 1)], pad)
    s, a, w = e3(s6, a6, sw, aw, -np.inf)
    assert np.array_equal(ranks(s, pos), ranks(s6, pos)) and not w.any() and np.array_equal(a, a6)
    c = tq @ e.T
    for qi in (0, 7):
        for vi, v in enumerate(videos):
            rows = np.flatnonzero(vids == v)
            assert a6[qi, vi] == rows[np.argmax(c[qi, rows])]


def test_cv_pick_tie_and_oof():
    fold = np.array([0, 0, 1, 1])
    cands = [np.array([1, 2, 1, 2]), np.array([1, 2, 2, 2]), np.array([1, 1, 2, 2])]
    pick = cv_pick(cands, fold)                                    # fold 0: cand 2 (2 hits); fold 1: cand 0 (tie 0/2 -> first)
    assert pick == [2, 0]
    assert oof(cands, pick, fold).tolist() == [1, 2, 2, 2]         # fold 0 <- cand 0, fold 1 <- cand 2
    f1, f2 = fold_split([f"v{i}" for i in range(394)])
    assert len(f1) == len(f2) == 197 and not set(f1) & set(f2) and fold_split([f"v{i}" for i in range(394)]) == (f1, f2)
