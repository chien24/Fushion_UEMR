"""End-to-end smoke test of uemr2.experiments on a tiny fake world (CPU)."""

import numpy as np
import pytest

from uemr2 import experiments as X
from uemr2.score import fold_split
from uemr2.windows import dense


def _unit(x):
    return (x / np.linalg.norm(x, axis=-1, keepdims=True)).astype(np.float32)


def world(seed=0):
    rng = np.random.default_rng(seed)
    val, dev = [f"v{i:02d}" for i in range(12)], [f"d{i}" for i in range(6)]
    allv = val + dev
    dur = {v: float(rng.uniform(30, 90)) for v in allv}
    gt = {v: sorted((a, a + float(rng.uniform(3, 12))) for a in rng.uniform(0, dur[v] - 15, 3)) for v in allv}
    parts = {}

    def rows(segs_by):
        vid = np.array([v for v in allv for _ in segs_by[v]])
        s = np.array([x for v in allv for x in segs_by[v]], np.float32)
        n = len(vid)
        return {"video_id": vid, "ts": s[:, 0], "te": s[:, 1], "e": _unit(rng.normal(size=(n, 8))),
                "q": _unit(rng.normal(size=(n, 4))), "h": _unit(rng.normal(size=(n, 8)))}

    parts["gt"] = rows(gt)
    parts["uniav_pred"] = rows({v: [(a + 1, b + 1) for a, b in gt[v][:2]] for v in allv})
    parts["global"] = rows({v: [(0.0, dur[v])] for v in allv})
    for name in ("uni_chunk", "uni_M", "uni_M_gt", "uniav_shift", "gt_noise_03", "gt_noise_05", "gt_noise_07", "gt_noise_09"):
        parts[name] = rows({v: [(a, a + 6.0) for a in rng.uniform(0, dur[v] - 6, 3)] for v in allv})
    windows = {w: rows({v: [tuple(x) for x in dense(dur[v], w)] for v in allv}) for w in (4, 8, 16)}

    def text(vs):
        vid = [v for v in vs for _ in gt[v]]
        g = np.array([x for v in vs for x in gt[v]], np.float32)
        n = len(vid)
        return {"omni": _unit(rng.normal(size=(n, 8))), "iv2": _unit(rng.normal(size=(n, 4))), "video_id": np.array(vid),
                "gt_ts": g[:, 0], "gt_te": g[:, 1], "caption_id": np.array([f"c{i}" for i in range(n)])}

    return {"videos": {"val": val, "dev": dev, "all": allv}, "text": {"val": text(val), "dev": text(dev)},
            "parts": parts, "windows": windows}


def test_pipeline_runs_and_beta_zero_is_column_b():
    w = world()
    z = X.zstats(w)
    assert ("A", "uniav_pred") in z and ("B", "gap_multi") in z
    fq = X.fold_of_queries(w, fold_split(w["videos"]["val"]))
    b_rank, _ = X.score_part(w, z, "val", "val", "uniav_pred")
    chosen, out = X.cv_beta(w, z, "val", fq, "A", ("uniav_pred", "gt"), betas=(0.0,))
    assert chosen == [0.0, 0.0] and np.array_equal(out["uniav_pred"][0], b_rank)
    chosen, out = X.cv_beta(w, z, "val", fq, "B", ("uniav_pred", "gt", "global"))
    assert len(out["gt"][0]) == len(w["text"]["val"]["video_id"])
    ch, rk, hit, nwin = X.cv_e3(w, z, "val", fq, "A", "gap", gammas=(-np.inf,))
    assert np.array_equal(rk, b_rank) and set(nwin) == {16, 8, 4, "multi"}
    ch, rk, hit, nwin = X.cv_e3(w, z, "all", fq, "B", "dense")
    assert rk.max() <= len(w["videos"]["all"]) and hit.dtype == bool
    chosen, out = X.cv_beta_ab(w, z, "val", fq, ("uniav_pred",))
    assert len(chosen) == 2 and len(chosen[0]) == 2
    d_rank, _ = X.score_part(w, z, "val", "val", "uniav_pred", "A", alone=True)
    assert d_rank.min() >= 1
    r = X.text_to_clip(w, "val", "A")
    assert 0 <= r["R@1"] <= 100
    miss = X.miss_labels(w)
    assert miss.dtype == bool and len(miss) == len(fq)


def test_tables_and_gates():
    pytest.importorskip("pandas")
    from uemr2.tables import dev_reference, val_tables
    w = world(1)
    w["text"]["val"]["dup"] = np.random.default_rng(0).integers(0, 40, len(w["text"]["val"]["video_id"])).astype(float)
    z = X.zstats(w)
    ref = dev_reference(w, z)
    assert (ref["ghi chú"] == "dev: rò rỉ, tham khảo").all()
    folds = fold_split(w["videos"]["val"])
    for g in ("val", "all"):
        t = val_tables(w, z, folds, g)
        assert {"A2", "E3", "B", "pt"} <= set(t["gates"]) and len(t["noise"]) == 6
        b = t["main"][t["main"]["thí nghiệm"] == "(b) Omni ft"]
        assert len(b) == 7
