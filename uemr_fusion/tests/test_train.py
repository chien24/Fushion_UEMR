"""End-to-end training on the fake cache (CPU, a few seconds): files, val report, alpha-0 parity, overfit."""

import json

import numpy as np
import pytest
import torch

from omni_retrieval.evaluate import video_ranks
from uemr_fusion.data import FusionSplit
from uemr_fusion.evaluate import build_index, fuse, raw_vectors
from uemr_fusion.model import ContextFusion, FusionConfig
from uemr_fusion.train import main

SMALL = ["--d-h", "8", "--heads", "4", "--ff", "32", "--batch", "4", "--device", "cpu"]


def test_alpha_zero_reproduces_column_b(fusion_cache):
    """Plan C6.1: an untrained fusion (alpha = 0) ranks val exactly like the raw Omni vectors."""
    val = FusionSplit(fusion_cache, "val", "uniav_pred")
    model = ContextFusion(FusionConfig(d=val.d, d_h=8, heads=4, ff=32))
    text = val.text()
    r_c = video_ranks(build_index(val, fuse(model, val)), text)["ranks"]
    r_b = video_ranks(build_index(val, raw_vectors(val)), text)["ranks"]
    assert np.array_equal(r_c, r_b)


def test_train_end_to_end(fusion_cache, tmp_path):
    run = tmp_path / "runs" / "D2"
    assert main(["--cache", str(fusion_cache), "--partition", "uniav_pred", "--out", str(run), "--epochs", "3",
                 "--tau-s", "0.02", "--lam", "0.5", *SMALL]) == 0
    for f in ("config.json", "history.jsonl", "best.pt", "val.json", "ranks.npz"):
        assert (run / f).is_file(), f
    hist = [json.loads(l) for l in (run / "history.jsonl").read_text().splitlines()]
    assert [h["epoch"] for h in hist] == [0, 1, 2, 3] and "alpha" in hist[-1]
    rep = json.loads((run / "val.json").read_text())
    for key in ("val394 (b)", "val394 (c)", "all498 (b)", "all498 (c)", "val394 bootstrap (c)-(b) by video",
                "by_duration", "alpha", "best_epoch"):
        assert key in rep, key
    assert rep["val394 (c)"]["queries"] == 10 and rep["val394 (c)"]["videos"] == 5
    assert rep["all498 (c)"]["videos"] == 8                               # val + dev distractors
    ranks = np.load(run / "ranks.npz")
    assert ranks["ranks_c_val394"].shape == (10,)
    ck = torch.load(run / "best.pt", weights_only=False)
    assert ck["config"]["d"] == 16 and 1 <= ck["epoch"] <= 3
    # a finished run is skipped
    before = (run / "val.json").read_text()
    main(["--cache", str(fusion_cache), "--partition", "uniav_pred", "--out", str(run), *SMALL])
    assert (run / "val.json").read_text() == before


def test_ablation_flags_and_global_partition(fusion_cache, tmp_path):
    for name, extra in (("glob", ["--partition", "global"]),
                        ("full", ["--use-global", "--shuffle-z"]),
                        ("cross_only", ["--no-self", "--delta", "0"]),
                        ("tau", ["--learn-tau", "--lam", "0"])):
        run = tmp_path / name
        assert main(["--cache", str(fusion_cache), "--out", str(run), "--epochs", "1", "--no-val", *extra, *SMALL]) == 0
        assert (run / "best.pt").is_file(), name
    args = json.loads((tmp_path / "glob" / "config.json").read_text())["args"]
    assert args["lam"] == 0.0                                             # one vector per video: L_evt off


def test_overfit_mode(fusion_cache, tmp_path):
    main(["--cache", str(fusion_cache), "--out", str(tmp_path / "of"), "--overfit", "6", "--steps", "120",
          "--lr", "3e-3", "--alpha-lr", "3e-2", "--dropout", "0", *SMALL])
    res = json.loads((tmp_path / "of" / "overfit.json").read_text())
    assert res["videos"] == 6 and res["captions"] == 12 and res["steps"] == 120
    assert res["R@1 after"] >= res["R@1 before"] and res["loss last 10"] < res["loss first 10"]


def test_seed_mean_bootstrap(tmp_path):
    from uemr_fusion.evaluate import seed_mean_bootstrap
    cap, vid = np.array(["q0", "q1", "q2", "q3"]), np.array(["X", "X", "Y", "Z"])
    b = np.array([1, 2, 1, 3])                                   # R@1 (b) 50
    paths = []
    for name, c in (("D3", [1, 1, 1, 3]), ("D3_seed1", [1, 1, 1, 1])):   # R@1 (c) 75 and 100
        (tmp_path / name).mkdir()
        arrays = {"caption_id": cap, "video_id": vid}
        for gal in ("val394", "all498"):
            arrays[f"ranks_b_{gal}"], arrays[f"ranks_c_{gal}"] = b, np.array(c)
        np.savez(tmp_path / name / "ranks.npz", **arrays)
        paths.append(tmp_path / name / "ranks.npz")
    rows = seed_mean_bootstrap(paths, n_boot=200)
    by = {(r["gallery"], r["run"]): r for r in rows}
    assert len(rows) == 6 and by["val394", "D3"]["(c) - (b)"] == pytest.approx(25.0)
    mean = by["val394", "mean of 2"]
    assert mean["R@1 (c)"] == pytest.approx(87.5) and mean["(c) - (b)"] == pytest.approx(37.5)
    assert mean["ci_low"] <= 37.5 <= mean["ci_high"] and mean["videos"] == 3
    np.savez(tmp_path / "D3_seed1" / "ranks.npz", caption_id=cap[::-1], video_id=vid,
             **{f"ranks_{k}_{g}": b for k in "bc" for g in ("val394", "all498")})
    with pytest.raises(ValueError, match="query order"):
        seed_mean_bootstrap(paths)
