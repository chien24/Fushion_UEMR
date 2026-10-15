"""configs/final_protocol.json is the one source of the frozen values; local_npz; miss rate; theta rule."""

import numpy as np
import pytest

from omni_retrieval import partitions_uemr as pu
from omni_retrieval.collect import MissingEmbeddings, local_npz
from omni_retrieval.protocol import P


def test_protocol_values_reach_the_code():
    assert (pu.D_MIN, pu.CHUNK_SEC, pu.HALF_WINDOW, pu.MAX_ITER) == (2.0, 14.0, 1.0, 100)
    assert (pu.NOISE_TARGETS, pu.NOISE_TOL, pu.MISS_TIOU) == ((0.9, 0.7, 0.5, 0.3), 0.02, 0.3)
    assert (P["scoring"]["agg"], P["scoring"]["beta"]) == ("max", 0.0)
    assert P["eval"]["bootstrap"] == {"n": 1000, "seed": 0, "unit": "video", "ci": 95}
    assert P["fusion"]["run_extra_fusion"] is False
    from omni_retrieval.search import POOLED
    assert P["partitions"]["pooled"] == POOLED


def test_uniav_app_defaults_match_protocol():
    """``uniav_app`` stays independent of ``omni_retrieval`` (own runtime), so its defaults are checked here."""
    from uniav_app.config import THETA_GRID, Config
    post = P["uniav"]["postprocess"]
    c = Config()
    assert (c.k_max, c.d_min, c.merge_tiou) == (post["k_max"], post["d_min"], post["merge_tiou"])
    assert list(THETA_GRID) == post["theta"]["grid"]


def test_theta_rule_on_the_logged_dev_sweep():
    """K_pred on dev of the current UniAV ([UA c22]) -> 0.40, as reported; a tie goes to the smaller theta."""
    from uniav_app.evaluate import choose_theta
    k_dev = {0.25: 17.490, 0.30: 12.298, 0.35: 9.029, 0.40: 6.952, 0.45: 5.087, 0.50: 2.817}
    rows = [{"split": "dev", "theta": t, "mean_K_pred": k, "mean_K_GT": 7.433} for t, k in k_dev.items()]
    rows += [{"split": "val", "theta": t, "mean_K_pred": 7.69, "mean_K_GT": 7.69} for t in k_dev]   # never used
    assert choose_theta(rows, "dev") == (P["reference_current_uniav"]["theta"], "dev")


def test_miss_rate():
    gt = {"A": [[0.0, 10.0], [20.0, 30.0]], "B": [[0.0, 4.0]]}
    part = {"A": [[0.0, 9.0]], "B": [[3.0, 20.0]]}                     # A: 0.9, 0.0 ; B: 1/20
    assert pu.miss_rate(part, gt) == pytest.approx(100 * 2 / 3)
    assert pu.miss_rate(gt, gt) == 0 and pu.miss_rate({}, gt) == 100


def _seg_npz(path, parts):
    np.savez(path, partition=np.array(parts), emb=np.zeros((len(parts), 2), np.float16))


def test_local_npz_checks_content_and_rebuilds(tmp_path):
    src, dst = tmp_path / "drive", tmp_path / "local"
    src.mkdir()
    _seg_npz(src / "segments.npz", ["gt", "uniav_pred"])
    np.savez(src / "text.npz", emb=np.zeros((1, 2)))
    assert local_npz(src, dst, need=("gt",)) == dst and (dst / "text.npz").is_file()
    with pytest.raises(MissingEmbeddings):                              # stale copy, no way to rebuild
        local_npz(src, dst, need=("gt_noise_09",))
    calls = []

    def rebuild(path):
        calls.append(path)
        _seg_npz(path, ["gt", "uniav_pred", "gt_noise_09"])

    local_npz(src, dst, need=("gt_noise_09",), rebuild=rebuild)
    assert len(calls) == 1
    with np.load(src / "segments.npz") as b:                            # the Drive copy is fixed too
        assert "gt_noise_09" in set(b["partition"].tolist())
    (src / "segments.npz").write_text("not a zip")                      # broken file on Drive, good local copy
    local_npz(src, dst, need=("gt",), rebuild=rebuild)
    assert len(calls) == 1
    with pytest.raises(MissingEmbeddings):                              # a rebuild that cannot provide the partition
        local_npz(src, tmp_path / "other", need=("gt_noise_03",), rebuild=lambda p: _seg_npz(p, ["gt"]))


def test_p5_curve_helpers_reproduce_the_report_in_any_order():
    curve = [(0.3, 20.66), (1.0, 37.52), (0.5, 28.42), (0.9, 36.73), (0.7, 34.29)]   # shuffled on purpose
    assert pu.equivalent_tiou(curve, 24.88) == pytest.approx(0.409, abs=1e-3)       # R6 ~ noisy GT at 0.41
    assert pu.equivalent_tiou(curve, 18.91) == pytest.approx(0.255, abs=1e-3)       # break-even with uni_M_gt
    assert pu.equivalent_tiou(curve, 37.52) == pytest.approx(1.0)
    ci_low = [(0.5, 7.66), (1.0, 16.49), (0.3, 0.00), (0.9, 15.58), (0.7, 13.37)]     # vs uni_M_gt, by video
    assert pu.significant_down_to(ci_low) == 0.5
    assert pu.significant_down_to([(1.0, -0.1), (0.9, 2.0)]) is None
