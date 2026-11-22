"""Post-lock report analyses on YouCook2 (uemr_htstep.bao_cao_yc2). CPU, synthetic data."""

import numpy as np

from uemr_htstep import bao_cao_yc2 as bc


def test_miss_r6():
    caps = [{"video_id": "a", "gt_ts": 0.0, "gt_te": 10.0}, {"video_id": "a", "gt_ts": 50.0, "gt_te": 60.0},
            {"video_id": "z", "gt_ts": 0.0, "gt_te": 5.0}]
    best = bc.miss_r6(caps, {"a": [[0.0, 10.0], [20.0, 30.0]]})
    assert np.allclose(best, [1.0, 0.0, 0.0])


def test_pick_kinds():
    rk = {("U", "R8"): np.array([1, 1, 200, 1]), ("U", "R6"): np.array([1, 3, 300, 1]), ("b", "R0"): np.array([50, 2, 400, 1])}
    got = bc.pick(rk, np.array([0, 0, 0, 1]), np.array([0.9, 0.1, 0.0, 0.8]), np.random.default_rng(0))
    assert got["thắng Cả video"] == (0, 1)
    assert got["bỏ sót nhưng đoạn lấp cứu"] == (1, 1)
    assert got["đúng khoảnh khắc"] == (3, 1)
    assert got["thất bại"] == (2, 1)
