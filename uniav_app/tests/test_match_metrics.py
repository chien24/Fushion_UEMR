"""tIoU, one-to-one matching, Recall@tIoU / AR@N and mAP on hand-computed examples (CPU, no model)."""

import numpy as np
import pytest

from uniav_app.match import best_match, hungarian_match, tiou, tiou_matrix
from uniav_app.metrics import average_precision, interpolated_ap, mean_ap, mean_best_tiou, prf, recall_at_n


def test_tiou():
    assert tiou((0, 10), (0, 10)) == 1.0
    assert tiou((0, 10), (5, 15)) == pytest.approx(5 / 15)
    assert tiou((0, 10), (10, 20)) == 0.0
    assert tiou((0, 10), (2, 4)) == pytest.approx(0.2)
    assert tiou((3, 3), (3, 3)) == 0.0          # empty union


def test_tiou_matrix_and_dict_input():
    m = tiou_matrix([{"ts": 0, "te": 10}, {"t_s": 5, "t_e": 15}], [(0, 10), (10, 20)])
    assert m.shape == (2, 2)
    np.testing.assert_allclose(m, [[1.0, 0.0], [5 / 15, 5 / 15]])
    assert tiou_matrix([], [(0, 1)]).shape == (0, 1)


def test_hungarian_beats_greedy():
    # greedy by tIoU takes P0-G0 (0.818) and leaves P1 with G1 (0.143 < 0.5): 1 pair.
    # one-to-one optimum: P0-G1 (0.538) + P1-G0 (0.6): 2 pairs.
    preds, gts = [(1, 11), (0, 6)], [(0, 10), (4, 14)]
    pairs = hungarian_match(preds, gts, 0.5)
    assert {(p, g) for p, g, _ in pairs} == {(0, 1), (1, 0)}
    assert pairs[0][2] == pytest.approx(0.6) and pairs[1][2] == pytest.approx(7 / 13)
    assert hungarian_match(preds, gts, 0.9) == []
    assert hungarian_match([], gts, 0.5) == []


def test_best_match_not_one_to_one():
    assert best_match([(0, 20)], [(0, 10), (10, 20)]) == [(0, 0.5), (0, 0.5)]
    assert best_match([], [(0, 1)]) == [(-1, 0.0)]


def test_recall_at_n_by_hand():
    preds = {"A": [(50, 60, 0.7), (0, 10, 0.9), (21, 29, 0.8)]}     # unsorted on purpose
    gts = {"A": [(0, 10), (20, 30)]}                                # (21,29) vs (20,30): tIoU 0.8
    assert recall_at_n(preds, gts, 1, (0.5,)) == {0.5: 0.5}
    assert recall_at_n(preds, gts, 2, (0.5, 0.9)) == {0.5: 1.0, 0.9: 0.5}
    assert recall_at_n(preds, gts, "K_GT", (0.5,)) == {0.5: 1.0}   # k = min(2, 3)
    assert recall_at_n(preds, gts, {"A": 1}, (0.5,)) == {0.5: 0.5}
    assert mean_best_tiou(preds, gts, "K_GT") == pytest.approx((1.0 + 0.8) / 2)
    assert recall_at_n({}, gts, 5, (0.5,)) == {0.5: 0.0}          # a video without proposals


def test_interpolated_ap_perfect():
    assert interpolated_ap(np.array([1.0, 1.0]), np.array([0.5, 1.0])) == pytest.approx(1.0)


def test_average_precision_by_hand():
    # sorted by score: .9 TP, .8 FP (GT0 already taken), .7 TP, .6 FP, .5 TP ; npos = 3
    # prec 1, .5, .667, .5, .6 ; rec 1/3, 1/3, 2/3, 2/3, 1
    # interpolated: (1/3)*1 + (1/3)*(2/3) + (1/3)*0.6 = 0.755556
    preds = {"A": [(0, 10, 0.9), (1, 9, 0.8), (20, 30, 0.7)], "B": [(50, 60, 0.6), (0, 10, 0.5)]}
    gts = {"A": [(0, 10), (20, 30)], "B": [(0, 10)]}
    assert average_precision(preds, gts, 0.5) == pytest.approx((1 + 2 / 3 + 0.6) / 3)
    m = mean_ap(preds, gts, (0.5, 0.9))
    assert m["AP"][0.9] == pytest.approx((1 + 2 / 3 + 0.6) / 3)      # (1,9) has tIoU 0.8 < 0.9: still FP
    assert m["mAP"] == pytest.approx((1 + 2 / 3 + 0.6) / 3)
    assert average_precision({"A": [(0, 10, 1.0)]}, {"A": [(0, 10)]}, 0.5) == pytest.approx(1.0)
    assert average_precision({}, gts, 0.5) == 0.0


def test_prf_by_hand():
    preds = {"A": [(0, 10, 1), (20, 30, 1), (40, 50, 1)], "B": []}
    gts = {"A": [(0, 10), (22, 30)], "B": [(0, 5)]}
    r = prf(preds, gts, 0.5)
    assert (r["tp"], r["n_pred"], r["n_gt"]) == (2, 3, 3)
    assert r["precision"] == pytest.approx(2 / 3) and r["recall"] == pytest.approx(2 / 3)
    assert r["f1"] == pytest.approx(2 / 3)
    assert prf(preds, gts, 0.9)["tp"] == 1
