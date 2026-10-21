"""Partitions and dev rules of the YC2 supplementary run (omni_retrieval.bo_sung). No model, no GPU."""

import numpy as np
import pytest

from omni_retrieval.bo_sung import centred_windows, fill_gaps, frame_index, overlap, pick_fill, pick_level, uncovered


def test_uncovered_counts_start_and_end():
    assert uncovered([[10, 20], [15, 30]], 60) == [[0.0, 10.0], [30.0, 60.0]]
    assert uncovered([[0, 60]], 60) == []
    assert uncovered([], 60) == [[0.0, 60.0]]


def test_fill_gaps_keeps_r6_and_cuts_long_gaps():
    segs, n = fill_gaps([[10, 20], [15, 30]], 60, l_min=5, l_fill=10)
    assert n == 4
    assert segs == [[0, 10], [10, 20], [15, 30], [30, 40], [40, 50], [50, 60]]
    segs, n = fill_gaps([[10, 20], [15, 30]], 60, l_min=11, l_fill=10)   # the 10 s gap at the start is too short
    assert n == 3 and segs[0] == [10, 20]
    _, n = fill_gaps([[0, 10]], 35, l_min=3, l_fill=10)                   # 25 s / 10 = 2.5 -> 3 (half up)
    assert n == 3
    _, n = fill_gaps([[0, 10]], 13, l_min=3, l_fill=10)                   # short gap -> still one window
    assert n == 1
    segs, n = fill_gaps([], 20, l_min=3, l_fill=7)
    assert n == 3 and segs[0][0] == 0 and segs[-1][1] == pytest.approx(20)


def test_centred_windows_clip_to_video():
    w = centred_windows([50, 1, 99, 120], 100, 4)                          # half = 12.5; 120 clamps to 100
    assert w == [[0.0, 13.5], [37.5, 62.5], [86.5, 100.0], [87.5, 100.0]]


def test_overlap():
    assert overlap([[0, 10], [5, 15]]) == pytest.approx(0.25)
    assert overlap([[0, 10], [10, 20]]) == pytest.approx(0.0)
    assert overlap([]) == 0.0


def test_pick_level_prefers_theta_then_topk():
    rows = [{"method": "theta", "param": 0.30, "K dev": 12.8}, {"method": "theta", "param": 0.25, "K dev": 18.25},
            {"method": "theta", "param": 0.26, "K dev": 17.75}, {"method": "topk", "param": 14, "K dev": 13.9}]
    assert pick_level(rows, 18.0)["param"] == 0.26                        # 17.75 and 18.25 tie -> larger theta
    assert pick_level(rows, 14.0) == {"method": "topk", "param": 14, "K dev": 13.9}   # theta misses by 1.2 > 0.5
    assert pick_level(rows[:3], 14.0)["param"] == 0.30                    # no top-k row: closest theta


def test_pick_fill_tie_goes_to_fewer_vectors():
    rows = [{"cfg": "a", "R@1": 30.0, "vec/video": 12.0}, {"cfg": "b", "R@1": 29.8, "vec/video": 10.0},
            {"cfg": "c", "R@1": 29.6, "vec/video": 8.0}]
    assert pick_fill(rows)["cfg"] == "b"                                  # c is 0.4 below the best: out
    assert pick_fill(rows, tie=0.5)["cfg"] == "c"


def test_frame_index_matches_video_decord():
    assert frame_index(1.0, 2.0, 30, 100).tolist() == np.linspace(30, 60, 8, dtype=int).tolist()
    assert frame_index(3.0, 10.0, 30, 100)[-1] == 99                      # end clipped to the last frame
    with pytest.raises(ValueError):
        frame_index(5.0, 6.0, 30, 100)
