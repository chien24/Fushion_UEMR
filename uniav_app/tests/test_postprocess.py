"""UEMR post-processing (section A): theta, K_max, top-1 fallback, merge tIoU > 0.7, d_min, time order."""

import pytest

from uniav_app.postprocess import (extend_short, filter_theta, merge_overlaps, postprocess_uemr, sort_by_time,
                                   top_k)


def P(ts, te, conf):
    return {"ts": ts, "te": te, "conf": conf}


PROPS = [P(50, 60, 0.6), P(0, 10, 0.9), P(20, 30, 0.8), P(70, 80, 0.3)]


def test_filter_and_top_k():
    assert [p["conf"] for p in filter_theta(PROPS, 0.5)] == [0.6, 0.9, 0.8]
    assert [p["conf"] for p in top_k(PROPS, 2)] == [0.9, 0.8]


def test_theta_kmax_sorted():
    out = postprocess_uemr(PROPS, 100, theta=0.5, k_max=2, d_min=2, merge_tiou=0.7)
    assert [(s["ts"], s["te"], s["conf"]) for s in out] == [(0, 10, 0.9), (20, 30, 0.8)]


def test_keep_top1_when_empty():
    out = postprocess_uemr(PROPS, 100, theta=0.95, k_max=30, d_min=2, merge_tiou=0.7)
    assert len(out) == 1 and out[0]["conf"] == 0.9
    assert postprocess_uemr([], 100, theta=0.5) == []


def test_merge_union_conf_max():
    out = merge_overlaps([P(0, 10, 0.5), P(0.5, 10, 0.9)], 0.7)       # tIoU 0.95
    assert len(out) == 1
    assert (out[0]["ts"], out[0]["te"], out[0]["conf"], out[0]["merged"]) == (0, 10, 0.9, 2)


def test_merge_is_strict_and_iterated():
    assert len(merge_overlaps([P(0, 10, 0.9), P(0, 7, 0.8)], 0.7)) == 2   # tIoU exactly 0.7: not merged
    # A-C (0.833) first, then [0,12] with B (0.833): one segment
    out = merge_overlaps([P(0, 10, 0.9), P(1, 11, 0.8), P(0, 12, 0.3)], 0.7)
    assert len(out) == 1
    assert (out[0]["ts"], out[0]["te"], out[0]["conf"], out[0]["merged"]) == (0, 12, 0.9, 3)
    # non-overlapping segments untouched
    assert len(merge_overlaps([P(0, 10, 0.9), P(20, 30, 0.8)], 0.7)) == 2


def test_extend_short():
    out = extend_short([P(10, 11, 1), P(0, 0.5, 1), P(99.5, 100, 1), P(30, 40, 1)], 2.0, 100)
    assert [(s["ts"], s["te"]) for s in out] == [(9.5, 11.5), (0.0, 2.0), (98.0, 100.0), (30, 40)]
    assert out[0]["extended"] and "extended" not in out[3]
    assert [(s["ts"], s["te"]) for s in extend_short([P(0.2, 0.8, 1)], 2.0, 1.5)] == [(0.0, 1.5)]
    assert extend_short([P(10, 11, 1)], 2.0, None)[0]["ts"] == pytest.approx(9.5)


def test_sort_by_time():
    assert [s["ts"] for s in sort_by_time(PROPS)] == [0, 20, 50, 70]


def test_full_order_merge_then_extend():
    props = [P(10, 11, 0.9), P(10.05, 11, 0.8), P(40, 41, 0.7)]
    out = postprocess_uemr(props, 41.5, theta=0.5, k_max=30, d_min=2.0, merge_tiou=0.7)
    assert [(round(s["ts"], 3), round(s["te"], 3), s["conf"]) for s in out] == [(9.5, 11.5, 0.9), (39.5, 41.5, 0.7)]


def test_api_config_uses_athena_select_events():
    pytest.importorskip("torch")
    pytest.importorskip("yaml")
    from uniav_app.postprocess import postprocess_api
    props = [P(0, 10, 0.9), P(1, 10, 0.8), P(20, 30, 0.5), P(40, 50, 0.2)]
    out = postprocess_api(props, min_score=0.4, max_overlap=0.3, max_events=30)
    assert [(s["ts"], s["te"]) for s in out] == [(0, 10), (20, 30)]   # (1,10) overlaps 0.9 > 0.3; 0.2 < 0.4
