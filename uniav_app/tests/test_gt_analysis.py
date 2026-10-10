"""GT manifests, gallery rule, error analysis flags, count statistics, theta choice."""

import numpy as np
import pytest

from uniav_app.analysis import analyze_video, count_stats, duration_bin
from uniav_app.config import Config
from uniav_app.evaluate import choose_theta, theta_sweep
from uniav_app.subset import manifest_event_id, parse_manifest, select_gallery


def test_manifest_parsing():
    recs = [{"id": "a_b_c_1", "video": "a_b_c.mp4", "timestamps": [20.0, 30.0], "text": "two"},
            {"id": "a_b_c_0", "video": "a_b_c.mp4", "timestamps": [0, 10], "text": "one"}]
    assert manifest_event_id(recs[0]) == 1
    gt = parse_manifest(recs, "val")
    assert list(gt) == ["a_b_c"]
    assert [(e["event_id"], e["ts"], e["te"], e["caption"]) for e in gt["a_b_c"]] == [(0, 0.0, 10.0, "one"),
                                                                                       (1, 20.0, 30.0, "two")]


def test_select_gallery_rule():
    rows = select_gallery({"v1", "v2", "v3"}, {"d1", "d2", "d3", "t1"}, {"v1", "v2", "d1", "d2", "d3", "t1"},
                          n_videos=4, seed=0, exclude={"t1"})
    vids = [r["video_id"] for r in rows]
    assert vids == sorted(vids)
    assert {r["video_id"] for r in rows if r["split"] == "val"} == {"v1", "v2"}   # v3 has no mp4
    assert sum(r["split"] == "dev" for r in rows) == 2 and "t1" not in vids


def test_analyze_video_flags():
    gts = [{"ts": 0, "te": 10, "caption": "a"}, {"ts": 10, "te": 20, "caption": "b"},
           {"ts": 40, "te": 50, "caption": "c"}, {"ts": 60, "te": 80, "caption": "d"}]
    segs = [{"t_s": 0, "t_e": 20, "conf": 0.9}, {"t_s": 60, "t_e": 70, "conf": 0.8},
            {"t_s": 70, "t_e": 80, "conf": 0.7}, {"t_s": 90, "t_e": 95, "conf": 0.6}]
    g, s = analyze_video(gts, segs)
    assert [r["missed"] for r in g] == [False, False, True, False]
    assert [r["fragmented"] for r in g] == [False, False, False, True]     # GT d cut into two segments
    assert g[0]["in_merged_seg"] and g[1]["in_merged_seg"]
    assert [r["merge"] for r in s] == [True, False, False, False]          # segment 0 covers a and b
    assert [r["fp_no_overlap"] for r in s] == [False, False, False, True]
    assert sum(r["match@0.5"] >= 0 for r in g) == 2                        # one of a/b, one of the d parts
    assert sum(r["left_offset"] is not None for r in g) == 2
    assert g[0]["best_tiou"] == pytest.approx(0.5) and g[2]["best_tiou"] == 0.0
    assert duration_bin(4.9) == "<5s" and duration_bin(80) == ">=80s"


def test_analyze_video_empty():
    g, s = analyze_video([{"ts": 0, "te": 5, "caption": "x"}], [])
    assert g[0]["missed"] and s == []


def test_count_stats():
    c = count_stats({"a": 2, "b": 6, "c": 4}, {"a": 4, "b": 3, "c": 4})
    assert c["mean_K_pred"] == 4 and c["mean_K_GT"] == pytest.approx(11 / 3)
    assert (round(c["pct_fewer"], 1), round(c["pct_equal"], 1), round(c["pct_more"], 1)) == (33.3, 33.3, 33.3)
    assert c["pct_ratio_gt_2"] == 0.0 and c["pct_ratio_lt_0.5"] == 0.0


def test_choose_theta_closest_count():
    rows = [{"split": "dev", "theta": 0.1, "mean_K_pred": 20, "mean_K_GT": 7.5},
            {"split": "dev", "theta": 0.3, "mean_K_pred": 8.0, "mean_K_GT": 7.5},
            {"split": "dev", "theta": 0.4, "mean_K_pred": 7.0, "mean_K_GT": 7.5},
            {"split": "val", "theta": 0.5, "mean_K_pred": 7.5, "mean_K_GT": 7.5}]
    assert choose_theta(rows, "dev") == (0.3, "dev")                       # tie 0.5 vs 0.5 -> smaller theta
    assert choose_theta([r for r in rows if r["split"] == "val"], "dev") == (0.5, "val")


def test_evaluate_split_end_to_end_without_judge():
    from uniav_app.evaluate import evaluate_split
    gt = {"v": [{"event_id": 0, "ts": 0, "te": 10, "caption": "a", "split": "val"},
                {"event_id": 1, "ts": 60, "te": 80, "caption": "b", "split": "val"}]}
    raw = {"v": {"duration": 100.0, "n": 100, "proposals": [{"ts": 0, "te": 10, "conf": 0.9},
                                                            {"ts": 60, "te": 70, "conf": 0.8}]}}
    ev = [{"t_s": 0, "t_e": 10, "conf": 0.9, "caption": "a"}, {"t_s": 60, "t_e": 70, "conf": 0.8, "caption": "x"},
          {"t_s": 70, "t_e": 80, "conf": 0.7, "caption": "y"}]
    pred = {"v": {"split": "val", "events": ev}}
    pred_api = {"v": {"split": "val", "events": ev[:2]}}
    out = evaluate_split(["v"], gt, raw, pred, pred_api, judge=None)
    vr = out["video_rows"][0]
    assert vr["split"] == "val" and vr["n_fragmented_gt"] == 1 and vr["K_pred"] == 3
    assert all(r["split"] == "val" for r in out["gt_rows"] + out["seg_rows"])
    assert out["metrics"]["errors"]["pct_gt_fragmented"] == 50.0
    assert out["metrics"]["comparison"]["uniav_uemr"]["R@0.5"] == 1.0


def test_theta_sweep_counts_fall_with_theta():
    raw = {"v": {"duration": 100.0, "proposals": [{"ts": 10 * i, "te": 10 * i + 8, "conf": 0.9 - 0.1 * i}
                                                  for i in range(8)]}}
    gt = {"v": [{"ts": 0, "te": 8}, {"ts": 10, "te": 18}, {"ts": 20, "te": 28}]}
    rows = theta_sweep(raw, gt, {"v": "dev"}, Config(theta_grid=(0.05, 0.45, 0.85)))
    assert [r["mean_K_pred"] for r in rows] == [8, 5, 1]
    assert rows[1]["R@0.5"] == pytest.approx(1.0) and rows[1]["P@0.5"] == pytest.approx(3 / 5)
    assert np.isclose(rows[2]["AR@K_GT@0.5"], 1 / 3)
