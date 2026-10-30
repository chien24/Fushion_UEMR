"""Unit tests of uemr_htstep (CPU, no data files)."""

import tarfile

import numpy as np
import pytest

from omni_retrieval.evaluate import rank_metrics, text_to_clip as yc2_text_to_clip, video_ranks
from omni_retrieval.search import SegmentIndex
from uemr_htstep import data, metrics, partitions


def htstep_json():
    return {
        "a": {"role": "val_seen", "duration": 100.0, "segments": [[10, 20], [30, 40], [50, 60]],
              "captions": ["Add salt.", "Stir well!", "add  SALT"]},          # two steps, one headline
        "b": {"role": "val_seen", "duration": 80.0, "segments": [[5, 15]], "captions": ["add salt"]},
        "c": {"role": "val_seen", "duration": 50.0, "segments": [[0, 10]], "captions": ["Serve."]},
        "nomp4": {"role": "val_seen", "duration": 50.0, "segments": [[0, 10]], "captions": ["x"]},
        "nostep": {"role": "val_seen", "duration": 50.0, "segments": [], "captions": []},
        "yc2vid": {"role": "val_seen", "duration": 50.0, "segments": [[0, 10]], "captions": ["y"]},
        "tr": {"role": "train", "duration": 50.0, "segments": [[0, 10]], "captions": ["z"]},
    }


MANIFEST = {"shards": {"videos/t_1000.tar": {"size": 1, "videos": ["a", "b", "nostep"]},
                       "videos/t_1001.tar": {"size": 1, "videos": ["c", "yc2vid", "tr"]}}}


def test_normalize():
    assert data.normalize("  Add  SALT, then stir!\n") == "add salt then stir"
    assert data.normalize("Add salt.") == data.normalize("add  SALT") == "add salt"


def test_select_videos_rule_and_yc2_filter():
    keep, counts = data.select_videos(htstep_json(), data.video_tars(MANIFEST), {"yc2vid", "other"})
    assert keep == ["a", "b", "c"]
    assert counts == {"val_seen": 6, "with mp4": 5, "with >= 1 Full Match step": 5, "YouCook2 id": 1,
                      "YouCook2 id among mp4 + step": 1, "kept": 3}


def test_video_in_two_tars_is_an_error():
    with pytest.raises(ValueError):
        data.video_tars({"shards": {"x.tar": {"videos": ["a"]}, "y.tar": {"videos": ["a"]}}})


def gallery():
    h = htstep_json()
    caps, dropped = data.queries(data.steps(h, ["a", "b", "c"]), {"a": 100.0, "b": 80.0, "c": 50.0})
    assert not dropped
    return caps


def test_queries_clamp_and_drop():
    h = {"v": {"role": "val_seen", "segments": [[90, 130], [99.5, 120]], "captions": ["p", "q"]}}
    caps, dropped = data.queries(data.steps(h, ["v"]), {"v": 100.0})
    assert [(c["gt_ts"], c["gt_te"], c["gt_seg_key"]) for c in caps] == [(90.0, 100.0, "v__90.00_100.00")]
    assert [d["caption_id"] for d in dropped] == ["v#1"]


def test_headline_stats():
    s = data.headline_stats(gallery())
    assert s["distinct headlines"] == 3 and s["queries"] == 5
    assert s["% queries whose headline is in >= 2 videos"] == pytest.approx(60.0)   # the three "add salt"
    assert s["top"][0] == {"headline": "add salt", "videos": 2}


def test_cross_check():
    rows = data.steps(htstep_json(), ["a"])
    official = {"a": {"annotations": [{"segment": [10, 20], "partial": "Full Match", "id": 1},
                                      {"segment": [30, 40], "partial": "Partial Match", "id": 2},
                                      {"segment": [50, 60], "partial": "Full Match", "id": 3}]}}
    _, bad = data.cross_check(rows, official, {1: "Add salt.", 2: "Stir well!", 3: "add  SALT"}, n=10)
    assert [b["caption_id"] for b in bad] == ["a#1"]                     # Partial Match in the official file


def test_extract_members(tmp_path):
    for n in ("x.mp4", "y.mp4"):
        (tmp_path / n).write_bytes(n.encode())
    with tarfile.open(tmp_path / "t.tar", "w") as t:
        t.add(tmp_path / "x.mp4", arcname="videos/x.mp4")
        t.add(tmp_path / "y.mp4", arcname="videos/y.mp4")
    found = data.extract_members(tmp_path / "t.tar", {"y.mp4": str(tmp_path / "out" / "y.mp4")})
    assert found == ["y.mp4"] and (tmp_path / "out" / "y.mp4").read_bytes() == b"y.mp4"
    assert not (tmp_path / "out" / "x.mp4").exists() and not list((tmp_path / "out").glob("*.part"))


def test_partitions_r3_has_k_pred_segments():
    dur = {"a": 100.0, "b": 33.3}
    r6 = {"a": [[10.0, 20.0], [50.0, 70.0], [80.0, 85.0]], "b": [[1.0, 30.0]]}
    gt = {"a": [[10.0, 20.0]], "b": [[5.0, 15.0]]}
    p = partitions.build(r6, gt, dur, chunk_sec=14.0, d_min=2.0, l_min=5.0, l_fill=10.0)
    for v in dur:
        segs = p["uni_M"][v]
        assert len(segs) == len(r6[v])
        assert segs[0][0] == 0.0 and abs(segs[-1][1] - dur[v]) < 1e-6
        assert all(abs(x[1] - y[0]) < 0.011 for x, y in zip(segs, segs[1:]))
    assert p["global"]["a"] == [[0.0, 100.0]]
    assert p["uni_chunk"]["a"][:2] == [[0.0, 14.0], [14.0, 28.0]] and p["uni_chunk"]["a"][-1][1] == 100.0
    # R8 keeps every R6 segment and fills the gaps 0-10 (1 window), 20-50 (3), 70-80 (1), 85-100 (2)
    assert [s for s in p["r8"]["a"] if s in r6["a"]] == r6["a"] and len(p["r8"]["a"]) == 3 + 1 + 3 + 1 + 2
    assert set(p) == set(partitions.ROWS)
    with pytest.raises(ValueError):
        partitions.build({"a": r6["a"]}, gt, dur, 14.0, 2.0, 5.0, 10.0)


def test_r6_chain():
    props = [{"ts": 10.0, "te": 20.0004, "conf": 0.9}, {"ts": 10.5, "te": 20.0, "conf": 0.5},
             {"ts": 98.9, "te": 99.6, "conf": 0.6}, {"ts": 40.0, "te": 50.0, "conf": 0.1}]
    # theta 0.4 keeps three, the two overlapping merge (tIoU > 0.7), the short one is widened to d_min inside [0, 99]
    assert partitions.r6(props, 99.0, 0.4, 30, 2.0, 0.7) == [[10.0, 20.0], [97.0, 99.0]]


# ---------------------------------------------------------------- scoring

def toy():
    """3 videos; queries = the 5 steps of gallery(); one 2-d direction per headline."""
    caps = gallery()
    d = {"add salt": [1, 0], "stir well": [0, 1], "serve": [-1, 0]}
    emb = np.array([d[c["headline"]] for c in caps], np.float32)
    text = {"caption_id": np.array([c["caption_id"] for c in caps]), "video_id": np.array([c["video_id"] for c in caps]),
            "gt_ts": np.array([c["gt_ts"] for c in caps], np.float32), "gt_te": np.array([c["gt_te"] for c in caps], np.float32),
            "gt_seg_key": np.array([c["gt_seg_key"] for c in caps]), "emb": emb}
    # segments: video b has the best "add salt" segment (score 1.0) but at the wrong time for a's queries
    seg = [("a", 10, 20, [0.8, 0.6]), ("a", 30, 40, [0, 1]), ("a", 52, 62, [0.9, 0.1]),
           ("b", 5, 15, [1, 0]), ("c", 0, 10, [-1, 0])]
    e = np.array([s[3] for s in seg], np.float32)
    e /= np.linalg.norm(e, axis=1, keepdims=True)
    index = SegmentIndex(e, [s[0] for s in seg], [s[1] for s in seg], [s[2] for s in seg], "toy")
    return caps, text, index


def test_strict_equals_yc2_video_ranks():
    caps, text, index = toy()
    got = metrics.text_to_video(index, text, caps, "strict")
    ref = video_ranks(index, text)
    assert np.array_equal(got["ranks"], ref["ranks"])
    top1 = (ref["ranks"] == 1) & ref["covered"]
    from omni_retrieval.search import tiou_vec
    hit = tiou_vec(ref["best_ts"], ref["best_te"], text["gt_ts"], text["gt_te"]) >= 0.5
    assert np.array_equal(got["vcmr"], top1 & hit)


def test_multi_positive_video_and_vcmr():
    caps, text, index = toy()
    s, m = (metrics.text_to_video(index, text, caps, mode) for mode in ("strict", "multi"))
    ids = list(text["caption_id"])
    a0, a2, b0 = ids.index("a#0"), ids.index("a#2"), ids.index("b#0")
    # a's "add salt" queries: video b scores 1.0 > a's 0.9 -> strict rank 2, multi rank 1 (b also has "add salt")
    assert s["ranks"][a0] == 2 and m["ranks"][a0] == 1 and m["ranks"][a2] == 1
    # multi VCMR: best right video is b, its segment [5, 15] is b's own "add salt" step -> hit
    assert m["vcmr"][a0] and not s["vcmr"][a0]
    # b's query: strict and multi agree (b is top), VCMR hit in both
    assert s["ranks"][b0] == m["ranks"][b0] == 1 and s["vcmr"][b0] and m["vcmr"][b0]
    assert (m["ranks"] <= s["ranks"]).all()


def test_multi_vcmr_needs_same_headline_step_in_that_video():
    caps, text, index = toy()
    # a#1 "stir well": only a has it; a's best segment for it is [30, 40] = the step -> hit in both modes
    i = list(text["caption_id"]).index("a#1")
    for mode in ("strict", "multi"):
        r = metrics.text_to_video(index, text, caps, mode)
        assert r["ranks"][i] == 1 and r["vcmr"][i]
    # a#0's best segment sits on a's *other* "add salt" step (a#2, 50-60): a hit for multi only
    j = list(text["caption_id"]).index("a#0")
    index2 = SegmentIndex(np.array([[1, 0], [0, 1]], np.float32), ["a", "a"], [50.0, 30.0], [60.0, 40.0])
    s, m = (metrics.text_to_video(index2, text, caps, mode) for mode in ("strict", "multi"))
    assert s["ranks"][j] == 1 and not s["vcmr"][j] and m["vcmr"][j]


def test_query_without_video_gets_n_plus_1():
    caps, text, index = toy()
    small = index.subset(["a", "b"])
    for mode in ("strict", "multi"):
        r = metrics.text_to_video(small, text, caps, mode)
        assert r["ranks"][list(text["caption_id"]).index("c#0")] == small.n_videos + 1


def test_text_to_clip_strict_equals_yc2_and_multi(tmp_path):
    caps, text, _ = toy()
    keys = sorted({c["gt_seg_key"] for c in caps})
    clip_emb = np.array([text["emb"][list(text["gt_seg_key"]).index(k)] for k in keys], np.float32)
    own_a2 = next(c["gt_seg_key"] for c in caps if c["caption_id"] == "a#2")
    clip_emb[keys.index(own_a2)] = [0.6, 0.8]          # a#2's own clip is off; a#0 and b#0's clips are "add salt"
    clip_emb /= np.linalg.norm(clip_emb, axis=1, keepdims=True)
    np.savez(tmp_path / "segments.npz", partition=np.array(["gt"] * len(keys)), seg_key=np.array(keys), emb=clip_emb)
    np.savez(tmp_path / "text.npz", **{k: v for k, v in text.items()})
    k, e = metrics.gt_clips(tmp_path / "segments.npz")
    strict = metrics.text_to_clip(text, caps, k, e, "strict")
    ref = yc2_text_to_clip(tmp_path)["subset"]
    assert {m: v for m, v in rank_metrics(strict).items()} == {m: ref[m] for m in rank_metrics(strict)}
    multi = metrics.text_to_clip(text, caps, k, e, "multi")
    i = list(text["caption_id"]).index("a#2")
    assert strict[i] > 1 and multi[i] == 1 and (multi <= strict).all()


def test_right_answers_two_steps_same_headline_one_video():
    r = metrics.right_answers(["a", "a"], ["add salt", "stir"], ["a", "b"], [{"add salt", "stir"}, {"add salt"}], "multi")
    assert r.tolist() == [[True, True], [True, False]]
    with pytest.raises(ValueError):
        metrics.right_answers(["a"], ["x"], ["a"], [set()], "loose")
