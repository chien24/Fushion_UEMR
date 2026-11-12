"""Unit tests of uemr_htstep (CPU, no data files). The UniAV tests need UNIAV_REPO = a checkout of HanLi-Retriever 77fc4fa."""

import csv
import json
import os

import numpy as np
import pytest

from omni_retrieval.evaluate import paired_bootstrap, rank_metrics, text_to_clip as yc2_text_to_clip, video_ranks
from omni_retrieval.search import SegmentIndex
from uemr_htstep import data, metrics, partitions

# ---------------------------------------------------------------- gallery, queries


def rows_of(steps):
    """Records of ``val_omni_htstep.jsonl``: (video, step, text, window, partial)."""
    return [{"id": f"{v}#{i}", "video_id": v, "text": t, "window": list(w), "partial": p}
            for v, i, t, w, p in steps]


def manifest():
    return rows_of([("a", 0, "Cook rice: Add salt.", (10, 20), "Full Match"), ("a", 1, "Cook rice: Stir well!", (30, 40), "Partial Match"),
                    ("a", 2, "Cook rice: add  SALT", (50, 60), "Full Match"), ("b", 0, "Cook rice: add salt", (5, 15), "Full Match"),
                    ("c", 0, "Serve: Serve.", (0, 10), "Partial Match")])


def annotations():
    ann = {v: {"subset": "val_seen", "annotations": [{"segment": [1, 2]}]} for v in ("a", "b", "c", "nomp4", "nofeat", "both")}
    ann["t1"] = {"subset": "train", "annotations": [{"segment": [0, 1]}] * 60 + [{"segment": [2, 3]}]}   # capped at 48
    ann["t2"] = {"subset": "train", "annotations": [{"segment": [3, 3]}, {"segment": [0, 2]}, {"segment": [2, 4]}]}  # end > start
    ann["t3"] = {"subset": "train", "annotations": []}                                                               # no step
    return ann


def test_normalize():
    assert data.normalize("  Add  SALT, then stir!\n") == "add salt then stir"
    assert data.normalize("Cook rice: Add salt.") == data.normalize("cook RICE add  salt")


def test_select_gallery_rule_and_train_counts():
    rows = manifest() + rows_of([("nofeat", 0, "x", (0, 1), "Full Match"), ("both", 0, "x", (0, 1), "Full Match"),
                                 ("t1", 0, "x", (0, 1), "Full Match")])
    keep, counts = data.select_gallery(annotations(), {"a", "b", "c", "both", "nomp4"}, rows, {"both"}, {"t1"})
    assert keep == ["a", "b", "c"]
    assert counts == {"val_seen": 6, "with UniAV features": 5, "in val_omni_htstep.jsonl": 4, "in train of a model": 1, "kept": 3}
    assert data.select_gallery(annotations(), {"a"}, manifest(), {"t1"}, {"t2"})[1]["in train of a model"] == 0


def test_train_steps_per_video():
    assert data.train_steps_per_video(annotations()) == {"train_videos": 2, "train_steps": 50, "mean": 25.0}


def test_queries_and_gt():
    caps = data.queries(rows_of([("a", 0, "T: X", (9.994, 20.004), "Partial Match"), ("z", 0, "T: Y", (1, 2), "Full Match")]), ["a"])
    assert caps == [{"caption_id": "a#0", "video_id": "a", "sentence": "T: X", "headline": "t x", "partial": "Partial Match",
                     "gt_ts": 9.99, "gt_te": 20.0, "gt_seg_key": "a__9.99_20.00"}]
    assert data.gt_partition(data.queries(manifest(), ["a", "b", "c"]))["a"] == [[10, 20], [30, 40], [50, 60]]


def test_headline_stats():
    caps = data.queries(manifest(), ["a", "b", "c"])
    s = data.headline_stats(caps)
    assert s["distinct texts"] == 3 and s["queries"] == 5
    assert s["% queries whose text is in >= 2 videos"] == pytest.approx(60.0)   # the three "cook rice add salt"
    assert s["top"][0] == {"text": "cook rice add salt", "videos": 2}


def teacher_dir(tmp_path):
    np.savez(tmp_path / "a.npz", segment=np.array([[10.0, 20.0], [30.0, 40.0]]), emb=np.array([[1, 0], [0, 1]], np.float16))
    np.savez(tmp_path / "b.npz", segment=np.array([[5.0, 15.0]]), emb=np.array([[3, 4]], np.float16))
    return tmp_path


def test_teacher_vectors_and_queries(tmp_path):
    teacher = data.teacher_vectors(teacher_dir(tmp_path))
    assert set(teacher) == {("a", 10.0, 20.0), ("a", 30.0, 40.0), ("b", 5.0, 15.0)}
    assert teacher[("a", 30.0, 40.0)]["row"] == 1 and teacher[("a", 30.0, 40.0)]["emb"].tolist() == [0.0, 1.0]
    rows = [dict(r, segment=r["window"]) for r in manifest()]
    caps = data.queries(rows, ["a", "b", "c"])
    keep, by_clip = data.teacher_queries(caps, rows, teacher)
    assert [caps[i]["caption_id"] for i in keep] == ["a#0", "a#1", "b#0"]       # a#2 and c#0 have no teacher vector
    assert set(by_clip) == {"a__10.00_20.00", "a__30.00_40.00", "b__5.00_15.00"}


def test_smoke_videos_cover_teacher_clips_and_av1():
    teacher = {(v, float(i), float(i + 1)): {} for v, n in (("long", 9), ("s1", 5), ("s2", 6), ("few", 2), ("s3", 5), ("s4", 7)) for i in range(n)}
    dur = {"long": 900.0, "s1": 10.0, "s2": 20.0, "few": 1.0, "s3": 30.0, "s4": 40.0, "av1": 50.0, "av1b": 60.0}
    assert data.smoke_videos(teacher, dur, {"av1b", "av1"}) == ["av1", "s1", "s2", "s3", "s4"]


def test_stored_cos_reads_the_right_row(tmp_path):
    from uemr_htstep.teacher_encode import stored_cos

    np.savez(tmp_path / "a.npz", emb=np.array([[1, 0], [0, 1]], np.float16))
    clips = [{"id": "x", "npz": str(tmp_path / "a.npz"), "row": 0}, {"id": "y", "npz": str(tmp_path / "a.npz"), "row": 1}]
    cos, vecs = stored_cos(lambda c: {"x": [2, 0], "y": [1, 1]}[c["id"]], clips)
    assert cos["x"] == pytest.approx(1.0) and cos["y"] == pytest.approx(2 ** -0.5) and set(vecs) == {"x", "y"}


def test_teacher_check_conditions():
    from uemr_htstep.teacher_encode import judge

    run = lambda t, b: judge(dict(enumerate(t)), dict(enumerate(b)), min_mean=0.999, min_clip=0.995, ratio=10, list_below=0.999)
    one_off = run([0.9999, 0.9999, 0.9985, 0.9999], [0.98, 0.98, 0.97, 0.98])     # one clip off at decoding level: listed, not fatal
    assert one_off["ok"] and set(one_off["below"]) == {2}
    assert not run([0.9999, 0.9999], [0.98, 0.9999])["discriminates"]             # tie on one clip: not closer
    close = run([0.9995, 0.9995], [0.9990, 0.9990])                               # best is closer than 10x of the teacher's gap
    assert close["gap_ratio"] == pytest.approx(2.0) and not close["discriminates"] and not close["ok"] and close["absolute"]
    assert not run([0.9999, 0.9949], [0.90, 0.90])["absolute"]                    # one clip under the 0.995 bar
    assert not run([0.9985, 0.9985], [0.90, 0.90])["absolute"]                    # mean under 0.999


def test_pairwise_cos_of_repeats():
    from uemr_htstep.teacher_encode import pairwise_cos

    assert pairwise_cos([[1, 0], [2, 0], [1, 1]]) == pytest.approx([1.0, 2 ** -0.5, 2 ** -0.5])


# ---------------------------------------------------------------- partitions and the rules of §4.2


def test_build_rows():
    dur = {"a": 100.0, "b": 33.3}
    r6 = {"a": [[10.0, 20.0], [50.0, 70.0], [80.0, 85.0]], "b": [[1.0, 30.0]]}
    r6gt = {"a": [[11.0, 21.0]], "b": [[2.0, 3.0]]}
    gt = {"a": [[10.0, 20.0]], "b": [[5.0, 15.0]]}
    p = partitions.build(r6, r6gt, gt, dur, chunk_sec=14.0, d_min=2.0, l_min=5.0, l_fill=10.0)
    for v in dur:
        segs = p["uni_M"][v]                                                  # R3: K_pred equal segments
        assert len(segs) == len(r6[v])
        assert segs[0][0] == 0.0 and abs(segs[-1][1] - dur[v]) < 1e-6
    assert p["global"]["a"] == [[0.0, 100.0]] and p["uniav_gtart"] == r6gt and p["gt"] == gt
    assert p["uni_chunk"]["a"][:2] == [[0.0, 14.0], [14.0, 28.0]] and p["uni_chunk"]["a"][-1][1] == 100.0
    # R8 keeps every R6 segment and fills the gaps 0-10 (1 window), 20-50 (3), 70-80 (1), 85-100 (2)
    assert [s for s in p["r8"]["a"] if s in r6["a"]] == r6["a"] and len(p["r8"]["a"]) == 3 + 1 + 3 + 1 + 2
    assert set(p) == set(partitions.ROWS)
    with pytest.raises(ValueError):
        partitions.build({"a": r6["a"]}, r6gt, gt, dur, 14.0, 2.0, 5.0, 10.0)


def test_r6_chain_and_past_end():
    props = [{"ts": 10.0, "te": 20.0004, "conf": 0.9}, {"ts": 10.5, "te": 20.0, "conf": 0.5},
             {"ts": 98.9, "te": 99.6, "conf": 0.6}, {"ts": 40.0, "te": 50.0, "conf": 0.1}]
    # theta 0.4 keeps three, the two overlapping merge (tIoU > 0.7), the short one is widened to d_min inside [0, 99]
    assert partitions.r6(props, 99.0, 0.4, 30, 2.0, 0.7) == [[10.0, 20.0], [97.0, 99.0]]
    # a proposal past the end of the mp4 is dropped, one crossing it is cut
    assert partitions.r6([{"ts": 120.0, "te": 130.0, "conf": 0.9}, {"ts": 95.0, "te": 130.0, "conf": 0.9}], 99.0, 0.4, 30, 2.0, 0.7) \
        == [[95.0, 99.0]]


def test_pick_theta_is_the_closest_count_and_uses_no_label():
    props = {"a": [{"ts": 0, "te": 10, "conf": 0.9}, {"ts": 20, "te": 30, "conf": 0.5}, {"ts": 40, "te": 50, "conf": 0.2}],
             "b": [{"ts": 0, "te": 10, "conf": 0.8}, {"ts": 20, "te": 30, "conf": 0.3}]}
    dur = {"a": 60.0, "b": 60.0}
    theta, k, _ = partitions.pick_theta(props, dur, target=2.0, k_max=30, d_min=2.0, merge_tiou=0.7,
                                        grid=[0.1, 0.2, 0.25, 0.3, 0.4])
    assert k == 2.0 and theta == 0.3                    # 0.25 and 0.3 tie at K = 2 -> larger theta
    assert partitions.mean_k(props, dur, 0.85, 30, 2.0, 0.7) == 1.0         # one proposal is always kept


def test_pick_chunk_matches_the_clip_budget():
    dur = {"a": 100.0, "b": 40.0}
    length, total = partitions.pick_chunk(dur, target=14, d_min=2.0)
    assert total == 14
    assert sum(len(partitions.uni_chunk(d, length, 2.0)) for d in dur.values()) == 14


# ---------------------------------------------------------------- scoring


def toy():
    """3 videos; queries = the 5 steps of the manifest; one 2-d direction per text."""
    caps = data.queries(manifest(), ["a", "b", "c"])
    d = {"cook rice add salt": [1, 0], "cook rice stir well": [0, 1], "serve serve": [-1, 0]}
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
    from omni_retrieval.search import tiou_vec
    hit = tiou_vec(ref["best_ts"], ref["best_te"], text["gt_ts"], text["gt_te"]) >= 0.5
    assert np.array_equal(got["vcmr"], (ref["ranks"] == 1) & ref["covered"] & hit)


def test_multi_positive_video_and_vcmr():
    caps, text, index = toy()
    s, m = (metrics.text_to_video(index, text, caps, mode) for mode in ("strict", "multi"))
    ids = list(text["caption_id"])
    a0, a2, b0 = ids.index("a#0"), ids.index("a#2"), ids.index("b#0")
    # a's "add salt" queries: video b scores 1.0 > a's 0.9 -> strict rank 2, multi rank 1 (b also has "add salt")
    assert s["ranks"][a0] == 2 and m["ranks"][a0] == 1 and m["ranks"][a2] == 1
    # multi VCMR: best right video is b, its segment [5, 15] is b's own "add salt" step -> hit
    assert m["vcmr"][a0] and not s["vcmr"][a0]
    assert s["ranks"][b0] == m["ranks"][b0] == 1 and s["vcmr"][b0] and m["vcmr"][b0]
    assert (m["ranks"] <= s["ranks"]).all()


def test_multi_vcmr_two_steps_with_the_same_text_in_one_video():
    caps, text, _ = toy()
    # a#0's best segment sits on a's *other* "add salt" step (a#2, 50-60): a hit for multi only
    j = list(text["caption_id"]).index("a#0")
    index = SegmentIndex(np.array([[1, 0], [0, 1]], np.float32), ["a", "a"], [50.0, 30.0], [60.0, 40.0])
    s, m = (metrics.text_to_video(index, text, caps, mode) for mode in ("strict", "multi"))
    assert s["ranks"][j] == 1 and not s["vcmr"][j] and m["vcmr"][j]


def test_query_without_video_gets_n_plus_1():
    caps, text, index = toy()
    small = index.subset(["a", "b"])
    for mode in ("strict", "multi"):
        assert metrics.text_to_video(small, text, caps, mode)["ranks"][list(text["caption_id"]).index("c#0")] == small.n_videos + 1


def test_sim_override_equals_the_plain_product_and_fuses():
    caps, text, index = toy()
    sim = text["emb"] @ index.emb.T
    for mode in ("strict", "multi"):
        a, b = metrics.text_to_video(index, text, caps, mode), metrics.text_to_video(index, text, caps, mode, sim=sim)
        assert np.array_equal(a["ranks"], b["ranks"]) and np.array_equal(a["vcmr"], b["vcmr"])
    # the score of a video is the max over its segments of the fused matrix: raising video c's only segment wins every query
    u = sim.copy()
    u[:, np.where(index.seg_video == "c")[0]] = 10.0
    assert (metrics.text_to_video(index, text, caps, "multi", sim=u)["ranks"][-1] == 1)


def test_columns_of_section_6():
    q_o, q_s = np.array([[1.0, 0.0]], np.float32), np.array([[0.0, 1.0]], np.float32)
    e_o = np.array([[1.0, 0.0], [0.5, 0.5]], np.float32)
    e_s = np.array([[0.0, 1.0], [0.0, 0.2]], np.float32)
    fill = np.array([False, True])
    assert metrics.column_sim("b", q_o, q_s, e_o, e_s).tolist() == [[1.0, 0.5]]
    assert metrics.column_sim("S", q_o, q_s, e_o, e_s)[0].tolist() == pytest.approx([1.0, 0.2])
    assert metrics.column_sim("U", q_o, q_s, e_o, e_s, beta=0.1)[0].tolist() == pytest.approx([1.1, 0.52])
    assert metrics.column_sim("C1", q_o, q_s, e_o, e_s, fill=fill)[0].tolist() == pytest.approx([1.0, 0.2])
    with pytest.raises(ValueError):
        metrics.column_sim("x", q_o, q_s, e_o, e_s)


def test_student_rows_follow_the_index_order():
    _, _, index = toy()
    vid = ["c", "a", "b", "a", "a"]
    ts, te = [0.0, 52.0, 5.0, 30.0, 10.0], [10.0, 62.0, 15.0, 40.0, 20.0]
    emb = np.arange(10, dtype=np.float32).reshape(5, 2)
    got = metrics.student_rows(index, vid, ts, te, emb)
    assert got.tolist() == [[8.0, 9.0], [6.0, 7.0], [2.0, 3.0], [4.0, 5.0], [0.0, 1.0]]   # rows: a 10, a 30, a 52, b, c


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


def test_right_answers_two_steps_same_text_one_video():
    r = metrics.right_answers(["a", "a"], ["add salt", "stir"], ["a", "b"], [{"add salt", "stir"}, {"add salt"}], "multi")
    assert r.tolist() == [[True, True], [True, False]]
    with pytest.raises(ValueError):
        metrics.right_answers(["a"], ["x"], ["a"], [set()], "loose")


def test_video_bootstrap_uses_the_query_videos():
    b = paired_bootstrap(np.array([1, 1, 2, 2]), np.array([2, 2, 2, 2]), 1, 200, 0, "video", ["a", "a", "b", "b"])
    assert b["diff"] == pytest.approx(50.0) and b["n_units"] == 2


# ---------------------------------------------------------------- UniAV adapter (real model, random weights)

UNIAV_REPO = os.environ.get("UNIAV_REPO", "")
needs_repo = pytest.mark.skipif(not UNIAV_REPO, reason="set UNIAV_REPO to a checkout of HanLi-Retriever")


@pytest.fixture(scope="module")
def uniav(tmp_path_factory):
    """A small HT-Step dataset on disk: two articles of two steps, one video of the first; a random model, its article
    conditioning switched on (it is zero at init, so it would otherwise hide the use of ``cand``)."""
    import torch

    from uemr_htstep import uniav_htstep as uh

    d = tmp_path_factory.mktemp("htstep")
    with open(d / "taxonomy.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["global_step_index", "activity", "variation", "headline"])
        w.writeheader()
        for i, (a, h) in enumerate([("A", "a0"), ("A", "a1"), ("B", "b0"), ("B", "b1")]):
            w.writerow({"global_step_index": 100 + i, "activity": a, "variation": "none", "headline": h})
    (d / "annotations.json").write_text(json.dumps({"v1": {"subset": "val_seen", "duration": 60.0, "activity": "A", "annotations": [
        {"segment": [5.0, 15.0], "id": 100, "partial": "Full Match"}, {"segment": [20.0, 40.0], "id": 101, "partial": "Full Match"}]}}))
    np.savez(d / "step_emb.npz", ids=np.arange(100, 104), emb=np.random.default_rng(0).normal(size=(4, 512)).astype(np.float32))
    (d / "k400").mkdir()
    np.savez(d / "k400" / "v1.npz", k768=np.random.default_rng(1).normal(size=(60, 768)).astype(np.float16),
             a768=np.random.default_rng(2).normal(size=(60, 768)).astype(np.float16))
    import sys

    sys.path.insert(0, UNIAV_REPO)
    from libs.core import cfg
    from libs.modeling import make_multimodal_meta_arch

    cfg["model"]["num_articles"] = 2
    torch.manual_seed(0)
    net = make_multimodal_meta_arch("EventCaptionTransformer", **cfg["model"])
    torch.nn.init.normal_(net.article_cond.attn.out_proj.weight, std=0.5)
    torch.save({"state_dict": net.state_dict()}, d / "ck.pth")
    model, ds = uh.load(UNIAV_REPO, d / "ck.pth", str(d / "annotations.json"), str(d / "taxonomy.csv"), str(d / "step_emb.npz"),
                        str(d / "k400"), str(d / "omni"), device="cpu")
    return model, ds, uh


@needs_repo
def test_main_mode_does_not_read_the_gt_article(uniav):
    """Plan §9: changing the GT ``cand`` of the video leaves the main mode unchanged, and changes the oracle mode."""
    import torch

    model, ds, uh = uniav
    item = ds[0]
    other = dict(item, cand=torch.tensor([2, 3]), article=1)
    a, b = uh.predict(model, item, "pred"), uh.predict(model, other, "pred")
    for key in ("segments", "scores", "labels"):
        assert torch.equal(a[key], b[key]), key
    assert a["article_top1"] == b["article_top1"] == a["article_pred"]
    assert torch.equal(a["feat0"], b["feat0"])
    g, h = uh.predict(model, item, "gt"), uh.predict(model, other, "gt")
    assert not torch.equal(g["feat0"], h["feat0"])       # guards the test: the oracle mode does read cand
    assert torch.equal(item["cand"], torch.tensor([0, 1]))    # the dataset item is not changed
    pred_cand = set(model.article_steps[a["article_top1"]])
    assert set(a["labels"].tolist()) <= pred_cand


@needs_repo
def test_seconds_steps_round_trip(uniav):
    _, ds, uh = uniav
    item = ds[0]
    sec = np.asarray(item["segments_sec"])
    steps = uh.sec_to_steps(sec, item["feat_stride"], item["fps"])
    assert np.allclose(steps, item["segments"].numpy())
    assert np.allclose(uh.steps_to_sec(steps, item["feat_stride"], item["fps"]), sec)


@needs_repo
def test_student_vectors_shape_and_norm(uniav):
    model, ds, uh = uniav
    p = uh.predict(model, ds[0], "pred")
    v = uh.student_vectors(model, p, [[0.0, 60.0], [5.0, 15.0], [20.0, 40.0]])
    assert v.shape == (3, 3584) and np.allclose(np.linalg.norm(v, axis=1), 1.0, atol=1e-4)


def test_no_size_cap_keeps_the_other_file_checks(tmp_path):
    from omni_retrieval.encode import no_size_cap

    class Loader:
        def _check_video_file(self, video_file):
            if not os.path.isfile(video_file):
                raise FileNotFoundError(video_file)
            if os.path.getsize(video_file) > 10:
                raise ValueError("Video too large")

    big, small = tmp_path / "big.mp4", tmp_path / "small.mp4"
    big.write_bytes(b"x" * 20)
    small.write_bytes(b"x" * 5)
    check = no_size_cap(Loader._check_video_file, cap=10)
    check(Loader(), str(big))                         # over the cap: no longer skipped
    check(Loader(), str(small))
    with pytest.raises(FileNotFoundError):
        check(Loader(), str(tmp_path / "missing.mp4"))


def test_train_teacher_mean(tmp_path):
    from uemr_htstep.run_uniav import train_teacher_mean

    np.savez(tmp_path / "a.npz", emb=np.array([[3.0, 0.0], [0.0, 2.0]], np.float16))
    np.savez(tmp_path / "b.npz", emb=np.array([[0.0, -5.0]], np.float16))
    assert np.allclose(train_teacher_mean(tmp_path), [1 / 3, 0.0])


def test_final_dev_pick_breaks_ties_toward_5_10():
    from uemr_htstep.final_dev import pick

    r = lambda lm, lf, v: {"l_min": lm, "l_fill": lf, "R@1 multi": v}
    assert pick([r(3, 8, 50.0), r(5, 15, 51.0), r(5, 10, 50.5)]) == r(5, 15, 51.0)          # highest wins
    assert pick([r(3, 8, 51.0), r(5, 10, 51.0), r(5, 15, 51.0)]) == r(5, 10, 51.0)          # tie: (5, 10)
    assert pick([r(3, 8, 51.0), r(3, 10, 51.0), r(5, 15, 40.0)]) == r(3, 10, 51.0)          # tie without (5, 10): closest
