"""SegmentIndex / metrics on fake vectors (no model, no GPU)."""

import numpy as np
import pytest

from omni_retrieval.evaluate import evaluate_partition, gallery_videos, paired_bootstrap, rank_metrics
from omni_retrieval.search import SegmentIndex, tiou, tiou_vec


def unit(*v):
    v = np.asarray(v, dtype=np.float32)
    return v / np.linalg.norm(v)


@pytest.fixture
def toy():
    # rows deliberately NOT grouped by video, to test the grouping
    emb = np.stack([
        unit(1, 0, 0),      # A seg 0  [0, 10]
        unit(0, 1, 0),      # B seg 0  [0, 5]
        unit(0.6, 0.8, 0),  # A seg 1  [10, 20]
        unit(0, 0, 1),      # C seg 0  [0, 30]
        unit(0.8, 0, 0.6),  # B seg 1  [5, 9]
    ])
    vid = np.array(["A", "B", "A", "C", "B"])
    ts = np.array([0, 0, 10, 0, 5], dtype=np.float32)
    te = np.array([10, 5, 20, 30, 9], dtype=np.float32)
    return SegmentIndex(emb, vid, ts, te, "toy")


def test_grouping(toy):
    assert list(toy.videos) == ["A", "B", "C"]
    assert list(toy.counts) == [2, 2, 1]
    assert toy.pad.shape == (3, 2) and toy.pad[2, 1] == -1
    # each video's segments are in time order
    for v in range(3):
        rows = toy.pad[v][toy.pad[v] >= 0]
        assert np.all(np.diff(toy.ts[rows]) >= 0)


def test_maxsim_scores_and_timestamps(toy):
    q = unit(1, 0, 0)
    # hand-computed cosines: A -> max(1.0, 0.6) = 1.0 @ [0,10]; B -> max(0, 0.8) = 0.8 @ [5,9]; C -> 0
    hits = toy.search(q, top_k=3)
    assert [h["video_id"] for h in hits] == ["A", "B", "C"]
    assert [round(h["score"], 4) for h in hits] == [1.0, 0.8, 0.0]
    assert [h["best_segment"] for h in hits] == [(0.0, 10.0), (5.0, 9.0), (0.0, 30.0)]
    assert [round(s, 4) for _, _, s in hits[1]["segment_scores"]] == [0.0, 0.8]

    q2 = unit(0, 1, 0)   # A -> 0.8 @ [10,20], B -> 1.0 @ [0,5]
    hits = toy.search(q2, top_k=2)
    assert [(h["video_id"], h["best_segment"]) for h in hits] == [("B", (0.0, 5.0)), ("A", (10.0, 20.0))]


def test_against_bruteforce_loop():
    rng = np.random.default_rng(0)
    n, d = 60, 16
    emb = rng.normal(size=(n, d)).astype(np.float32)
    emb /= np.linalg.norm(emb, axis=1, keepdims=True)
    vid = rng.choice([f"v{i}" for i in range(9)], size=n)
    ts = rng.uniform(0, 100, size=n).astype(np.float32)
    te = ts + 5
    index = SegmentIndex(emb, vid, ts, te)
    q = rng.normal(size=(7, d)).astype(np.float32)
    for agg in ("max", "topk_mean", "lse"):
        scores, best = index.score_matrix(q, agg=agg, k=2, tau=0.1, chunk=3)
        for j, v in enumerate(index.videos):
            s = q @ emb[vid == v].T                      # [7, k_v]
            if agg == "max":
                ref = s.max(1)
            elif agg == "topk_mean":
                ref = -np.sort(-s, 1)[:, :2].mean(1)
            else:
                ref = 0.1 * np.log(np.exp(s / 0.1).mean(1))
            np.testing.assert_allclose(scores[:, j], ref, rtol=1e-5, atol=1e-5)
            # argmax row points at the right timestamp
            np.testing.assert_allclose(index.ts[best[:, j]], ts[vid == v][s.argmax(1)])


def test_event_single_pooling(toy):
    pooled = toy.mean_pooled()
    assert list(pooled.counts) == [1, 1, 1]
    a = unit(1, 0, 0) + unit(0.6, 0.8, 0)
    np.testing.assert_allclose(pooled.emb[0], a / np.linalg.norm(a), atol=1e-6)
    assert (pooled.ts[0], pooled.te[0]) == (0, 20)


def test_rank_metrics_hand_computed():
    m = rank_metrics(np.array([1, 3, 12]))
    assert m["R@1"] == pytest.approx(100 / 3)
    assert m["R@5"] == pytest.approx(200 / 3)
    assert m["R@10"] == pytest.approx(200 / 3)
    assert m["MedR"] == 3 and m["MnR"] == pytest.approx(16 / 3)


def test_tiou():
    assert tiou((0, 10), (5, 15)) == pytest.approx(5 / 15)
    assert tiou((0, 10), (20, 30)) == 0
    np.testing.assert_allclose(tiou_vec(np.array([0, 0]), np.array([10, 10]),
                                        np.array([0, 5]), np.array([10, 15])), [1.0, 5 / 15])


def test_evaluate_partition_hand_computed(toy):
    # q0 -> A (rank 1), best seg A[0,10]; GT [0,8] -> tIoU 0.8 -> moment hit
    # q1 -> positive A, but B scores 1.0 > A 0.8 -> rank 2
    # q2 -> positive C (score 1.0, rank 1), best seg [0,30]; GT [0,10] -> tIoU 1/3 -> miss
    text = {"emb": np.stack([unit(1, 0, 0), unit(0, 1, 0), unit(0, 0, 1)]),
            "video_id": np.array(["A", "A", "C"]),
            "gt_ts": np.array([0, 10, 0], dtype=np.float32),
            "gt_te": np.array([8, 20, 10], dtype=np.float32)}
    row = evaluate_partition(toy, text)
    assert row["R@1"] == pytest.approx(200 / 3)
    assert row["MedR"] == 1 and row["MnR"] == pytest.approx(4 / 3)
    assert row["mom@.5|top1"] == pytest.approx(50.0)        # 1 of the 2 top-1 hits
    assert row["VCMR R@1 (tIoU.5)"] == pytest.approx(100 / 3)
    assert row["vec/video"] == pytest.approx(5 / 3)


def test_gallery_filter_hand_computed(toy, tmp_path):
    # without B, q1's positive A (0.8) is the best video left -> every query ranks 1
    text = {"emb": np.stack([unit(1, 0, 0), unit(0, 1, 0), unit(0, 0, 1)]),
            "video_id": np.array(["A", "A", "C"]),
            "gt_ts": np.array([0, 10, 0], dtype=np.float32),
            "gt_te": np.array([8, 20, 10], dtype=np.float32)}
    (tmp_path / "subset_videos.json").write_text(
        '{"videos": [{"video_id": "A", "split": "val"}, {"video_id": "B", "split": "dev"}, '
        '{"video_id": "C", "split": "val"}]}')
    videos = gallery_videos(tmp_path / "subset_videos.json", "val")
    assert videos == {"A", "C"} and gallery_videos(tmp_path / "subset_videos.json", "all") is None
    row = evaluate_partition(toy, text, videos=videos)
    assert row["videos"] == 2 and row["vec/video"] == pytest.approx(1.5)
    assert row["R@1"] == pytest.approx(100.0) and row["MnR"] == pytest.approx(1.0)
    assert toy.subset(videos).n_videos == 2 and toy.n_videos == 3      # the original index is untouched


def test_paired_bootstrap():
    a, b = np.array([1, 1, 2, 5]), np.array([1, 3, 2, 1])
    r = paired_bootstrap(a, b, n_boot=500)
    assert r["diff"] == pytest.approx(0.0)                    # hits a: 1,1,0,0  b: 1,0,0,1
    assert r["ci_low"] <= 0 <= r["ci_high"]
    same = paired_bootstrap(a, a)
    assert same["diff"] == same["ci_low"] == same["ci_high"] == 0
    best = paired_bootstrap(np.ones(10), np.full(10, 3))
    assert best["diff"] == best["ci_low"] == 100 and best["p(diff<=0)"] == 0


def test_paired_bootstrap_by_video():
    # video X: a hits both queries, b none; video Y: equal. diff = 2/5 = 40 points
    a = np.array([1, 1, 1, 4, 4])
    b = np.array([3, 2, 1, 4, 4])
    groups = np.array(["X", "X", "Y", "Y", "Y"])
    r = paired_bootstrap(a, b, n_boot=2000, unit="video", groups=groups)
    assert r["diff"] == pytest.approx(40.0) and r["n_units"] == 2 and r["unit"] == "video"
    # every resample is (X, X) -> 100, (Y, Y) -> 0 or one of each -> 40
    assert r["ci_low"] == pytest.approx(0.0) and r["ci_high"] == pytest.approx(100.0)
    one = paired_bootstrap(a[:2], b[:2], unit="video", groups=groups[:2])   # a single video: no spread
    assert one["diff"] == one["ci_low"] == one["ci_high"] == 100
    with pytest.raises(ValueError):
        paired_bootstrap(a, b, unit="video")
    with pytest.raises(ValueError):
        paired_bootstrap(a, b, unit="caption", groups=groups)


def test_paired_bootstrap_mean_of_seeds():
    b = np.array([1, 3, 2, 1, 5, 1])
    s0 = np.array([1, 1, 2, 1, 5, 3])            # hits 1,1,0,1,0,0 vs b 1,0,0,1,0,1 -> 0
    s1 = np.array([1, 1, 1, 1, 1, 1])            # all hits -> +50
    groups = np.array(["X", "X", "Y", "Y", "Z", "Z"])
    mean = paired_bootstrap(np.stack([s0, s1]), b, n_boot=500, unit="video", groups=groups)
    one = [paired_bootstrap(s, b, n_boot=500, unit="video", groups=groups)["diff"] for s in (s0, s1)]
    assert mean["diff"] == pytest.approx(np.mean(one)) == pytest.approx(25.0)
    twice = paired_bootstrap(np.stack([s0, s0]), b, n_boot=500, unit="video", groups=groups)
    alone = paired_bootstrap(s0, b, n_boot=500, unit="video", groups=groups)
    assert (twice["diff"], twice["ci_low"], twice["ci_high"]) == (alone["diff"], alone["ci_low"], alone["ci_high"])


def test_segment_index_bias(toy):
    # the toy rows in their input order; bias is given in that order (rows 1 and 4 are video B)
    emb = np.stack([unit(1, 0, 0), unit(0, 1, 0), unit(0.6, 0.8, 0), unit(0, 0, 1), unit(0.8, 0, 0.6)])
    vid = np.array(["A", "B", "A", "C", "B"])
    ts = np.array([0, 0, 10, 0, 5], np.float32)
    te = np.array([10, 5, 20, 30, 9], np.float32)
    biased = SegmentIndex(emb, vid, ts, te, "b", np.array([0.0, 0.5, 0.0, 0.0, 0.5], np.float32))
    q = unit(0.75, 0.5, 0.43)[None]
    plain, _ = toy.score_matrix(q)
    s, _ = biased.score_matrix(q)
    assert toy.bias is None
    np.testing.assert_allclose(s[0], plain[0] + np.array([0.0, 0.5, 0.0]), rtol=1e-6)
    assert biased.subset(["B", "C"]).bias.tolist() == [0.5, 0.5, 0.0]


def test_text_to_clip_matched(tmp_path):
    from omni_retrieval.evaluate import text_to_clip_matched
    rng = np.random.default_rng(0)
    vids = [f"v{i}" for i in range(6)]
    keys, emb, caps = [], [], []
    for v in vids:
        for j in range(3):                                   # 3 GT clips per video, one caption each
            keys.append(f"{v}__{j:.2f}_{j + 1:.2f}")
            emb.append(rng.normal(size=8))
            caps.append((f"{v}_{j}", v, keys[-1]))
    emb = np.asarray(emb, np.float32) / np.linalg.norm(emb, axis=1, keepdims=True)
    np.savez(tmp_path / "segments.npz", seg_key=np.array(keys), partition=np.array(["gt"] * len(keys)),
             video_id=np.array([k.split("__")[0] for k in keys]), emb=emb.astype(np.float16))
    np.savez(tmp_path / "text.npz", caption_id=np.array([c[0] for c in caps]), video_id=np.array([c[1] for c in caps]),
             gt_seg_key=np.array([c[2] for c in caps]), emb=emb.astype(np.float16))   # caption = its own clip
    rows = text_to_clip_matched(tmp_path, n_videos=4, n_rep=5, seed=1)
    assert len(rows) == 5 and all(r["queries"] == r["gallery"] == 12 and r["R@1"] == 100 for r in rows)
    with pytest.raises(ValueError):
        text_to_clip_matched(tmp_path, n_videos=7, n_rep=1)
