"""A2 (aggregation kwargs, index size, int8, latency row) and A4 (qualitative tables, timelines) on a fake cache."""

import json
import os

import matplotlib
import numpy as np
import pandas as pd
import pytest

matplotlib.use("Agg")

from omni_retrieval.efficiency import efficiency_row, index_bytes, int8_index, quantize_int8  # noqa: E402
from omni_retrieval.evaluate import evaluate_partition, load_index, load_text, video_ranks  # noqa: E402
from omni_retrieval.qualitative import flip_groups, jaccard, profile, run, same_video_wrong_event  # noqa: E402
from omni_retrieval.search import tiou  # noqa: E402

D = 16
DUR = {"A": 60.0, "B": 80.0, "C": 50.0}
GT = {"A": [[0.0, 10.0], [12.0, 30.0], [35.0, 58.0]], "B": [[5.0, 20.0], [22.0, 40.0], [50.0, 78.0]],
      "C": [[0.0, 6.0], [10.0, 30.0], [32.0, 48.0]]}


def _uniform(d, m):
    e = np.linspace(0.0, d, m + 1)
    return [[float(a), float(b)] for a, b in zip(e[:-1], e[1:])]


PARTS = {"global": {v: [[0.0, d]] for v, d in DUR.items()}, "gt": GT,
         "uni_M": {v: _uniform(d, 3) for v, d in DUR.items()}, "uni_chunk": {v: _uniform(d, 5) for v, d in DUR.items()},
         "uniav_pred": {v: [[ts + 1.0, te - 1.0] for ts, te in GT[v]] for v in GT}}


def _unit(x):
    return x / np.linalg.norm(x, axis=-1, keepdims=True)


@pytest.fixture
def cache(tmp_path):
    """segments / text / events npz, partition jsons, subset and near-duplicate csv; a segment's vector is the
    overlap-weighted mix of the GT step vectors of its video, a caption's vector its step vector plus noise."""
    rng = np.random.default_rng(0)
    step = {(v, i): rng.normal(size=D) for v in GT for i in range(3)}
    rows = []
    for name, part in PARTS.items():
        for v, segs in part.items():
            for ts, te in segs:
                e = sum(tiou((ts, te), tuple(g)) * step[v, i] for i, g in enumerate(GT[v])) + 0.2 * rng.normal(size=D)
                rows.append((name, v, ts, te, e))
    np.savez(tmp_path / "segments.npz", partition=np.array([r[0] for r in rows]), video_id=np.array([r[1] for r in rows]),
             ts=np.array([r[2] for r in rows], np.float32), te=np.array([r[3] for r in rows], np.float32),
             emb=_unit(np.stack([r[4] for r in rows])).astype(np.float16))
    caps = [(f"{v}_{i}", v, f"step {i} of {v}", *GT[v][i]) for v in GT for i in range(3)]
    np.savez(tmp_path / "text.npz", caption_id=np.array([c[0] for c in caps]), video_id=np.array([c[1] for c in caps]),
             sentence=np.array([c[2] for c in caps]), gt_ts=np.array([c[3] for c in caps], np.float32),
             gt_te=np.array([c[4] for c in caps], np.float32), gt_seg_key=np.array([""] * len(caps)),
             emb=_unit(np.stack([step[c[1], int(c[0][-1])] + 0.5 * rng.normal(size=D) for c in caps])).astype(np.float16))
    np.savez(tmp_path / "events.npz", seg_key=np.array([c[0] for c in caps]), video_id=np.array([c[1] for c in caps]),
             ts=np.array([c[3] for c in caps], np.float32), te=np.array([c[4] for c in caps], np.float32),
             caption=np.array([c[2] for c in caps]), emb=_unit(np.stack([step[c[1], int(c[0][-1])] for c in caps])))
    parts_dir = tmp_path / "partitions"
    parts_dir.mkdir()
    for name in ("gt", "uni_M", "uniav_pred"):
        (parts_dir / f"{name}.json").write_text(json.dumps(PARTS[name]))
    (tmp_path / "subset_videos.json").write_text(json.dumps(
        {"videos": [{"video_id": v, "split": "val", "duration": d} for v, d in DUR.items()]}))
    pd.DataFrame({"query_id": [c[0] for c in caps], "near-duplicates (other videos)": [0, 12, 25, 3, 15, 0, 11, 1, 29]}
                 ).to_csv(tmp_path / "dup.csv", index=False)
    return tmp_path


def test_index_bytes_and_int8():
    size = index_bytes(1000, 3584)
    assert size["fp16 (MB)"] == pytest.approx(1000 * 3584 * 2 / 1024 ** 2)
    assert size["fp32 (MB)"] == pytest.approx(2 * size["fp16 (MB)"])
    x = _unit(np.random.default_rng(1).normal(size=(50, 32))).astype(np.float32)
    codes, scale = quantize_int8(x)
    assert codes.dtype == np.int8 and np.abs(codes).max() == 127
    assert np.all(np.abs(codes * scale[:, None] - x) <= scale[:, None] / 2 + 1e-7)
    assert np.allclose(quantize_int8(np.zeros((1, 4)))[0], 0)                       # a zero row stays zero


def test_aggregation_kwargs_reach_the_scorer(cache):
    text, idx = load_text(cache), load_index(cache, "uni_chunk")
    r_max = video_ranks(idx, text)["ranks"]
    assert np.array_equal(video_ranks(idx, text, "topk_mean", k=1)["ranks"], r_max)     # k = 1 is max
    assert np.array_equal(video_ranks(idx, text, "lse", tau=1e-4)["ranks"], r_max)      # tau -> 0 is max
    k3 = evaluate_partition(idx, text, "topk_mean", k=3)
    k2 = evaluate_partition(idx, text, "topk_mean", k=2)
    assert k3["queries"] == k2["queries"] == 9


def test_efficiency_row_and_int8_index(cache):
    text, idx = load_text(cache), load_index(cache, "uni_chunk")
    row = efficiency_row(idx, text["emb"], needs_uniav=False, gpu=False)
    assert (row["vectors"], row["vec/video"], row["d"]) == (15, 5.0, D)
    assert row["Omni s/video"] == pytest.approx(5 * 0.26) and row["CPU 1 query (ms)"] > 0
    assert "UniAV s/video" not in row
    assert efficiency_row(idx, text["emb"], needs_uniav=True, uniav_sec_per_video=1.5, gpu=False)["UniAV s/video"] == 1.5
    q = int8_index(idx)
    assert q.emb.shape == idx.emb.shape and np.abs(q.emb - idx.emb).max() < 0.01


def test_qualitative_run(cache):
    out = run(cache, None, cache / "partitions", cache / "subset_videos.json", cache / "dup.csv", cache / "qual",
              n_examples=4, n_timelines=3)
    df = out["per_query"]
    assert len(df) == 9 and df.near_dup.tolist()[:3] == [0, 12, 25]
    assert np.array_equal(df["rank R6"], video_ranks(load_index(cache, "uniav_pred"), load_text(cache))["ranks"])
    a0 = df.set_index("caption_id").loc["A_0"]
    assert a0["cover tIoU R6"] == pytest.approx(0.8) and a0["K R6"] == 3          # [1, 9] inside [0, 10]
    assert a0["rel_pos"] == pytest.approx(5 / 60)
    groups = flip_groups(df)
    assert groups["R6 đúng, R3 sai"].sum() == ((df["rank R6"] == 1) & (df["rank R3"] > 1)).sum()
    prof = profile(df, groups)
    assert prof.iloc[0]["queries"] == 9 and len(prof) == 7
    for name in ("per_query", "profile", "r6_beats_r3", "r6_beats_r0", "r0_r2_beat_r6", "near_dup_10_29",
                 "same_video_wrong_event"):
        assert (cache / "qual" / f"qual_{name}.csv").is_file(), name
    assert 1 <= len(out["figures"]) <= 3 and all(os.path.isfile(f) for f in out["figures"])
    nd = out["near_dup_10_29"]
    if len(nd):
        assert set(nd.top) <= {1, 2, 3} and nd["caption GT gần argmax nhất"].str.startswith("step").all()


def test_same_video_wrong_event(cache):
    assert jaccard("Stir the onions in the pan", "stir onions and garlic") == pytest.approx(2 / 4)   # stir, onions
    assert jaccard("", "") == 0.0
    # A_0 found at the right video (rank 1) but its argmax is step 1 of A: a "same video, other event" error
    df = pd.DataFrame([{"caption_id": "A_0", "video_id": "A", "sentence": "step 0 of A", "gt_ts": 0.0, "gt_te": 10.0,
                        "rank R7": 1, "ts R7": 12.0, "te R7": 30.0, "moment tIoU R7": 0.0},
                       {"caption_id": "A_1", "video_id": "A", "sentence": "step 1 of A", "gt_ts": 12.0, "gt_te": 30.0,
                        "rank R7": 1, "ts R7": 12.0, "te R7": 30.0, "moment tIoU R7": 1.0}])   # right event: not kept
    out = same_video_wrong_event(cache, df, parts=("R7",))
    assert out.caption_id.tolist() == ["A_0"]
    r = out.iloc[0]
    assert (r["caption gần argmax nhất"], r["tIoU argmax với caption đó"], r["jaccard"]) == ("step 1 of A", 1.0, 1.0)
