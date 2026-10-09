"""analysis.py on a hand-built database (no model, no GPU, no sentence-transformers)."""

import json

import numpy as np
import pytest

from omni_retrieval.analysis import (ERROR_TYPES, analyze, normalize_caption, run_analysis,
                                     score_queries)
from omni_retrieval.evaluate import score_event_queries
from omni_retrieval.queries import text_key
from omni_retrieval.search import EventIndex


def unit(*v):
    v = np.asarray(v, dtype=np.float32)
    return v / np.linalg.norm(v)


# 5 events; Omni vectors = basis vectors so query vectors fix the ranking by hand
CAPTIONS = ["add oil to a pan", "chop the onion", "add oil to the pan", "boil water", "Add oil to a pan!"]
VIDEOS = ["A", "A", "B", "B", "C"]
TS = [0, 10, 0, 5, 0]
TE = [10, 20, 5, 40, 100]
# judge: "add oil to the pan" is a paraphrase (cos 0.85) of "add oil to a pan"; "!" / case is exact
G1 = unit(1, 0, 0)
JUDGE = {"add oil to a pan": G1, "add oil to the pan": unit(0.85, np.sqrt(1 - 0.85 ** 2), 0),
         "chop the onion": unit(0, 0, 1), "boil water": unit(0, 1, 0)}


def fake_judge(texts):
    return np.stack([JUDGE[normalize_caption(t)] for t in texts])


fake_judge.name = "fake"


def e(i):
    v = np.zeros(5, dtype=np.float32)
    v[i] = 1
    return v


@pytest.fixture
def index():
    return EventIndex(np.eye(5, dtype=np.float32), [f"{v}__{a}_{b}" for v, a, b in zip(VIDEOS, TS, TE)],
                      VIDEOS, TS, TE, CAPTIONS)


# q0: GT event 0 (A), top-1 = event 4 (C, same caption up to case/punct) -> other video, same meaning
# q1: GT event 1, top-1 = itself                                      -> correct
# q2: GT event 3 (B), top-1 = event 2 (B, "add oil to the pan")       -> same video, other event
# q3: GT event 0 (A), top-1 = event 3 (B, "boil water")               -> other video, different meaning
Q = np.stack([unit(*(0.5 * e(0) + 0.9 * e(4) + 0.1 * e(2))), unit(*(e(1) + 0.5 * e(0))),
              unit(*(e(2) + 0.3 * e(3))), unit(*(e(3) + 0.2 * e(0)))])
POS = np.array([0, 1, 3, 0])


def judge_inputs(index):
    cap = fake_judge(list(index.caption))
    return cap, np.array([normalize_caption(c) for c in index.caption])


def test_ranks_match_score_event_queries(index):
    cap, norm = judge_inputs(index)
    r = score_queries(index, Q, POS, cap, norm, tau=0.8, top_k=3, block=2)
    _, pq = score_event_queries(index, Q, np.array(VIDEOS)[POS], np.array(TS)[POS], np.array(TE)[POS],
                                POS, range(4), ["q"] * 4)
    assert r["ranks"].tolist() == [p["rank"] for p in pq] == [2, 1, 2, 2]
    assert r["top"][:, 0].tolist() == [4, 1, 2, 3]


def test_near_duplicates(index):
    cap, norm = judge_inputs(index)
    r = score_queries(index, Q, POS, cap, norm, tau=0.8, top_k=3)
    # event 0 has two same-meaning events (2 at 0.85, 4 at 1.0), both in other videos
    assert r["n_dup"].tolist() == [2, 0, 0, 2]
    assert r["n_dup_other"].tolist() == [2, 0, 0, 2]
    r9 = score_queries(index, Q, POS, cap, norm, tau=0.9, top_k=3)
    assert r9["n_dup_other"].tolist() == [1, 0, 0, 1]          # 0.85 paraphrase drops out


def test_semantic_metrics_and_error_types(index):
    cap, norm = judge_inputs(index)
    rep = analyze(index, Q, POS, range(4), ["a b", "a b c d e", "x", "y"], cap, norm,
                  taus=(0.8, 0.9), tau=0.8, top_k=3)
    s = rep["summary"]
    assert s["event R@1"] == pytest.approx(25.0)
    assert s["exact-caption@1"] == pytest.approx(50.0)               # q0 (exact string), q1 (itself)
    assert s["sem@1 (tau=0.8)"] == pytest.approx(50.0)
    assert s["misses that are same-meaning (tau=0.8) %"] == pytest.approx(100 * 25 / 75)
    assert s["queries with >=1 near-duplicate in another video (tau=0.8) %"] == pytest.approx(50.0)
    assert s["sem@1 (tau=0.9)"] == pytest.approx(50.0)
    types = [q["error type"] for q in rep["per_query"]]
    assert types == [ERROR_TYPES[2], ERROR_TYPES[0], ERROR_TYPES[1], ERROR_TYPES[3]]
    assert [x["n"] for x in rep["error_types"]] == [1, 1, 1, 1]


def test_group_tables_and_hubs(index):
    cap, norm = judge_inputs(index)
    rep = analyze(index, Q, POS, range(4), ["a b", "a b c d e", "x", "y"], cap, norm, tau=0.8, top_k=3)
    dup = {row[next(iter(row))]: row for row in rep["by_duplicates"]}
    assert dup["0"]["n"] == 2 and dup["1-2"]["n"] == 2
    assert dup["0"]["R@1"] == pytest.approx(50.0) and dup["1-2"]["R@1"] == 0
    assert dup["1-2"]["sem@1"] == pytest.approx(50.0)
    assert sum(row["n"] for row in rep["by_duration"]) == 4
    lens = {row["query length"]: row["n"] for row in rep["by_query_len"]}
    assert lens == {"1-4 words": 3, "5-8 words": 1}
    assert sorted(h["top1 count"] for h in rep["hubs"]) == [1, 1, 1, 1]
    assert sum(row["% top-1 slots"] for row in rep["hub_by_duration"]) == pytest.approx(100.0)
    assert rep["summary"]["events never in any top-k %"] == pytest.approx(0.0)
    assert rep["tail"] == []


def test_run_analysis_end_to_end(index, tmp_path):
    cache = tmp_path / "cache"
    cache.mkdir()
    np.savez(cache / "events.npz", emb=index.emb.astype(np.float16), seg_key=index.seg_key,
             video_id=index.video_id, ts=index.ts, te=index.te, caption=index.caption,
             split=np.array(["val"] * 5))
    np.savez(cache / "text.npz", caption_id=np.array([f"c{i}" for i in range(4)]),
             video_id=np.array(VIDEOS)[POS], sentence=np.array(CAPTIONS)[POS],
             gt_ts=np.array(TS, dtype=np.float32)[POS], gt_te=np.array(TE, dtype=np.float32)[POS],
             gt_seg_key=index.seg_key[POS], emb=Q.astype(np.float16))
    # custom: one paraphrase of event 1, one query whose video is not in the database
    qfile = tmp_path / "q.jsonl"
    rows = [{"query": "dice an onion", "video_id": "A", "ts": 10.0, "te": 20.0},
            {"query": "unknown", "video_id": "Z", "ts": 0.0, "te": 5.0}]
    qfile.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
    store = tmp_path / "custom_text"
    store.mkdir()
    np.savez(store / "chunk_a.npz", **{f"{text_key('dice an onion')}__text": e(1)})

    out = tmp_path / "analysis"
    rep = run_analysis(cache, fake_judge, taus=(0.8,), tau=0.8, custom_queries=qfile,
                       custom_text_store=store, out_dir=out, top_k=3)
    assert rep["gt"]["summary"]["event R@1"] == pytest.approx(25.0)
    assert rep["custom"]["summary"]["queries"] == 1 and rep["custom"]["summary"]["event R@1"] == 100.0
    assert (out / "gt_summary.json").is_file() and (out / "gt_per_query.csv").is_file()
    assert (out / "custom_per_query.csv").is_file() and (out / "settings.json").is_file()
    assert json.loads((out / "gt_summary.json").read_text(encoding="utf-8"))["summary"]["queries"] == 4
