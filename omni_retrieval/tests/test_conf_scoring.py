"""Exp 5: conf of segments_pred.jsonl matched to the cache rows, beta * log conf scoring (no model)."""

import json

import numpy as np
import pytest

from omni_retrieval.conf_scoring import align_conf, conf_rows, index_for, pick_beta, read_conf, spread, sweep
from omni_retrieval.evaluate import evaluate_partition
from omni_retrieval.partitions_uemr import segment_stats

PRED = [  # video, ts, te, conf  (jsonl order: by video, by time)
    ("A", 0.0, 10.004, 0.9), ("A", 12.3, 20.0, 0.95), ("A", 25.0, 31.7, 0.7),   # A's last te clipped to 30.0
    ("B", 5.0, 9.0, 0.6), ("B", 1.0, 4.5, 0.8),                                 # B out of order in the file
]


def _jsonl(path, rows=PRED):
    path.write_text("".join(json.dumps({"video_id": v, "ts": ts, "te": te, "conf": c}) + "\n" for v, ts, te, c in rows))
    return path


def test_align_conf(tmp_path):
    conf = read_conf(_jsonl(tmp_path / "p.jsonl"))
    assert [c for *_, c in conf["B"]] == [0.8, 0.6]                                # sorted by time
    # cache rows: rounded to 0.01 s, shuffled, A's last te clipped to the 30 s duration
    vid = np.array(["B", "A", "A", "B", "A"])
    ts = np.array([5.0, 25.0, 0.0, 1.0, 12.3])
    te = np.array([9.0, 30.0, 10.0, 4.5, 20.0])
    np.testing.assert_allclose(align_conf(vid, ts, te, conf), [0.6, 0.7, 0.9, 0.8, 0.95])
    with pytest.raises(ValueError, match="segments in the cache"):
        align_conf(vid[:4], ts[:4], te[:4], conf)                                  # A lost a segment
    with pytest.raises(ValueError, match="in segments_pred"):
        align_conf(vid, np.array([5.0, 25.0, 0.0, 1.0, 13.0]), te, conf)           # ts moved by 0.7 s
    with pytest.raises(ValueError, match="in segments_pred"):
        align_conf(vid, ts, np.array([9.0, 30.0, 8.0, 4.5, 20.0]), conf)           # a te shorter but not the last
    zero = read_conf(_jsonl(tmp_path / "z.jsonl", [("A", 0.0, 1.0, 0.0)]))
    with pytest.raises(ValueError, match="conf <= 0"):
        align_conf(np.array(["A"]), np.array([0.0]), np.array([1.0]), zero)


def _cache(tmp_path):
    """segments.npz with uniav_pred rows of PRED (+ a gt row that must be ignored) and a text.npz."""
    emb = np.eye(5, 8, dtype=np.float32)
    emb[1] = emb[0]                                         # A seg 1 duplicates A seg 0: a cosine tie inside A
    rows = [("A", 0.0, 10.0), ("A", 12.3, 20.0), ("A", 25.0, 30.0), ("B", 1.0, 4.5), ("B", 5.0, 9.0)]
    np.savez(tmp_path / "segments.npz", partition=np.array(["uniav_pred"] * 5 + ["gt"]),
             video_id=np.array([r[0] for r in rows] + ["A"]), ts=np.array([r[1] for r in rows] + [0.0], np.float32),
             te=np.array([r[2] for r in rows] + [10.0], np.float32),
             emb=np.vstack([emb, emb[:1]]).astype(np.float16))
    # query 0 -> A (matches A seg 0 and seg 1 equally), query 1 -> B (B seg 1)
    text = {"emb": np.stack([emb[0], emb[4]]), "video_id": np.array(["A", "B"]),
            "gt_ts": np.array([12.3, 5.0], np.float32), "gt_te": np.array([20.0, 9.0], np.float32)}
    return text


def test_beta_zero_is_column_b_and_bias_moves_argmax(tmp_path):
    text = _cache(tmp_path)
    rows = conf_rows(tmp_path, _jsonl(tmp_path / "p.jsonl"))
    assert len(rows["video_id"]) == 5 and rows["log_conf"].max() < 0
    assert index_for(rows, 0.0).bias is None
    b0, b1 = sweep(rows, text, betas=(0.0, 1.0))
    plain = evaluate_partition(index_for(rows, 0.0), text, "max")
    assert {k: b0[k] for k in plain} == plain and b0["beta"] == 0.0
    # A seg 0 and seg 1 tie on cosine; beta 0 takes seg 0 (wrong moment), beta 1 takes seg 1 (conf 0.95 > 0.9)
    assert b1["R@1"] == b0["R@1"] == 100
    assert b0["VCMR R@1 (tIoU.5)"] == 50 and b1["VCMR R@1 (tIoU.5)"] == 100
    assert pick_beta([dict(b0, **{"R@1": 50.0}), dict(b1, **{"R@1": 50.0})]) == 0.0      # tie -> smaller beta
    assert pick_beta([dict(b0, **{"R@1": 40.0}), dict(b1, **{"R@1": 50.0})]) == 1.0
    sp = spread(rows, text, videos={"A"})
    assert sp["segments"] == 3 and sp["log conf max"] == pytest.approx(np.log(0.95))


def test_segment_stats():
    st = segment_stats({"A": [[0, 10], [5, 15]], "B": [[0, 20]], "C": [[0, 1]]}, {"A": 30.0, "B": 40.0})
    assert st["videos"] == 2 and st["K mean"] == 1.5 and st["len median (s)"] == 10.0
    assert st["coverage mean (%)"] == pytest.approx(100 * (15 / 30 + 20 / 40) / 2)
