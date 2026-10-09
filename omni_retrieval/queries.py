"""Custom (hand-written) queries with a GT event, for testing retrieval beyond the val captions.

File format: JSONL (or a JSON list), one query per line::

    {"query": "bring a pot of water to a boil", "video_id": "yxjnWx6TaQ8", "ts": 23.0, "te": 29.0}

``ts``/``te`` is the GT window: where in the video the described step happens. Extra fields
(e.g. ``source_caption``) are kept and ignored. The GT window is matched to a database event
of the same video by largest tIoU (>= ``min_tiou``), so it does not have to equal the YouCook2
segment to the second. Queries whose video is not in the database (e.g. in smoke mode), or
whose window overlaps no event, are reported and left out of the metrics.

Vectors are cached by the text itself (``q_<md5>``): editing a query re-encodes only it.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from .search import EventIndex, tiou_vec


def text_key(text: str) -> str:
    return "q_" + hashlib.md5(text.strip().encode("utf-8")).hexdigest()[:12]


def load_queries(path) -> list[dict]:
    raw = Path(path).read_text(encoding="utf-8").strip()
    rows = json.loads(raw) if raw.startswith("[") else [json.loads(l) for l in raw.splitlines() if l.strip()]
    out = []
    for i, r in enumerate(rows):
        missing = [k for k in ("query", "video_id", "ts", "te") if k not in r]
        if missing:
            raise ValueError(f"query #{i} lacks {missing}: {r}")
        out.append(dict(r, qid=f"custom_{i:03d}", key=text_key(r["query"]),
                        ts=float(r["ts"]), te=float(r["te"])))
    return out


def encode_records(queries: list[dict]) -> list[dict]:
    """Manifest for ``python -m omni_retrieval.encode --modality text`` (one record per distinct text)."""
    seen, out = set(), []
    for q in queries:
        if q["key"] not in seen:
            seen.add(q["key"])
            out.append({"id": q["key"], "text": q["query"].strip()})
    return out


def resolve_gt(queries: list[dict], index: EventIndex, min_tiou: float = 0.5) -> list[dict]:
    """Attach the database row of each query's GT event (``pos``, -1 when unresolved) and why."""
    out = []
    for q in queries:
        rows = np.where(index.video_id == q["video_id"])[0]
        if not len(rows):
            out.append(dict(q, pos=-1, gt_tiou=0.0, status="video not in database"))
            continue
        ov = tiou_vec(index.ts[rows], index.te[rows], q["ts"], q["te"])
        j = int(np.argmax(ov))
        ok = ov[j] >= min_tiou
        out.append(dict(q, pos=int(rows[j]) if ok else -1, gt_tiou=float(ov[j]),
                        status="ok" if ok else f"no event with tIoU >= {min_tiou} (best {ov[j]:.2f})"))
    return out


def evaluate_custom(cache_dir, queries_path, text_store, min_tiou: float = 0.5):
    """Evaluate hand-written queries against ``events.npz``.

    Returns ``(summary, per_query, skipped)``. ``text_store`` holds the query vectors
    (``q_<md5>__text``) written by the text encoder.
    """
    from .evaluate import score_event_queries
    from .manifest import read_store

    index = EventIndex.load(cache_dir)
    queries = resolve_gt(load_queries(queries_path), index, min_tiou)
    vecs = read_store(text_store, "text")
    usable = [q for q in queries if q["pos"] >= 0 and q["key"] in vecs]
    skipped = [dict(q, status=q["status"] if q["pos"] < 0 else "not encoded yet")
               for q in queries if not (q["pos"] >= 0 and q["key"] in vecs)]
    if not usable:
        return {"queries": 0}, [], skipped
    emb = np.stack([vecs[q["key"]] for q in usable]).astype(np.float32)
    summary, per_query = score_event_queries(
        index, emb, [q["video_id"] for q in usable], [q["ts"] for q in usable],
        [q["te"] for q in usable], [q["pos"] for q in usable],
        [q["qid"] for q in usable], [q["query"] for q in usable])
    for row, q in zip(per_query, usable):
        row["gt_tiou_to_event"] = round(q["gt_tiou"], 2)
        if "source_caption" in q:
            row["source_caption"] = q["source_caption"]
    summary["skipped"] = len(skipped)
    return summary, per_query, skipped
