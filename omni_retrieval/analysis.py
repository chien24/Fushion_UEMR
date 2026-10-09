"""Error analysis and a "same meaning" metric for event retrieval (follow-up to the first ft/pt run).

Plain ``event R@k`` accepts exactly one event per query. YouCook2 repeats the same step in many
videos ("add oil to a pan"), so many misses are the right step in another video. Two tools:

1. **Semantic hit** (``sem@k``): a returned event also counts when its caption means the same as
   the GT caption -- cosine >= ``tau`` under an *independent* sentence encoder (the judge, not
   Omni) -- and ``exact-caption@k`` when the normalised caption strings are identical. The
   judge only reads annotations; it is an evaluation device, never part of retrieval.
2. **Where the misses come from**: ranks broken down by GT event duration, by how many
   near-duplicate events (same meaning, other video) the GT has in the database, and by query
   length; top-1 error types; hub events (returned as top-1 far more often than chance) and
   hubness per event duration; the long tail (rank > 100); judge calibration pairs.

    from omni_retrieval.analysis import caption_embedder, run_analysis
    report = run_analysis(cache_dir, caption_embedder(), out_dir=cache_dir / "analysis")
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Callable

import numpy as np

from .evaluate import QUERY_BLOCK, load_text, rank_metrics, save_csv
from .search import EventIndex

DEFAULT_JUDGE = "sentence-transformers/all-MiniLM-L6-v2"
TAUS = (0.7, 0.8, 0.9)
TAU = 0.8           # threshold for error types, near-duplicate counts and groups
TOP_K = 10
TAIL_RANK = 100

DURATION_BINS = (("<5s", 0, 5), ("5-10s", 5, 10), ("10-20s", 10, 20), ("20-40s", 20, 40),
                 ("40-80s", 40, 80), (">=80s", 80, np.inf))
DUPLICATE_BINS = (("0", 0, 1), ("1-2", 1, 3), ("3-9", 3, 10), ("10-29", 10, 30), (">=30", 30, np.inf))
QUERY_LEN_BINS = (("1-4 words", 1, 5), ("5-8 words", 5, 9), ("9-12 words", 9, 13), (">=13 words", 13, np.inf))
CALIBRATION_BINS = ((0.5, 0.6), (0.6, 0.7), (0.7, 0.8), (0.8, 0.9), (0.9, 1.0001))

ERROR_TYPES = ("correct", "same video, other event", "other video, same meaning",
               "other video, different meaning")


# --------------------------------------------------------------------------- #
# Caption judge                                                               #
# --------------------------------------------------------------------------- #

def normalize_caption(s: str) -> str:
    return " ".join(re.sub(r"[^a-z0-9]+", " ", str(s).lower()).split())


def caption_embedder(model_name: str = DEFAULT_JUDGE, device: str | None = None,
                     batch_size: int = 256) -> Callable[[list[str]], np.ndarray]:
    """``f(captions) -> [n, d]`` L2-normalised, from a sentence-transformers model."""
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(model_name, device=device)

    def embed(texts: list[str]) -> np.ndarray:
        return model.encode(list(texts), batch_size=batch_size, normalize_embeddings=True,
                            convert_to_numpy=True, show_progress_bar=False).astype(np.float32)

    embed.name = model_name
    return embed


def _unit(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    return x / np.clip(np.linalg.norm(x, axis=1, keepdims=True), 1e-12, None)


# --------------------------------------------------------------------------- #
# Core                                                                        #
# --------------------------------------------------------------------------- #

def _topk(sim: np.ndarray, k: int) -> np.ndarray:
    part = np.argpartition(-sim, k - 1, axis=1)[:, :k]
    order = np.argsort(-np.take_along_axis(sim, part, 1), axis=1, kind="stable")
    return np.take_along_axis(part, order, 1)


def score_queries(index: EventIndex, q_emb, pos, cap_emb, cap_norm, tau: float = TAU,
                  top_k: int = TOP_K, block: int = QUERY_BLOCK) -> dict:
    """Ranks (strict, as ``evaluate.score_event_queries``), top-k rows, and the judge's view.

    ``cap_emb[j]`` / ``cap_norm[j]``: judge vector / normalised string of event j's caption.
    The GT caption of query i is the caption of its GT event ``pos[i]``.
    """
    pos = np.asarray(pos)
    n, N = len(pos), len(index)
    k = min(top_k, N)
    ranks = np.empty(n, dtype=np.int64)
    top = np.empty((n, k), dtype=np.int64)
    n_dup = np.empty(n, dtype=np.int64)
    n_dup_other = np.empty(n, dtype=np.int64)
    q_emb = np.asarray(q_emb, dtype=np.float32)
    for s in range(0, n, block):
        blk = slice(s, s + block)
        sim = index.scores(q_emb[blk])                                        # [b, N] Omni
        q = np.arange(sim.shape[0])
        ranks[blk] = 1 + (sim > sim[q, pos[blk]][:, None]).sum(1)
        top[blk] = _topk(sim, k)
        dup = (cap_emb[pos[blk]] @ cap_emb.T) >= tau                          # [b, N] judge
        dup[q, pos[blk]] = False
        n_dup[blk] = dup.sum(1)
        other = index.video_id[None, :] != index.video_id[pos[blk]][:, None]
        n_dup_other[blk] = (dup & other).sum(1)
    capsim = np.einsum("nd,nkd->nk", cap_emb[pos], cap_emb[top])
    return {"pos": pos, "ranks": ranks, "top": top, "capsim": capsim,
            "is_pos": top == pos[:, None], "exact": cap_norm[top] == cap_norm[pos][:, None],
            "n_dup": n_dup, "n_dup_other": n_dup_other}


def summarize(r: dict, taus=TAUS, tau: float = TAU) -> dict:
    k = r["top"].shape[1]
    ks = [x for x in (1, 5, 10) if x <= k]
    out = {f"event {m}": v for m, v in rank_metrics(r["ranks"]).items()}
    exact = r["is_pos"] | r["exact"]
    for kk in ks:
        out[f"exact-caption@{kk}"] = 100 * float(exact[:, :kk].any(1).mean())
    for t in taus:
        hit = r["is_pos"] | (r["capsim"] >= t)
        for kk in ks:
            out[f"sem@{kk} (tau={t})"] = 100 * float(hit[:, :kk].any(1).mean())
    miss = 100 - out["event R@1"]
    gain = out[f"sem@1 (tau={tau})"] - out["event R@1"] if f"sem@1 (tau={tau})" in out else float("nan")
    out[f"misses that are same-meaning (tau={tau}) %"] = 100 * gain / miss if miss > 0 else float("nan")
    out[f"queries with >=1 near-duplicate in another video (tau={tau}) %"] = 100 * float((r["n_dup_other"] > 0).mean())
    out["queries"] = int(len(r["ranks"]))
    return out


def error_types(r: dict, index: EventIndex, tau: float = TAU) -> np.ndarray:
    t1 = r["top"][:, 0]
    same_video = index.video_id[t1] == index.video_id[r["pos"]]
    return np.where(r["is_pos"][:, 0], ERROR_TYPES[0],
                    np.where(same_video, ERROR_TYPES[1],
                             np.where(r["capsim"][:, 0] >= tau, ERROR_TYPES[2], ERROR_TYPES[3])))


def group_table(name: str, values, bins, r: dict, sem1: np.ndarray) -> list[dict]:
    values = np.asarray(values)
    rows = []
    for label, lo, hi in bins:
        sel = (values >= lo) & (values < hi)
        if not sel.any():
            continue
        m = rank_metrics(r["ranks"][sel])
        rows.append({name: label, "n": int(sel.sum()), "% queries": 100 * float(sel.mean()),
                     "R@1": m["R@1"], "R@5": m["R@5"], "R@10": m["R@10"], "MedR": m["MedR"],
                     "MnR": m["MnR"], "sem@1": 100 * float(sem1[sel].mean())})
    return rows


def hub_tables(r: dict, index: EventIndex, n_top: int = 15) -> tuple[list[dict], list[dict], dict]:
    """Events returned as top-1 most often, and over-/under-retrieval per event duration.

    ``slots/events`` > 1: events of that duration take more top-1 slots than their share of the
    database (they attract queries); < 1: they are retrieved less than chance.
    """
    N, n = len(index), len(r["ranks"])
    cnt1 = np.bincount(r["top"][:, 0], minlength=N)
    cntk = np.bincount(r["top"].ravel(), minlength=N)
    dur = index.te - index.ts
    hubs = [{"top1 count": int(cnt1[j]), f"top{r['top'].shape[1]} count": int(cntk[j]),
             "video_id": str(index.video_id[j]), "event": f"[{index.ts[j]:.1f}, {index.te[j]:.1f}]",
             "duration (s)": round(float(dur[j]), 1), "caption": str(index.caption[j])}
            for j in np.argsort(-cnt1, kind="stable")[:n_top] if cnt1[j] > 0]
    by_dur = []
    for label, lo, hi in DURATION_BINS:
        sel = (dur >= lo) & (dur < hi)
        if not sel.any():
            continue
        share_ev = float(sel.mean())
        share_slots = float(cnt1[sel].sum()) / n
        by_dur.append({"event duration": label, "events": int(sel.sum()), "% events": 100 * share_ev,
                       "% top-1 slots": 100 * share_slots, "slots/events": share_slots / share_ev})
    stats = {"events never in any top-k %": 100 * float((cntk == 0).mean()),
             "max top-1 count": int(cnt1.max()) if N else 0,
             "expected top-1 count per event": n / N if N else float("nan")}
    return hubs, by_dur, stats


def calibration_pairs(r: dict, index: EventIndex, per_bin: int = 5, seed: int = 0) -> list[dict]:
    """GT caption vs top-1 caption (other video) per judge-similarity bin, to eyeball ``tau``."""
    t1 = r["top"][:, 0]
    other = index.video_id[t1] != index.video_id[r["pos"]]
    rng = np.random.default_rng(seed)
    out = []
    for lo, hi in CALIBRATION_BINS:
        idx = np.where(other & (r["capsim"][:, 0] >= lo) & (r["capsim"][:, 0] < hi))[0]
        for i in rng.permutation(idx)[:per_bin]:
            out.append({"judge sim": round(float(r["capsim"][i, 0]), 3), "bin": f"[{lo:.1f}, {min(hi, 1):.1f})",
                        "GT caption": str(index.caption[r["pos"][i]]), "top-1 caption": str(index.caption[t1[i]])})
    return out


def analyze(index: EventIndex, q_emb, pos, query_ids, query_texts, cap_emb, cap_norm,
            taus=TAUS, tau: float = TAU, top_k: int = TOP_K) -> dict:
    """Everything for one query set (GT captions or custom queries)."""
    r = score_queries(index, q_emb, pos, cap_emb, cap_norm, tau, top_k)
    pos = r["pos"]
    types = error_types(r, index, tau)
    sem = r["is_pos"] | (r["capsim"] >= tau)
    sem1, sem5 = sem[:, 0], sem[:, :5].any(1)
    dur = (index.te - index.ts)[pos]
    words = np.array([len(str(t).split()) for t in query_texts])
    t1 = r["top"][:, 0]
    per_query = [{
        "query_id": str(query_ids[i]), "query": str(query_texts[i]), "video_id": str(index.video_id[pos[i]]),
        "gt": f"[{index.ts[pos[i]]:.1f}, {index.te[pos[i]]:.1f}]", "gt_caption": str(index.caption[pos[i]]),
        "duration (s)": round(float(dur[i]), 1), "words": int(words[i]),
        "near-duplicates (other videos)": int(r["n_dup_other"][i]), "near-duplicates (all)": int(r["n_dup"][i]),
        "rank": int(r["ranks"][i]), "error type": str(types[i]),
        "top1_video": str(index.video_id[t1[i]]), "top1": f"[{index.ts[t1[i]]:.1f}, {index.te[t1[i]]:.1f}]",
        "top1_caption": str(index.caption[t1[i]]), "top1 judge sim": round(float(r["capsim"][i, 0]), 3),
        "sem@1": bool(sem1[i]), "sem@5": bool(sem5[i]),
    } for i in range(len(pos))]
    n = len(pos)
    err = [{"top-1 is": t, "n": int((types == t).sum()), "%": 100 * float((types == t).mean()) if n else 0.0}
           for t in ERROR_TYPES]
    hubs, hub_by_dur, hub_stats = hub_tables(r, index)
    tail = sorted((q for q in per_query if q["rank"] > TAIL_RANK), key=lambda q: -q["rank"])
    return {
        "summary": {**summarize(r, taus, tau), **hub_stats},
        "error_types": err,
        "by_duration": group_table("GT event duration", dur, DURATION_BINS, r, sem1),
        "by_duplicates": group_table(f"near-duplicates in other videos (tau={tau})", r["n_dup_other"],
                                     DUPLICATE_BINS, r, sem1),
        "by_query_len": group_table("query length", words, QUERY_LEN_BINS, r, sem1),
        "hubs": hubs, "hub_by_duration": hub_by_dur,
        "tail": tail, "calibration": calibration_pairs(r, index),
        "per_query": per_query,
    }


# --------------------------------------------------------------------------- #
# Entry point                                                                 #
# --------------------------------------------------------------------------- #

def _custom_inputs(index: EventIndex, queries_path, text_store, min_tiou: float = 0.5):
    from .manifest import read_store
    from .queries import load_queries, resolve_gt

    queries = resolve_gt(load_queries(queries_path), index, min_tiou)
    vecs = read_store(text_store, "text")
    usable = [q for q in queries if q["pos"] >= 0 and q["key"] in vecs]
    if not usable:
        return None
    emb = np.stack([vecs[q["key"]] for q in usable]).astype(np.float32)
    return emb, [q["pos"] for q in usable], [q["qid"] for q in usable], [q["query"] for q in usable]


TABLES = ("error_types", "by_duration", "by_duplicates", "by_query_len", "hubs", "hub_by_duration",
          "tail", "calibration", "per_query")


def save_report(part: dict, out_dir, prefix: str) -> None:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    json.dump({k: part[k] for k in ("summary", "error_types", "by_duration", "by_duplicates",
                                    "by_query_len", "hub_by_duration")},
              open(out_dir / f"{prefix}_summary.json", "w", encoding="utf-8"),
              indent=1, ensure_ascii=False, default=float)
    for name in TABLES:
        if part[name]:
            save_csv(part[name], out_dir / f"{prefix}_{name}.csv")


def run_analysis(cache_dir, embed: Callable[[list[str]], np.ndarray], taus=TAUS, tau: float = TAU,
                 custom_queries=None, custom_text_store=None, out_dir=None, top_k: int = TOP_K) -> dict:
    """Analyse the GT queries (and the custom queries when given) of one ``omni_cache/<tag>``."""
    index = EventIndex.load(cache_dir)
    text = load_text(cache_dir)
    cap_emb = _unit(embed([str(c) for c in index.caption]))
    cap_norm = np.array([normalize_caption(c) for c in index.caption])
    pos = np.array([index.row_of.get(k, -1) for k in text["gt_seg_key"]])
    if (pos < 0).any():
        raise ValueError(f"{int((pos < 0).sum())} captions have no event in events.npz")

    report = {"judge": getattr(embed, "name", "custom"), "taus": list(taus), "tau": tau,
              "gt": analyze(index, text["emb"], pos, text["caption_id"], text["sentence"],
                            cap_emb, cap_norm, taus, tau, top_k),
              "custom": None}
    if custom_queries and custom_text_store:
        inp = _custom_inputs(index, custom_queries, custom_text_store)
        if inp is not None:
            report["custom"] = analyze(index, *inp, cap_emb, cap_norm, taus, tau, top_k)
    if out_dir is not None:
        for part in ("gt", "custom"):
            if report[part] is not None:
                save_report(report[part], out_dir, part)
        json.dump({"judge": report["judge"], "taus": report["taus"], "tau": tau},
                  open(Path(out_dir) / "settings.json", "w"), indent=1)
    return report
