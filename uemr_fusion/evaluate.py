"""Scoring of fused vectors with the P4 code (``omni_retrieval.search.SegmentIndex``, ``evaluate``), agg ``max``.

* ``fuse``          c_i of every video of a split (eval mode, deterministic)
* ``attention_maps`` step-1 cross-attention weights of a few videos (UEMR_FINAL §8, qualitative)
* ``build_index``   a ``SegmentIndex`` of fused (or raw) vectors; ``use_global`` adds c_g = e_g, span [0, duration]
* ``split_metrics`` text -> video R@k over one split's own videos (the dev check of every epoch)
* ``final_eval``    val once: galleries 394 (val) and 498 (val + dev), column (b) from the same cache, R@1 by GT
                    length and by near-duplicate count, paired bootstrap (c) - (b) by video and by query
* ``seed_mean_bootstrap``  mean R@1 of several seed runs - column (b), bootstrap by video, from their ``ranks.npz``
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from omni_retrieval.evaluate import evaluate_partition, paired_bootstrap, rank_metrics, video_ranks
from omni_retrieval.search import SegmentIndex

from .data import FusionSplit, derangement

DURATION_BINS = (("<10s", 0, 10), ("10-40s", 10, 40), (">=40s", 40, np.inf))
DUPLICATE_BINS = (("0", 0, 1), ("1-2", 1, 3), ("3-9", 3, 10), ("10-29", 10, 30), (">=30", 30, np.inf))
DUP_COLUMN = "near-duplicates (other videos)"


def autocast(device):
    return torch.autocast("cuda", dtype=torch.bfloat16, enabled=torch.device(device).type == "cuda")


@torch.no_grad()
def fuse(model, ds: FusionSplit, shuffle_z: bool = False, seed: int = 0, chunk: int = 64) -> list[np.ndarray]:
    """``[K_v, d]`` float32 per video, in ``ds.video_ids`` order. ``shuffle_z``: Z of another video of the same
    chunk (fixed seed), as during training."""
    was = model.training
    model.eval()
    gen = torch.Generator().manual_seed(seed)
    out = []
    for s in range(0, ds.n_videos, chunk):
        vidx = torch.arange(s, min(s + chunk, ds.n_videos), device=ds.device)
        perm = derangement(len(vidx), gen).to(ds.device) if shuffle_z and len(vidx) > 1 else None
        b = ds.batch(vidx, perm)
        with autocast(ds.device):
            C = model(b["E"], b["kmask"], b["segs"], b["Z"], b["t_sec"], b["duration"])
        k = b["kmask"].sum(1).tolist()
        out += [C[i, :k[i]].float().cpu().numpy() for i in range(len(vidx))]
    model.train(was)
    return out


@torch.no_grad()
def attention_maps(model, ds: FusionSplit, videos) -> list[dict]:
    """Step-1 cross-attention of the video indices ``videos`` (float32, mean over heads): per video its
    ``segs`` [K, 2], the Z time axis ``t_sec`` [T] and ``attn`` [K, T] (each row sums to 1 over its window)."""
    was = model.training
    model.eval()
    model.keep_attn = True
    try:
        vidx = torch.as_tensor(list(videos), device=ds.device)
        b = ds.batch(vidx)
        model(b["E"], b["kmask"], b["segs"], b["Z"], b["t_sec"], b["duration"])
        a = model.last_attn.float().mean(1).cpu().numpy()
    finally:
        model.keep_attn, model.last_attn = False, None
        model.train(was)
    k = b["kmask"].sum(1).tolist()
    return [{"video_id": ds.video_ids[v], "segs": b["segs"][i, :k[i]].cpu().numpy(),
             "t_sec": b["t_sec"][i].cpu().numpy(), "attn": a[i, :k[i]]} for i, v in enumerate(vidx.tolist())]


def raw_vectors(ds: FusionSplit) -> list[np.ndarray]:
    """The cached Omni vectors as they are (column b), the input of the fusion."""
    k = ds.kmask.sum(1).tolist()
    return [ds.E[i, :k[i]].float().cpu().numpy() for i in range(ds.n_videos)]


def build_index(ds: FusionSplit, vecs: list[np.ndarray], use_global: bool = False, name: str = "") -> SegmentIndex:
    k = ds.kmask.sum(1).tolist()
    segs = ds.segs.cpu().numpy()
    emb = [np.concatenate(vecs)]
    vid = [np.repeat(np.array(ds.video_ids), k)]
    ts = [np.concatenate([segs[i, :k[i], 0] for i in range(ds.n_videos)])]
    te = [np.concatenate([segs[i, :k[i], 1] for i in range(ds.n_videos)])]
    if use_global:
        ok = ds.eg_ok.cpu().numpy()
        g = ds.eg.float().cpu().numpy()[ok]
        emb.append(g / np.linalg.norm(g, axis=1, keepdims=True))
        vid.append(np.array(ds.video_ids)[ok])
        ts.append(np.zeros(int(ok.sum()), np.float32))
        te.append(ds.duration.cpu().numpy()[ok])
    return SegmentIndex(np.concatenate(emb), np.concatenate(vid), np.concatenate(ts), np.concatenate(te), name)


def merge(*indexes: SegmentIndex, name: str = "") -> SegmentIndex:
    return SegmentIndex(np.concatenate([i.emb for i in indexes]), np.concatenate([i.seg_video for i in indexes]),
                        np.concatenate([i.ts for i in indexes]), np.concatenate([i.te for i in indexes]), name)


def split_metrics(model, ds: FusionSplit, use_global: bool = False, shuffle_z: bool = False) -> dict:
    """R@1/5/10, MedR, MnR of the split's captions against the split's videos (dev: 773 x 104)."""
    vecs = fuse(model, ds, shuffle_z) if model is not None else raw_vectors(ds)
    r = video_ranks(build_index(ds, vecs, use_global), ds.text())
    return rank_metrics(r["ranks"])


def group_rows(name: str, values, bins, ranks_b, ranks_c) -> list[dict]:
    values = np.asarray(values)
    rows = []
    for label, lo, hi in bins:
        sel = (values >= lo) & (values < hi)
        if sel.any():
            b, c = 100 * float((ranks_b[sel] <= 1).mean()), 100 * float((ranks_c[sel] <= 1).mean())
            rows.append({name: label, "n": int(sel.sum()), "R@1 (b)": b, "R@1 (c)": c, "(c) - (b)": c - b})
    return rows


def duplicate_counts(text: dict, dup_csv) -> np.ndarray | None:
    """Near-duplicate count of every query, from ``omni_cache/<tag>/analysis/gt_per_query.csv`` (judge tau 0.8)."""
    if not dup_csv:
        return None
    import pandas as pd
    dup = pd.read_csv(dup_csv, usecols=["query_id", DUP_COLUMN])
    m = dict(zip(dup["query_id"].astype(str), dup[DUP_COLUMN]))
    return np.array([m.get(str(c), np.nan) for c in text["caption_id"]], dtype=np.float64)


def seed_mean_bootstrap(ranks_paths, n_boot: int = 1000, seed: int = 0,
                        galleries=("val394", "all498")) -> list[dict]:
    """Column (c) of runs that differ only by seed, against column (b), bootstrap by video (``UEMR_BUOC_TIEP_P6.md`` §3.A).

    ``ranks_paths``: the ``ranks.npz`` of each run (``final_eval`` arrays; same queries in the same order and the
    same column (b), both checked). Per gallery: one row per run and one ``mean`` row, the mean R@1 over the runs
    minus R@1 (b), on one resampling of the source videos shared by every row (seed ``seed``).
    """
    runs = [np.load(p, allow_pickle=True) for p in ranks_paths]
    names = [Path(p).parent.name for p in ranks_paths]
    ref = runs[0]
    for name, r in zip(names[1:], runs[1:]):
        if not np.array_equal(r["caption_id"], ref["caption_id"]):
            raise ValueError(f"{name}: other queries or query order than {names[0]}")
    rows = []
    for gal in galleries:
        b = ref[f"ranks_b_{gal}"]
        for name, r in zip(names[1:], runs[1:]):
            if not np.array_equal(r[f"ranks_b_{gal}"], b):
                raise ValueError(f"{name}: column (b) of {gal} differs from {names[0]} (other cache?)")
        c = np.stack([r[f"ranks_c_{gal}"] for r in runs])
        for name, a in [*zip(names, c), (f"mean of {len(runs)}", c)]:
            bs = paired_bootstrap(a, b, 1, n_boot, seed, "video", ref["video_id"])
            rows.append({"gallery": gal, "run": name, "R@1 (c)": 100 * float((a <= 1).mean()),
                         "R@1 (b)": 100 * float((b <= 1).mean()), "(c) - (b)": bs["diff"], "ci_low": bs["ci_low"],
                         "ci_high": bs["ci_high"], "p(diff<=0)": bs["p(diff<=0)"], "videos": bs["n_units"]})
    return rows


def final_eval(model, val: FusionSplit, dev: FusionSplit, use_global: bool = False, shuffle_z: bool = False,
               dup_csv=None, n_boot: int = 1000) -> tuple[dict, dict]:
    """Report and per-query arrays of one trained fusion on val (scored once)."""
    text = val.text()
    vec_c = {"val": fuse(model, val, shuffle_z), "dev": fuse(model, dev, shuffle_z)}
    vec_b = {"val": raw_vectors(val), "dev": raw_vectors(dev)}
    idx = {}
    for col, vecs, glob in (("c", vec_c, use_global), ("b", vec_b, False)):
        v = build_index(val, vecs["val"], glob, f"({col})")
        idx[col, "val394"] = v
        idx[col, "all498"] = merge(v, build_index(dev, vecs["dev"], glob), name=f"({col})")
    report, arrays = {"alpha": float(model.alpha.detach().cpu())}, {"caption_id": text["caption_id"],
                                                                   "video_id": text["video_id"]}
    for gal in ("val394", "all498"):
        for col in ("b", "c"):
            report[f"{gal} ({col})"] = evaluate_partition(idx[col, gal], text, "max")
            r = video_ranks(idx[col, gal], text)
            arrays[f"ranks_{col}_{gal}"] = r["ranks"]
            arrays[f"best_ts_{col}_{gal}"], arrays[f"best_te_{col}_{gal}"] = r["best_ts"], r["best_te"]
        rc, rb = arrays[f"ranks_c_{gal}"], arrays[f"ranks_b_{gal}"]
        report[f"{gal} bootstrap (c)-(b) by video"] = paired_bootstrap(rc, rb, 1, n_boot, 0, "video", text["video_id"])
        report[f"{gal} bootstrap (c)-(b) by query"] = paired_bootstrap(rc, rb, 1, n_boot, 0)
    rc, rb = arrays["ranks_c_val394"], arrays["ranks_b_val394"]
    report["by_duration"] = group_rows("GT duration", text["gt_te"] - text["gt_ts"], DURATION_BINS, rb, rc)
    dup = duplicate_counts(text, dup_csv)
    if dup is not None:
        report["by_duplicates"] = group_rows("near-duplicates", np.nan_to_num(dup, nan=-1), DUPLICATE_BINS, rb, rc)
        report["queries without a duplicate count"] = int(np.isnan(dup).sum())
    return report, arrays
