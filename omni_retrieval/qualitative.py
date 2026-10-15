"""Qualitative analysis of the video-retrieval table (``UEMR_FINAL.md`` §8, ``docs/UEMR_VIEC_TIEP_THEO.md`` §A4).

Everything is read from one collected Omni cache (``segments.npz``, ``text.npz``, ``events.npz``) and one
partition folder, so the same call reruns on a new UniAV (``omni_cache/ft/<hash>/``, ``partitions_<hash>/``):

* ``per_query``      one row per val caption: GT length / position, near-duplicates, K and cover tIoU of R6 / R3
                     (cover = best tIoU of the caption's GT window with any segment: < 0.3 = UniAV missed it),
                     rank, argmax segment and moment tIoU under R0, R2, R3, R6, R7
* ``profile``        those features averaged over flip groups (R6 right and R0 wrong, R0 right and R6 wrong, ...)
* ``examples``       R6 beats R3 the most; R0 / R2 beat R6; near-duplicate queries (10-29) with the top videos
                     of R6 and the GT caption closest to each argmax segment
* ``plot_timeline``  GT / R6 / R3 segments of one video, the query's GT event and both argmax segments

    from omni_retrieval.qualitative import run
    out = run(db_dir, val_videos, parts_dir, subset_json, dup_csv, out_dir)
"""

from __future__ import annotations

import json
import textwrap
from pathlib import Path

import numpy as np
import pandas as pd

from .evaluate import load_index, load_text, video_ranks
from .partitions_uemr import read_durations
from .search import EventIndex, tiou_vec

PARTS = ("global", "uni_chunk", "uni_M", "uniav_pred", "gt")
SHORT = {"global": "R0", "uni_chunk": "R2", "uni_M": "R3", "uniav_pred": "R6", "gt": "R7"}
DUP_COLUMN = "near-duplicates (other videos)"
MISS_TIOU = 0.3


def _cover(segs, ts: float, te: float) -> float:
    s = np.asarray(segs, dtype=np.float64).reshape(-1, 2)
    return float(tiou_vec(s[:, 0], s[:, 1], ts, te).max()) if len(s) else 0.0


def per_query(db_dir, videos, parts_dir, subset_path, dup_csv=None, parts=PARTS) -> pd.DataFrame:
    """One row per caption of ``text.npz``, gallery ``videos`` (None = every video of the cache)."""
    text = load_text(db_dir)
    dur = read_durations(subset_path)
    df = pd.DataFrame({"caption_id": text["caption_id"].astype(str), "video_id": text["video_id"].astype(str),
                       "sentence": text["sentence"].astype(str), "gt_ts": text["gt_ts"].astype(float),
                       "gt_te": text["gt_te"].astype(float)})
    df["gt_len"] = df.gt_te - df.gt_ts
    df["video_dur"] = df.video_id.map(dur)
    df["rel_pos"] = ((df.gt_ts + df.gt_te) / 2 / df.video_dur).clip(0, 1)
    for p in parts:
        s = SHORT.get(p, p)
        r = video_ranks(load_index(db_dir, p, videos), text)
        df[f"rank {s}"] = r["ranks"]
        df[f"ts {s}"], df[f"te {s}"] = r["best_ts"].astype(float), r["best_te"].astype(float)
        df[f"moment tIoU {s}"] = tiou_vec(r["best_ts"], r["best_te"], text["gt_ts"], text["gt_te"])
    for p in ("uniav_pred", "uni_M"):
        seg = json.loads((Path(parts_dir) / f"{p}.json").read_text(encoding="utf-8"))
        s = SHORT[p]
        df[f"K {s}"] = df.video_id.map(lambda v: len(seg.get(v, [])))
        df[f"cover tIoU {s}"] = [_cover(seg.get(v, []), a, b) for v, a, b in zip(df.video_id, df.gt_ts, df.gt_te)]
    df["near_dup"] = np.nan
    if dup_csv:
        dup = pd.read_csv(dup_csv, usecols=["query_id", DUP_COLUMN])
        df["near_dup"] = df.caption_id.map(dict(zip(dup.query_id.astype(str), dup[DUP_COLUMN])))
    return df


def flip_groups(df: pd.DataFrame) -> dict[str, pd.Series]:
    """Queries whose R@1 flips between R6 and R0 / R2 / R3 (plus every query)."""
    hit = {s: df[f"rank {s}"] == 1 for s in SHORT.values() if f"rank {s}" in df}
    groups = {"tất cả": pd.Series(True, index=df.index)}
    for a in ("R0", "R2", "R3"):
        if a in hit and "R6" in hit:
            groups[f"R6 đúng, {a} sai"] = hit["R6"] & ~hit[a]
            groups[f"{a} đúng, R6 sai"] = hit[a] & ~hit["R6"]
    return groups


def profile(df: pd.DataFrame, groups: dict[str, pd.Series]) -> pd.DataFrame:
    rows = []
    for name, m in groups.items():
        d = df[m]
        rows.append({"nhóm": name, "queries": len(d), "GT len median (s)": d.gt_len.median(),
                     "% GT < 10 s": 100 * (d.gt_len < 10).mean(), "vị trí GT / thời lượng (mean)": d.rel_pos.mean(),
                     "% near-dup >= 10": 100 * (d.near_dup >= 10).mean(), "near-dup median": d.near_dup.median(),
                     f"% UniAV bỏ sót (cover R6 < {MISS_TIOU})": 100 * (d["cover tIoU R6"] < MISS_TIOU).mean(),
                     "cover tIoU R6 mean": d["cover tIoU R6"].mean(), "K R6 mean": d["K R6"].mean(),
                     "moment tIoU R6 mean": d["moment tIoU R6"].mean(), "video dur median (s)": d.video_dur.median()})
    return pd.DataFrame(rows)


SHOW = ["caption_id", "video_id", "sentence", "gt_ts", "gt_te", "gt_len", "near_dup", "cover tIoU R6", "K R6",
        "rank R0", "rank R2", "rank R3", "rank R6", "rank R7", "moment tIoU R6", "moment tIoU R3"]


def r6_beats(df: pd.DataFrame, other: str = "R3", n: int = 10) -> pd.DataFrame:
    """R6 rank 1, sorted by how far ``other`` ranked the same video."""
    d = df[df["rank R6"] == 1].assign(gain=lambda x: x[f"rank {other}"] - x["rank R6"])
    return d.nlargest(n, "gain")[[c for c in SHOW if c in d] + ["gain"]]


def beats_r6(df: pd.DataFrame, others=("R0", "R2"), n: int = 10) -> pd.DataFrame:
    """``other`` rank 1 while R6 misses, worst R6 rank first."""
    m = (df["rank R6"] > 1) & np.any([df[f"rank {o}"] == 1 for o in others], axis=0)
    return df[m].nlargest(n, "rank R6")[[c for c in SHOW if c in df]]


def near_dup_examples(db_dir, df: pd.DataFrame, videos, n: int = 5, lo: int = 10, hi: int = 30, top: int = 3,
                      r6_wrong: bool = True, seed: int = 0) -> pd.DataFrame:
    """Queries with ``lo <= near-duplicates < hi``: the ``top`` videos of R6, their argmax segment and the GT
    caption of that video closest (tIoU) to it -- the same step in another video shows up there."""
    sel = df[(df.near_dup >= lo) & (df.near_dup < hi)]
    if r6_wrong:
        sel = sel[sel["rank R6"] > 1]
    pick = sel.sample(min(n, len(sel)), random_state=seed) if len(sel) else sel
    index = load_index(db_dir, "uniav_pred", videos)
    ev = EventIndex.load(db_dir)
    text = load_text(db_dir)
    row_of = {c: i for i, c in enumerate(text["caption_id"].astype(str))}
    rows = []
    for _, r in pick.iterrows():
        for k, h in enumerate(index.search(text["emb"][row_of[r.caption_id]], top_k=top), 1):
            a, b = h["best_segment"]
            m = np.flatnonzero(ev.video_id == h["video_id"])
            t = tiou_vec(ev.ts[m], ev.te[m], a, b) if len(m) else np.zeros(0)
            j = m[int(np.argmax(t))] if len(m) else None
            rows.append({"caption_id": r.caption_id, "query": r.sentence, "near_dup": r.near_dup,
                         "rank R6": r["rank R6"], "top": k, "video": h["video_id"],
                         "đúng video": h["video_id"] == r.video_id, "score": round(h["score"], 4),
                         "argmax": f"[{a:.1f}, {b:.1f}]",
                         "caption GT gần argmax nhất": str(ev.caption[j]) if j is not None else "",
                         "tIoU với GT đó": round(float(t.max()), 3) if len(t) else None})
    return pd.DataFrame(rows)


def plot_timeline(ax, r, segs: dict[str, dict], duration: float):
    """Rows GT / R6 / R3 of video ``r.video_id``; the query's GT event in green, each argmax segment outlined."""
    v = r.video_id
    lanes = [("GT", "gt", None), (f"R6 (rank {r['rank R6']})", "uniav_pred", "R6"),
             (f"R3 (rank {r['rank R3']})", "uni_M", "R3")]
    for y, (label, part, s) in enumerate(lanes):
        for a, b in segs[part].get(v, []):
            ax.barh(y, b - a, left=a, height=0.6, color="C0" if s else "0.75", edgecolor="white", alpha=0.6)
        if s:
            a, b = r[f"ts {s}"], r[f"te {s}"]
            ax.barh(y, b - a, left=a, height=0.6, fill=False, edgecolor="C3", lw=2)
    ax.barh(0, r.gt_te - r.gt_ts, left=r.gt_ts, height=0.6, color="C2")
    ax.axvspan(r.gt_ts, r.gt_te, color="C2", alpha=0.08)
    ax.set_yticks(range(len(lanes)), [lane[0] for lane in lanes])
    ax.invert_yaxis()
    ax.set_xlim(0, duration)
    ax.set_xlabel("giây")
    ax.set_title("\n".join(textwrap.wrap(f"{v}: \"{r.sentence}\"", 90)), fontsize=9)


def plot_timelines(df: pd.DataFrame, picks: pd.DataFrame, parts_dir, out_dir, prefix: str) -> list[str]:
    import matplotlib.pyplot as plt

    segs = {p: json.loads((Path(parts_dir) / f"{p}.json").read_text(encoding="utf-8")) for p in ("gt", "uniav_pred", "uni_M")}
    paths = []
    for _, p in picks.iterrows():
        r = df[df.caption_id == p.caption_id].iloc[0]
        fig, ax = plt.subplots(figsize=(12, 2.2))
        plot_timeline(ax, r, segs, float(r.video_dur))
        path = Path(out_dir) / f"{prefix}_{r.caption_id}.png"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        paths.append(str(path))
    return paths


def run(db_dir, videos, parts_dir, subset_path, dup_csv, out_dir, n_examples: int = 10, n_timelines: int = 5) -> dict:
    """Every table and figure of A4 into ``out_dir``; returns the tables (DataFrames) and the figure paths."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    df = per_query(db_dir, videos, parts_dir, subset_path, dup_csv)
    tables = {"per_query": df, "profile": profile(df, flip_groups(df)),
              "r6_beats_r3": r6_beats(df, "R3", n_examples), "r6_beats_r0": r6_beats(df, "R0", n_examples),
              "r0_r2_beat_r6": beats_r6(df, ("R0", "R2"), n_examples),
              "near_dup_10_29": near_dup_examples(db_dir, df, videos)}
    for name, t in tables.items():
        t.to_csv(out / f"qual_{name}.csv", index=False)
    # timelines: R6 beats R3 (one per video) then R0 / R2 beat R6
    first = tables["r6_beats_r3"].drop_duplicates("video_id").head(max(1, n_timelines - 2))
    second = tables["r0_r2_beat_r6"].drop_duplicates("video_id").head(n_timelines - len(first))
    figs = (plot_timelines(df, first, parts_dir, out, "timeline_r6_beats_r3")
            + plot_timelines(df, second, parts_dir, out, "timeline_r0r2_beat_r6"))
    return {**tables, "figures": figs}
