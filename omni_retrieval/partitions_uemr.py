"""UEMR P2: the partitions built from UniAV's output (UEMR_FINAL.md §5; docs/UEMR_KE_HOACH_P2_P4.md §2).

Writes one ``<name>.json`` per partition, ``{video_id: [[ts, te], ...]}`` in seconds (2 decimals, sorted by
time), the format ``Config.extra_partitions`` reads:

* ``uniav_pred`` (R6) -- the ``(ts, te)`` of ``segments_pred.jsonl`` as they are
* ``uni_M``      (R3) -- ``K_pred`` equal segments (``partitions.uniform``)
* ``rand_M``     (R4) -- ``K_pred`` contiguous segments covering the video, each >= d_min, Dirichlet lengths
* ``kmedoids_M`` (R5) -- ``K_pred`` K-Medoids of the level-0 Z rows (cosine), clip ``[t - 1, t + 1]`` per medoid
* ``uni_chunk``  (R2) -- contiguous chunks of L = 14 s (median GT event length on trainsplit)

``K_pred`` of a video = its number of lines in ``segments_pred.jsonl``. Durations come only from
``subset_videos.json`` (mp4 header), the same source the Omni encoder clamps with. ``SOURCE.json`` records the
sha256 of ``segments_pred.jsonl`` and of every file written; a rerun refuses a different ``segments_pred.jsonl``
(R3-R6 must share one) unless ``--force``.

    python -m omni_retrieval.partitions_uemr --out <omni_cache>/common/partitions \\
        --segments <uniav_cache>/ft/segments_pred.jsonl --z-dir <uniav_cache>/ft/Z \\
        --subset <omni_cache>/common/subset_videos.json --seed 0

Controls of plan P6 §A3 (``--controls``), built from the partition files already in ``--out``:

* ``uniav_shift``           -- every R6 segment keeps its length, start drawn ``U[0, duration - length]``
* ``gt_shift``              -- the same for the R7 (``gt``) segments
* ``uniav_pred+uni_chunk``  -- union of R6 and R2 (duplicates dropped by the 0.01 s segment key)

    python -m omni_retrieval.partitions_uemr --controls --out <omni_cache>/common/partitions \\
        --subset <omni_cache>/common/subset_videos.json --seed 0
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

from .partitions import MIN_SEG_SEC, uniform

D_MIN = 2.0
CHUNK_SEC = 14.0
HALF_WINDOW = 1.0
MAX_ITER = 100
NAMES = ("uniav_pred", "uni_M", "rand_M", "kmedoids_M", "uni_chunk")
K_PRED_NAMES = ("uni_M", "rand_M", "kmedoids_M")       # must have exactly K_pred segments
CONTIGUOUS = ("uni_chunk", "uni_M", "rand_M")          # must tile [0, duration]
MIN_LEN = ("rand_M", "kmedoids_M")                     # every segment >= d_min
TOL = 0.01 + 1e-6                                      # 2-decimal rounding, plus float slack


def video_rng(seed: int, video_id: str) -> np.random.Generator:
    """Per-video generator, stable across sessions (Python's ``hash()`` is salted per process)."""
    return np.random.default_rng([seed, int(hashlib.md5(video_id.encode()).hexdigest()[:8], 16)])


def sha256(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _round(segs, duration: float) -> list[list[float]]:
    """2 decimals, never past the end of the video (rounding up could cross ``duration``)."""
    return sorted([round(float(ts), 2), min(round(float(te), 2), float(duration))] for ts, te in segs)


# --------------------------------------------------------------------------- #
# Partitions of one video                                                     #
# --------------------------------------------------------------------------- #

def uni_chunk(duration: float, length: float = CHUNK_SEC, d_min: float = D_MIN) -> list[list[float]]:
    """``[i*L, min((i+1)*L, duration)]``; a last chunk no longer than d_min is merged into the one before."""
    n = max(1, math.ceil(duration / length))
    segs = [[i * length, min((i + 1) * length, duration)] for i in range(n)]
    if len(segs) > 1 and segs[-1][1] - segs[-1][0] <= d_min:
        segs.pop()
        segs[-1][1] = duration
    return segs


def rand_m(duration: float, m: int, rng: np.random.Generator, d_min: float = D_MIN) -> list[list[float]]:
    """M contiguous segments tiling ``[0, duration]``: length_i = d_min + w_i * (duration - M d_min), w ~ Dir(1)."""
    free = duration - m * d_min
    if free < 0:
        print(f"[rand_M] WARNING: duration {duration} < {m} x d_min -> uniform")
        return uniform(duration, m)
    edges = np.concatenate([[0.0], np.cumsum(d_min + rng.dirichlet(np.ones(m)) * free)])
    edges[-1] = duration
    return [[float(a), float(b)] for a, b in zip(edges[:-1], edges[1:])]


def kmedoids(x: np.ndarray, m: int, rng: np.random.Generator, max_iter: int = MAX_ITER) -> tuple[np.ndarray, int, bool]:
    """Alternating K-Medoids on cosine distance. ``x``: (n, d). Returns ``(medoid rows, iterations, converged)``.

    k-medoids++ start (next medoid drawn with probability ~ squared distance to the nearest one), then
    assign every point to its nearest medoid and move each medoid to the point of its cluster with the
    smallest summed distance, until the medoids stop changing.
    """
    x = np.asarray(x, dtype=np.float32)
    x = x / np.clip(np.linalg.norm(x, axis=1, keepdims=True), 1e-12, None)
    dist = np.clip(1.0 - x @ x.T, 0.0, None)
    n = len(x)
    m = min(int(m), n)
    med = [int(rng.integers(n))]
    near = dist[med[0]].copy()
    while len(med) < m:
        p = near.astype(np.float64) ** 2    # float64: rng.choice checks that p sums to 1
        if p.sum() > 0:
            nxt = int(rng.choice(n, p=p / p.sum()))
        else:   # fewer distinct rows than m (a short video upsampled to 256 steps)
            nxt = int(rng.choice(np.setdiff1d(np.arange(n), med)))
        med.append(nxt)
        near = np.minimum(near, dist[nxt])
    med = np.array(med)
    for it in range(1, max_iter + 1):
        labels = dist[:, med].argmin(1)
        labels[med] = np.arange(m)          # a medoid always stays in its own cluster (ties at distance 0)
        new = med.copy()
        for k in range(m):
            idx = np.flatnonzero(labels == k)
            new[k] = idx[dist[np.ix_(idx, idx)].sum(1).argmin()]
        if np.array_equal(new, med):
            return med, it, True
        med = new
    return med, max_iter, False


def window(t: float, duration: float, half: float = HALF_WINDOW) -> list[float]:
    """``[t - half, t + half]`` inside ``[0, duration]``, shifted inwards at the borders (as ``extend_short``).

    ``t`` may lie past ``duration``: Z's ``t_sec`` is computed from the feature rows, not clipped.
    """
    if 2 * half >= duration:
        return [0.0, float(duration)]
    ts = min(max(float(t) - half, 0.0), float(duration) - 2 * half)
    return [ts, ts + 2 * half]


def kmedoids_m(z: np.ndarray, t_sec: np.ndarray, duration: float, m: int, rng: np.random.Generator,
               half: float = HALF_WINDOW) -> tuple[list[list[float]], int, bool]:
    med, it, ok = kmedoids(z, m, rng)
    return sorted(window(t_sec[j], duration, half) for j in med), it, ok


# --------------------------------------------------------------------------- #
# All videos                                                                  #
# --------------------------------------------------------------------------- #

def read_segments(path) -> dict[str, list[list[float]]]:
    out = defaultdict(list)
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                r = json.loads(line)
                out[r["video_id"]].append([float(r["ts"]), float(r["te"])])
    return {v: sorted(s) for v, s in out.items()}


def read_durations(subset_path) -> dict[str, float]:
    blob = json.loads(Path(subset_path).read_text(encoding="utf-8"))
    return {r["video_id"]: float(r["duration"]) for r in blob["videos"]}


def build(segments: dict, durations: dict, z_dir, seed: int = 0, chunk_sec: float = CHUNK_SEC,
          d_min: float = D_MIN) -> tuple[dict[str, dict], dict]:
    """``({name: {video_id: [[ts, te], ...]}}, kmedoids stats)`` for every video of ``durations``."""
    missing = sorted(set(durations) - set(segments))
    if missing:
        raise ValueError(f"{len(missing)} gallery videos have no line in segments_pred.jsonl: {missing[:5]}")
    parts = {name: {} for name in NAMES}
    iters, not_converged = [], []
    for vid in sorted(durations):
        dur, k = durations[vid], len(segments[vid])
        parts["uniav_pred"][vid] = _round(segments[vid], dur)
        parts["uni_M"][vid] = _round(uniform(dur, k), dur)
        parts["rand_M"][vid] = _round(rand_m(dur, k, video_rng(seed, vid), d_min), dur)
        with np.load(Path(z_dir) / f"{vid}.npz") as zf:
            segs, it, ok = kmedoids_m(zf["z"], zf["t_sec"], dur, k, video_rng(seed, vid))
        parts["kmedoids_M"][vid] = _round(segs, dur)
        parts["uni_chunk"][vid] = _round(uni_chunk(dur, chunk_sec, d_min), dur)
        iters.append(it)
        if not ok:
            not_converged.append(vid)
    stats = {"videos": len(iters), "mean_iterations": float(np.mean(iters)) if iters else 0.0,
             "max_iterations": int(max(iters)) if iters else 0, "max_iter": MAX_ITER,
             "not_converged": len(not_converged), "not_converged_videos": not_converged,
             "half_window": HALF_WINDOW, "distance": "1 - cos on L2-normalised fp32 Z rows (level 0)"}
    return parts, stats


def check(name: str, part: dict, durations: dict, k_pred: dict, d_min: float = D_MIN) -> dict:
    """The checks of plan §2.6: assert, and return the counts to print."""
    assert set(part) == set(durations), f"{name}: {len(part)} videos, expected {len(durations)}"
    for vid, segs in part.items():
        dur = durations[vid]
        assert segs, f"{name}/{vid}: no segment"
        assert all(0 <= ts < te <= dur for ts, te in segs), f"{name}/{vid}: segment outside [0, {dur}]: {segs}"
        assert all(a[0] <= b[0] for a, b in zip(segs, segs[1:])), f"{name}/{vid}: not sorted"
        if name in K_PRED_NAMES:
            assert len(segs) == k_pred[vid], f"{name}/{vid}: {len(segs)} segments, K_pred {k_pred[vid]}"
        if name in CONTIGUOUS:
            assert abs(segs[0][0]) <= TOL and abs(segs[-1][1] - dur) <= TOL, f"{name}/{vid}: does not cover [0, {dur}]"
            assert all(abs(a[1] - b[0]) <= TOL for a, b in zip(segs, segs[1:])), f"{name}/{vid}: gap or overlap"
        if name in MIN_LEN:
            assert all(te - ts >= d_min - TOL for ts, te in segs), f"{name}/{vid}: segment < d_min"
    counts = [len(s) for s in part.values()]
    lengths = [te - ts for s in part.values() for ts, te in s]
    # what omni_retrieval.partitions.clamp would drop at encode time (should be 0)
    dropped = sum(ts > durations[v] - MIN_SEG_SEC for v, s in part.items() for ts, _ in s)
    return {"partition": name, "videos": len(part), "segments": int(sum(counts)),
            "mean seg/video": round(float(np.mean(counts)), 2), "median seg/video": float(np.median(counts)),
            "median len (s)": round(float(np.median(lengths)), 2), "min len (s)": round(float(min(lengths)), 2),
            "dropped by clamp": int(dropped)}


# --------------------------------------------------------------------------- #
# Controls (plan P6 §A3)                                                      #
# --------------------------------------------------------------------------- #

SHIFTS = {"uniav_shift": "uniav_pred", "gt_shift": "gt"}
UNION = "uniav_pred+uni_chunk"
CONTROLS = (*SHIFTS, UNION)


def shift(segs, duration: float, rng: np.random.Generator) -> list[list[float]]:
    """Same lengths, starts ``U[0, duration - length]`` drawn independently, sorted by time.

    The upper bound is also kept ``<= duration - MIN_SEG_SEC`` so ``partitions.clamp`` never drops a short
    segment at encode time. Times are not rounded: rounding both ends could change a length by 0.01 s.
    """
    out = []
    for ts, te in segs:
        length = float(te) - float(ts)
        hi = max(0.0, min(float(duration) - length, float(duration) - MIN_SEG_SEC))
        start = float(rng.uniform(0.0, hi)) if hi > 0 else 0.0
        out.append([start, start + length])
    return sorted(out)


def union(*parts_of_video) -> list[list[float]]:
    """Segments of several partitions of one video, one per 0.01 s key (the encoder's dedup rule)."""
    seen = {}
    for segs in parts_of_video:
        for ts, te in segs:
            seen.setdefault((round(float(ts), 2), round(float(te), 2)), [float(ts), float(te)])
    return sorted(seen.values())


def coverage(segs, duration: float) -> float:
    """Length of the union of the segments / duration."""
    total, end = 0.0, -np.inf
    for ts, te in sorted(segs):
        ts, te = max(float(ts), end), float(te)
        if te > ts:
            total += te - ts
        end = max(end, te)
    return total / float(duration) if duration else 0.0


def check_shift(name: str, shifted: dict, source: dict, durations: dict) -> dict:
    """Assert the shifted partition has the source's K and multiset of lengths, inside every video."""
    for vid, segs in source.items():
        new = shifted[vid]
        assert len(new) == len(segs), f"{name}/{vid}: {len(new)} segments, source {len(segs)}"
        a = np.sort([te - ts for ts, te in segs])
        b = np.sort([te - ts for ts, te in new])
        assert np.allclose(a, b, atol=1e-6), f"{name}/{vid}: lengths differ"
        assert all(0 <= ts < te <= durations[vid] + 1e-6 for ts, te in new), f"{name}/{vid}: outside the video"
    cov_src = [coverage(s, durations[v]) for v, s in source.items()]
    cov_new = [coverage(s, durations[v]) for v, s in shifted.items()]
    return {"partition": name, "source": SHIFTS[name], "videos": len(shifted),
            "segments": sum(len(s) for s in shifted.values()),
            "coverage source": round(100 * float(np.mean(cov_src)), 2),
            "coverage shifted": round(100 * float(np.mean(cov_new)), 2)}


def build_controls(parts_dir, durations: dict, seed: int = 0) -> tuple[dict[str, dict], list[dict]]:
    """``uniav_shift``, ``gt_shift`` and the R6 + R2 union for every video of ``durations``.

    Reads ``uniav_pred.json``, ``uni_chunk.json`` and ``gt.json`` from ``parts_dir`` (``gt.json`` is written by
    ``manifest.prepare``). Videos missing from a source file are an error.
    """
    d = Path(parts_dir)
    src = {n: json.loads((d / f"{n}.json").read_text(encoding="utf-8")) for n in ("uniav_pred", "gt", "uni_chunk")}
    for n, part in src.items():
        missing = sorted(set(durations) - set(part))
        if missing:
            raise ValueError(f"{n}.json has no segments for {len(missing)} videos: {missing[:5]}")
    parts, rows = {}, []
    for name, base in SHIFTS.items():
        parts[name] = {v: shift(src[base][v], durations[v], video_rng(seed, f"{v}#{name}")) for v in sorted(durations)}
        rows.append(check_shift(name, parts[name], {v: src[base][v] for v in durations}, durations))
    parts[UNION] = {v: union(src["uniav_pred"][v], src["uni_chunk"][v]) for v in sorted(durations)}
    n_union = [len(s) for s in parts[UNION].values()]
    rows.append({"partition": UNION, "source": "uniav_pred + uni_chunk", "videos": len(n_union),
                 "segments": int(sum(n_union)), "coverage source": None,
                 "coverage shifted": round(100 * float(np.mean([coverage(s, durations[v])
                                                                for v, s in parts[UNION].items()])), 2)})
    return parts, rows


def run_controls(out_dir, subset_path, seed: int = 0) -> list[dict]:
    """Write the control partitions next to the others and record them in ``SOURCE.json`` (key ``controls``)."""
    out = Path(out_dir)
    durations = read_durations(subset_path)
    parts, rows = build_controls(out, durations, seed)
    files = {}
    for name, part in parts.items():
        path = out / f"{name}.json"
        path.write_text(json.dumps(part), encoding="utf-8")
        files[name] = {"file": path.name, "sha256": sha256(path)}
    src = out / "SOURCE.json"
    blob = json.loads(src.read_text(encoding="utf-8")) if src.is_file() else {}
    blob["controls"] = {"seed": seed, "commit": _git_commit(), "created": time.strftime("%Y-%m-%d %H:%M:%S"),
                        "sources": {n: sha256(out / f"{n}.json") for n in ("uniav_pred", "gt", "uni_chunk")},
                        "partitions": files, "summary": rows}
    src.write_text(json.dumps(blob, indent=1), encoding="utf-8")
    print(f"[partitions_uemr] controls {list(parts)} -> {out}")
    return rows


def _git_commit() -> str | None:
    try:
        return subprocess.run(["git", "-C", str(Path(__file__).resolve().parent), "rev-parse", "HEAD"],
                              capture_output=True, text=True, check=True).stdout.strip()
    except Exception:  # noqa: BLE001 - no git on the machine
        return None


def _segments_params(segments_path) -> dict:
    """theta / k_max / d_min of the UniAV run, from ``predictions.jsonl`` and ``theta.json`` next to it."""
    d = Path(segments_path).parent
    out = {}
    pred = d / "predictions.jsonl"
    if pred.is_file():
        with open(pred, encoding="utf-8") as f:
            first = f.readline()
        if first.strip():
            out.update(json.loads(first).get("params", {}))
    if (d / "theta.json").is_file():
        out["theta_json"] = json.loads((d / "theta.json").read_text(encoding="utf-8"))
    return out


def run(out_dir, segments_path, z_dir, subset_path, seed: int = 0, chunk_sec: float = CHUNK_SEC,
        d_min: float = D_MIN, force: bool = False) -> list[dict]:
    out = Path(out_dir)
    src = out / "SOURCE.json"
    seg_sha = sha256(segments_path)
    if src.is_file() and not force:
        old = json.loads(src.read_text(encoding="utf-8"))["segments_pred"]["sha256"]
        if old != seg_sha:
            raise RuntimeError(f"segments_pred.jsonl changed since {src} was written ({old[:12]} -> {seg_sha[:12]}); "
                               "R3-R6 must come from one file. Pass --force only if this is intended.")
    durations = read_durations(subset_path)
    segments = read_segments(segments_path)
    k_pred = {v: len(segments[v]) for v in durations if v in segments}
    parts, stats = build(segments, durations, z_dir, seed, chunk_sec, d_min)
    rows = [check(name, parts[name], durations, k_pred, d_min) for name in NAMES]

    out.mkdir(parents=True, exist_ok=True)
    files = {}
    for name in NAMES:
        path = out / f"{name}.json"
        path.write_text(json.dumps(parts[name]), encoding="utf-8")
        files[name] = {"file": path.name, "sha256": sha256(path), **{k: v for k, v in
                       next(r for r in rows if r["partition"] == name).items() if k != "partition"}}
    stats["segments_pred_sha256"] = seg_sha
    (out / "kmedoids_M.meta.json").write_text(json.dumps(stats, indent=1), encoding="utf-8")
    src.write_text(json.dumps({
        "segments_pred": {"path": str(segments_path), "sha256": seg_sha, **_segments_params(segments_path)},
        "z_dir": str(z_dir), "subset": str(subset_path), "seed": seed, "d_min": d_min, "chunk_sec": chunk_sec,
        "commit": _git_commit(), "created": time.strftime("%Y-%m-%d %H:%M:%S"), "partitions": files,
    }, indent=1), encoding="utf-8")
    print(f"[partitions_uemr] segments_pred sha256 {seg_sha[:16]}... | K-Medoids: mean {stats['mean_iterations']:.1f} "
          f"iterations, {stats['not_converged']} videos not converged -> {out}")
    return rows


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--out", required=True, help="<omni_cache>/common/partitions")
    p.add_argument("--segments", help="<uniav_cache>/ft/segments_pred.jsonl")
    p.add_argument("--z-dir", help="<uniav_cache>/ft/Z")
    p.add_argument("--subset", required=True, help="<omni_cache>/common/subset_videos.json")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--chunk-sec", type=float, default=CHUNK_SEC)
    p.add_argument("--d-min", type=float, default=D_MIN)
    p.add_argument("--force", action="store_true", help="accept a segments_pred.jsonl other than SOURCE.json's")
    p.add_argument("--controls", action="store_true", help="only the P6 §A3 controls, from the files in --out")
    a = p.parse_args(argv)
    if a.controls:
        rows = run_controls(a.out, a.subset, a.seed)
    else:
        if not (a.segments and a.z_dir):
            p.error("--segments and --z-dir are required unless --controls")
        rows = run(a.out, a.segments, a.z_dir, a.subset, a.seed, a.chunk_sec, a.d_min, a.force)
    cols = list(rows[0])
    print("| " + " | ".join(cols) + " |\n|" + "---|" * len(cols))
    for r in rows:
        print("| " + " | ".join(str(r[c]) for c in cols) + " |")
    return 0


if __name__ == "__main__":
    sys.exit(main())
