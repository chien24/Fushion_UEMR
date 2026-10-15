"""Identity of one UniAV output, so caches of different UniAV runs never mix (``docs/UEMR_VIEC_TIEP_THEO.md`` §A5).

The UniAV side hands over files, not a checkpoint: ``segments_pred.jsonl`` (``video_id, ts, te, conf``) and
``Z/<video_id>.npz``, possibly split over several folders. The hash is computed from their content (sorted
segment lines + ``video_id:sha256`` of every Z file), so the same output gives the same hash however it is
split. Everything that depends on UniAV is written under that hash:

    uniav_cache/<tag>_<hash8>/        gallery videos (val + dev)  segments_pred.jsonl, Z/, SOURCE.json
    uniav_cache/<tag>_train_<hash8>/  trainsplit videos           same
    omni_cache/common/partitions_<hash8>/, omni_cache/<tag>/<hash8>/   (Config.variant = hash8)
    fusion_cache/<tag>_<hash8>/, fusion_runs_v2/<tag>_<hash8>/

A step whose output folder carries the hash reruns by itself when the hash changes (the folder is new), and
the old results stay for the before/after table.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

from .partitions_uemr import _git_commit, sha256

SEGMENTS = "segments_pred.jsonl"


def _lines(src_dirs) -> dict[str, list[str]]:
    """``video_id -> [raw json line, ...]`` over every ``segments_pred.jsonl``; a video in two folders is an error."""
    out, seen_in = defaultdict(list), {}
    for d in map(Path, src_dirs):
        here = defaultdict(list)
        with open(d / SEGMENTS, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    here[json.loads(line)["video_id"]].append(line.rstrip("\n"))
        for v, ls in here.items():
            if v in seen_in:
                raise ValueError(f"{v} is in {seen_in[v]} and {d}")
            seen_in[v] = d
            out[v] = ls
    return dict(out)


def _z_files(src_dirs) -> dict[str, Path]:
    out = {}
    for d in map(Path, src_dirs):
        for p in sorted((d / "Z").glob("*.npz")):
            if p.stem in out:
                raise ValueError(f"Z of {p.stem} is in {out[p.stem].parent} and {p.parent}")
            out[p.stem] = p
    return out


def fingerprint(src_dirs) -> dict:
    """``{hash, hash8, videos, segments, z_files}``; ``hash`` = sha256 of the canonical segment lines and Z list."""
    lines, z = _lines(src_dirs), _z_files(src_dirs)
    h = hashlib.sha256()
    for v in sorted(lines):
        for r in sorted(json.dumps(json.loads(l), sort_keys=True) for l in lines[v]):
            h.update(r.encode() + b"\n")
    for v in sorted(z):
        h.update(f"Z {v}:{sha256(z[v])}\n".encode())
    digest = h.hexdigest()
    return {"hash": digest, "hash8": digest[:8], "videos": len(lines), "segments": sum(map(len, lines.values())),
            "z_files": len(z)}


def install(src_dirs, uniav_cache, splits: dict[str, list[str]], tag: str = "ft") -> dict:
    """Copy one UniAV output into ``uniav_cache/<tag>[_<split>]_<hash8>/``, one folder per entry of ``splits``.

    ``splits``: ``{"": gallery video ids, "train": trainsplit ids}`` (``""`` = no split suffix). Every listed
    video must have at least one segment line and a Z file. A folder whose ``SOURCE.json`` already has this
    hash is kept as it is. Returns ``{"hash8", "fingerprint", "dirs": {split: path}}``.
    """
    src_dirs = [Path(d) for d in ([src_dirs] if isinstance(src_dirs, (str, Path)) else src_dirs)]
    fp = fingerprint(src_dirs)
    lines, z = _lines(src_dirs), _z_files(src_dirs)
    delivered = {str(d): json.loads((d / "SOURCE.json").read_text(encoding="utf-8"))
                 for d in src_dirs if (d / "SOURCE.json").is_file()}
    dirs = {}
    for split, ids in splits.items():
        name = f"{tag}_{split}_{fp['hash8']}" if split else f"{tag}_{fp['hash8']}"
        dst = Path(uniav_cache) / name
        dirs[split] = dst
        src = dst / "SOURCE.json"
        if src.is_file() and json.loads(src.read_text(encoding="utf-8")).get("uniav_hash") == fp["hash"]:
            print(f"[uniav_version] {dst} already holds {fp['hash8']}, kept")
            continue
        no_seg = sorted(v for v in ids if v not in lines)
        no_z = sorted(v for v in ids if v not in z)
        if no_seg or no_z:
            raise ValueError(f"{name}: {len(no_seg)} videos without segments {no_seg[:5]}, "
                             f"{len(no_z)} without Z {no_z[:5]}")
        (dst / "Z").mkdir(parents=True, exist_ok=True)
        with open(dst / SEGMENTS, "w", encoding="utf-8", newline="\n") as f:
            for v in sorted(ids):
                f.write("\n".join(lines[v]) + "\n")
        for v in sorted(ids):
            shutil.copy2(z[v], dst / "Z" / f"{v}.npz")
        src.write_text(json.dumps({
            "uniav_hash": fp["hash"], "hash8": fp["hash8"], "split": split or "gallery", "videos": len(ids),
            "segments": sum(len(lines[v]) for v in ids), "files": {SEGMENTS: sha256(dst / SEGMENTS)},
            "z_files": len(ids), "sources": [str(d) for d in src_dirs], "fingerprint": fp, "delivered": delivered,
            "commit": _git_commit(), "created": time.strftime("%Y-%m-%d %H:%M:%S")}, indent=1), encoding="utf-8")
        print(f"[uniav_version] {name}: {len(ids)} videos, {sum(len(lines[v]) for v in ids)} segments")
    return {"hash8": fp["hash8"], "fingerprint": fp, "dirs": dirs}


def hash_of(uniav_dir) -> str:
    """``hash8`` recorded by ``install`` in ``<uniav_dir>/SOURCE.json`` (raises if the folder was not installed)."""
    blob = json.loads((Path(uniav_dir) / "SOURCE.json").read_text(encoding="utf-8"))
    if "hash8" not in blob:
        raise ValueError(f"{uniav_dir}/SOURCE.json has no hash8: not written by uniav_version.install")
    return blob["hash8"]


def z_overrun(z_dir, durations: dict[str, float], tol: float = 1.0) -> list[dict]:
    """Videos whose Z time axis runs past the mp4: ``max(t_sec) > duration + tol`` (feature file longer than the
    mp4, e.g. ``wii9jNiNl9Y``). A video without a Z file is reported with ``t_sec max = None``."""
    out = []
    for v in sorted(durations):
        p = Path(z_dir) / f"{v}.npz"
        if not p.is_file():
            out.append({"video_id": v, "duration": durations[v], "t_sec max": None, "over (s)": None})
            continue
        with np.load(p) as z:
            last = float(z["t_sec"].max())
        if last > durations[v] + tol:
            out.append({"video_id": v, "duration": durations[v], "t_sec max": round(last, 2),
                        "over (s)": round(last - durations[v], 2)})
    return out
