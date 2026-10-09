"""Gather the encoder store into the standard cache files.

``segments.npz`` -- one row per (partition, video, k):
    keys (``<vid>__<partition>__<k>``), seg_key, video_id, partition, ts, te,
    emb (fp16, L2-normalised). Rows sharing a seg_key share the same vector.
``text.npz``     -- one row per val caption of the subset:
    caption_id, video_id, sentence, gt_ts, gt_te, emb (fp16, L2-normalised).
``meta.json``    -- adapter, preprocessing, Omni-fix commit, videos, counts, time.

The store is either the chunk directory ``encode`` writes or a single ``.npz`` in the
``omniretriever.cli extract`` layout -- both are ``<id>__av`` / ``<id>__text`` keys.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

import numpy as np

from .manifest import store_files


class MissingEmbeddings(RuntimeError):
    pass


def _save_npz(path: Path, **arrays) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.stem + ".tmp.npz")
    np.savez(tmp, **arrays)
    os.replace(tmp, path)


def _gather(store, modality: str, wanted: dict[str, list[int]], n_rows: int) -> tuple[np.ndarray, set]:
    """Fill an ``[n_rows, D]`` fp16 matrix from the store, one key at a time.

    ``wanted`` maps an id to the output rows that take its vector (several rows share
    one segment). Each vector is L2-normalised on its own as it is read, so at most one
    fp32 vector plus the fp16 output is in memory -- never a dict of the whole store.
    """
    suffix = f"__{modality}"
    emb, found = None, set()
    for path in store_files(store):
        with np.load(path) as blob:
            for key in blob.files:
                rid = key[: -len(suffix)] if key.endswith(suffix) else None
                if rid not in wanted or rid in found:
                    continue
                v = blob[key].astype(np.float32).reshape(-1)
                if emb is None:
                    emb = np.zeros((n_rows, v.shape[0]), dtype=np.float16)
                emb[wanted[rid]] = (v / max(float(np.linalg.norm(v)), 1e-12)).astype(np.float16)
                found.add(rid)
    return (emb if emb is not None else np.zeros((n_rows, 0), np.float16)), found


def collect_segments(rows: list[dict], av_store, out_path, allow_missing: bool = False) -> dict:
    wanted: dict[str, list[int]] = {}
    for i, r in enumerate(rows):
        wanted.setdefault(r["seg_key"], []).append(i)
    emb, found = _gather(av_store, "av", wanted, len(rows))
    missing = sorted(set(wanted) - found)
    if missing and not allow_missing:
        raise MissingEmbeddings(f"{len(missing)} segments have no embedding in {av_store} "
                                f"(first: {missing[:3]}). Run the av encode cell again.")
    keep = np.array([r["seg_key"] in found for r in rows], dtype=bool)
    kept = [r for r, k in zip(rows, keep) if k]
    _save_npz(Path(out_path),
              keys=np.array([r["key"] for r in kept]),
              seg_key=np.array([r["seg_key"] for r in kept]),
              video_id=np.array([r["video_id"] for r in kept]),
              partition=np.array([r["partition"] for r in kept]),
              ts=np.array([r["ts"] for r in kept], dtype=np.float32),
              te=np.array([r["te"] for r in kept], dtype=np.float32),
              emb=emb[keep] if not keep.all() else emb)
    return {"rows": len(kept), "unique": len({r["seg_key"] for r in kept}), "missing": len(missing)}


def collect_text(captions: list[dict], text_store, out_path, allow_missing: bool = False) -> dict:
    wanted = {c["caption_id"]: [i] for i, c in enumerate(captions)}
    emb, found = _gather(text_store, "text", wanted, len(captions))
    missing = [c["caption_id"] for c in captions if c["caption_id"] not in found]
    if missing and not allow_missing:
        raise MissingEmbeddings(f"{len(missing)} captions have no embedding in {text_store} "
                                f"(first: {missing[:3]}). Run the text encode cell again.")
    keep = np.array([c["caption_id"] in found for c in captions], dtype=bool)
    kept = [c for c, k in zip(captions, keep) if k]
    _save_npz(Path(out_path),
              caption_id=np.array([c["caption_id"] for c in kept]),
              video_id=np.array([c["video_id"] for c in kept]),
              sentence=np.array([c["sentence"] for c in kept]),
              gt_ts=np.array([c["gt_ts"] for c in kept], dtype=np.float32),
              gt_te=np.array([c["gt_te"] for c in kept], dtype=np.float32),
              gt_seg_key=np.array([c["gt_seg_key"] for c in kept]),
              emb=emb[keep] if not keep.all() else emb)
    return {"captions": len(kept), "missing": len(missing)}


def _read_json(path) -> dict | None:
    path = Path(path)
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else None


def _read_lines(path) -> list[dict]:
    path = Path(path)
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def collect(cfg, prepared: dict, allow_missing: bool = False) -> dict:
    """Write segments.npz / text.npz / meta.json into ``cfg.out_dir``.

    ``prepared`` is the dict ``manifest.prepare`` returns (rows, captions, subset, ...).
    """
    out = Path(cfg.out_dir)
    seg = collect_segments(prepared["rows"], cfg.av_store, out / "segments.npz", allow_missing)
    txt = collect_text(prepared["captions"], cfg.text_store, out / "text.npz", allow_missing)
    enc = _read_json(Path(cfg.av_store) / "encoder_meta.json") or {}
    meta = {
        "tag": cfg.tag,
        "adapter": cfg.adapter,
        "adapter_source": _read_json(Path(cfg.adapter) / "SOURCE.json"),
        "encoder": enc,
        "text_encoder": _read_json(Path(cfg.text_store) / "encoder_meta.json"),
        "encode_runs": {m: _read_lines(Path(s) / "runs.jsonl")
                        for m, s in (("av", cfg.av_store), ("text", cfg.text_store))},
        "partitions": {name: sum(len(s) for s in segs.values()) for name, segs in prepared["parts"].items()},
        "extra_partitions": cfg.extra_partitions,
        "segments": seg, "text": txt,
        "n_videos": len(prepared["subset"]),
        "videos": prepared["subset"],
        "settings": {"n_videos": cfg.n_videos, "seed": cfg.seed, "smoke": cfg.smoke},
        "collected_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    (out / "meta.json").write_text(json.dumps(meta, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"[collect] segments: {seg} | text: {txt} -> {out}")
    return meta
