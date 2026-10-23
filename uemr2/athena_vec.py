"""Athena vectors of UEMR 2 (spec UEMR_ALIGN_FINAL.md 3.2, 3.3).

q: one forward of the caption weights (``state_dict``) per video, then ``span_vectors`` of any segments in seconds
   (seconds -> grid with ``uniav_app.timegrid.sec_to_grid``, as ``uniav_app.caption``). 512-d, L2.
u: ``embed_head.embed_captions(text_encoder(texts))`` = N(W_clip N(f_IV2(c) - mu) + b), as ``pipeline.embed_query``.
Caches: ``<dir>/<sha8 of the checkpoint>/<name>.npz`` with ``key`` (seg_key / caption_id) and ``vec``.
``repo_gt_vectors``: ``embeds_gt`` of the training repo's own ``inference`` (check 7.4.3).
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np


def sha256(path, bs=1 << 24) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(bs), b""):
            h.update(chunk)
    return h.hexdigest()


def load(checkpoint, run: str, **cfg_kw):
    """AthenaPipeline of an API checkpoint through ``uniav_app`` (strict load). q uses ``pipe.model`` (best_cap)."""
    from uniav_app.athena_api import load_pipeline
    from uniav_app.config import Config
    cfg = Config.colab(checkpoint=str(checkpoint), expected_run=run, **cfg_kw)
    return load_pipeline(cfg, use_seg_weights=False), cfg


def segments_by_video(*tables) -> dict[str, dict[str, tuple[float, float]]]:
    """``{video: {seg_key: (ts, te)}}`` from tables with ``seg_key``, ``video``, ``segs`` (data.gallery / train_rows)."""
    out: dict[str, dict] = {}
    for t in tables:
        for k, v, (a, b) in zip(t["seg_key"].tolist(), t["video"].tolist(), t["segs"].tolist()):
            out.setdefault(v, {})[k] = (float(a), float(b))
    return out


def q_vectors(pipe, records: dict, segs: dict[str, dict], log_every: int = 200) -> dict[str, np.ndarray]:
    """``{seg_key: q}`` for every segment of ``segs`` (``segments_by_video``); ``records``: uniav_app FeatureRecords."""
    import torch
    from uniav_app.caption import forward_caption_model
    from uniav_app.features.base import load_arrays
    from uniav_app.timegrid import sec_to_grid, spec_params
    tp = spec_params(pipe.spec)
    out = {}
    for i, (v, rows) in enumerate(sorted(segs.items()), 1):
        visual, audio = load_arrays(records[v], pipe.spec)
        fv, fa, n = pipe.spec.prepare(visual, audio)
        forward_caption_model(pipe, fv, fa)
        keys = list(rows)
        g = sec_to_grid([rows[k] for k in keys], n, **tp)
        with torch.no_grad():
            vec = pipe.model.span_vectors(g).float().cpu().numpy()
        out.update(zip(keys, vec))
        if i % log_every == 0 or i == len(segs):
            print(f"[q] {i}/{len(segs)} videos, {len(out)} segments", flush=True)
    return out


def save(path, d: dict, dtype=np.float32) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = sorted(d)
    tmp = path.with_name(path.name + ".part")
    with open(tmp, "wb") as f:
        np.savez(f, key=np.array(keys), vec=np.stack([d[k] for k in keys]).astype(dtype))
    os.replace(tmp, path)


def read(path) -> dict[str, np.ndarray]:
    if not Path(path).is_file():
        return {}
    with np.load(path) as z:
        return dict(zip(z["key"].tolist(), z["vec"].astype(np.float32)))


def q_cache(path, pipe, records, segs) -> dict[str, np.ndarray]:
    """Cached q: only the segments the file lacks are computed (videos with a missing key are re-run).
    shortcut: the file is written once at the end, so an interrupted run redoes its videos (~0.2 s each); upgrade
    to per-shard files if a run ever takes long."""
    have = read(path)
    todo = {v: rows for v, rows in segs.items() if any(k not in have for k in rows)}
    print(f"[q] {path}: {len(have)} cached, {len(todo)}/{len(segs)} videos to run")
    if todo:
        have.update(q_vectors(pipe, records, todo))
        save(path, have)
    return have


def u_vectors(pipe, texts, batch: int = 64) -> np.ndarray:
    """(C, 512) query vectors, spec 3.3 (needs the InternVideo2 video checkpoint for its text tower)."""
    import torch
    enc = pipe.text_encoder
    assert enc is not None, f"no InternVideo2 text tower ({pipe.cfg.iv2_video_encoder})"
    assert pipe.model.embed_head.clip_proj is not None, "checkpoint without clip_proj"
    out = []
    with torch.no_grad():
        for i in range(0, len(texts), batch):
            raw = enc(list(texts[i:i + batch])).to(pipe.device).float()
            out.append(pipe.model.embed_head.embed_captions(raw).cpu().numpy())
    return np.concatenate(out).astype(np.float32)


def repo_gt_vectors(repo, checkpoint, split: str, video_ids, caption_emb_file: str) -> dict:
    """``{video: (segments_sec, embeds_gt)}`` from the Athena training repo's ``EventCaptionTransformer.inference``
    (``libs/modeling/event_archs.py:589-595``) with the API checkpoint's ``state_dict`` (best_cap) and its config."""
    import torch
    sys.path.append(str(repo))                     # appended: the vendored ``athena`` package stays first
    from libs.datasets import YouCook2CaptionDataset
    from libs.modeling import EventCaptionTransformer, upgrade_state_dict
    import yaml
    ck = torch.load(checkpoint, map_location="cpu", weights_only=False)
    cfg = yaml.safe_load(open(Path(repo) / "configs" / "youcook2_event_yc2clean.yaml"))
    dcfg = dict(cfg["dataset"], omni_emb_file=None, caption_emb_file=caption_emb_file,
                json_file=str(Path(repo) / cfg["dataset"]["json_file"]), iv2_folder=str(Path(repo) / cfg["dataset"]["iv2_folder"]))
    ds = YouCook2CaptionDataset(False, [split], **dcfg)
    model = EventCaptionTransformer(**dict(ck["config"]["model"], omni_dim=0))
    model.load_state_dict(upgrade_state_dict({k: v.float() for k, v in ck["state_dict"].items()}), strict=True)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(dev).eval()
    pos = {x["id"]: i for i, x in enumerate(ds.data_list)}
    out = {}
    with torch.no_grad():
        for v in video_ids:
            r = model([ds[pos[v]]])[0]
            out[v] = (np.asarray(r["gt_segments"], np.float64), r["embeds_gt"].float().numpy())
    return out


def source_json(path, **fields) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(fields, indent=1, ensure_ascii=False), encoding="utf-8")
