"""Context Fusion on HT-Step by cross-fitting over two article folds (plan bổ sung §5.2), with the YC2 module unchanged.

``uemr_fusion.model.ContextFusion``, its losses and its caption sampler are reused as they are; the hyper-parameters are the
frozen YC2 config (``uemr_fusion.train`` defaults, run D3), read from the add-on protocol. Two things differ from
``uemr_fusion.train``, both written in the plan: a fixed number of epochs and the last epoch (no dev set, no checkpoint
choice), and in-batch negatives whose video has a step with the caption's text are masked (multi-positive).

Z = the level-0 features of UniAV in the main mode (``run_uniav feats``); ``z_proj`` takes their width. Model A learns on
the videos and queries of fold A and gives the vectors of fold B, and the other way round.
"""

from __future__ import annotations

import math
import time

import numpy as np
import torch

from uemr_fusion.data import FusionSplit, derangement, one_per_video_batches, positives
from uemr_fusion.evaluate import autocast, build_index, fuse, merge
from uemr_fusion.losses import event_loss, lse_scores, segment_scores, video_loss, with_global
from uemr_fusion.model import ContextFusion, FusionConfig, n_params
from uemr_fusion.train import cosine_warmup, optimizer


def make_split(index, global_index, feats: dict, durations: dict, captions: list[dict], q: np.ndarray, videos, name: str,
               device="cpu") -> FusionSplit:
    """A ``FusionSplit`` of ``videos`` built in memory: ``E`` = the Omni vectors of ``index`` (one row), ``eg`` = R0, ``Z`` and
    ``t_sec`` from ``feats[v] = (Z (T, D), t_sec (T,))``; the queries of those videos (``q`` in ``captions`` order)."""
    ds = object.__new__(FusionSplit)
    videos = sorted(videos)
    vi = {v: i for i, v in enumerate(videos)}
    rows = [np.flatnonzero(index.seg_video == v) for v in videos]
    K = max(len(r) for r in rows)
    ds.split, ds.partition, ds.d = name, index.name, int(index.emb.shape[1])
    ds.video_ids = videos
    ds.E = torch.zeros(len(videos), K, ds.d, dtype=torch.float16)
    ds.segs = torch.zeros(len(videos), K, 2)
    ds.kmask = torch.zeros(len(videos), K, dtype=torch.bool)
    for i, r in enumerate(rows):
        ds.E[i, :len(r)] = torch.from_numpy(index.emb[r]).half()
        ds.segs[i, :len(r)] = torch.from_numpy(np.stack([index.ts[r], index.te[r]], 1))
        ds.kmask[i, :len(r)] = True
    ds.Z = torch.from_numpy(np.stack([feats[v][0] for v in videos])).half()
    ds.t_sec = torch.from_numpy(np.stack([feats[v][1] for v in videos])).float()
    ds.duration = torch.tensor([durations[v] for v in videos], dtype=torch.float32)
    g = {v: e for v, e in zip(global_index.seg_video, global_index.emb)}
    ds.eg = torch.from_numpy(np.stack([g[v] for v in videos])).half()
    ds.eg_ok = torch.ones(len(videos), dtype=torch.bool)
    keep = [i for i, c in enumerate(captions) if c["video_id"] in vi]
    ds.caption_id = [captions[i]["caption_id"] for i in keep]
    ds.sentence = [captions[i]["sentence"] for i in keep]
    ds.q = torch.from_numpy(np.asarray(q, np.float32)[keep]).half()
    ds.cap_video = torch.tensor([vi[captions[i]["video_id"]] for i in keep], dtype=torch.long)
    ds.gt_seg = torch.tensor([[captions[i]["gt_ts"], captions[i]["gt_te"]] for i in keep], dtype=torch.float32)
    ds.pos = positives(ds.segs, ds.kmask, ds.cap_video, ds.gt_seg)
    heads = {}
    for c in captions:
        if c["video_id"] in vi:
            heads.setdefault(vi[c["video_id"]], set()).add(c["headline"])
    cap_head = [captions[i]["headline"] for i in keep]
    # same[i, v]: video v has a step with caption i's text (a right answer under multi-positive)
    ds.same = torch.tensor([[h in heads.get(v, ()) for v in range(len(videos))] for h in cap_head], dtype=torch.bool)
    ds.to(device)
    ds.same = ds.same.to(device)
    return ds


def train_fold(ds: FusionSplit, hp: dict, seed: int, device, shuffle_z: bool = False, log=print):
    """``(model, history)`` after ``hp['epochs']`` epochs (the last one is kept)."""
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    gen = torch.Generator().manual_seed(seed)
    lam = hp["lam"] if int(ds.kmask.sum(1).max()) > 1 else 0.0        # one vector per video (R0): L_evt off, as uemr_fusion.train
    cfg = FusionConfig(d=ds.d, d_h=hp["d_h"], d_z=int(ds.Z.shape[-1]), heads=hp["heads"], delta=hp["delta"], dropout=hp["dropout"],
                       ff=hp["ff"])
    model = ContextFusion(cfg).to(device)
    opt = optimizer(model, hp["lr"], hp["wd"], model.alpha, hp["alpha_lr"])
    n_batches = len(one_per_video_batches(ds.cap_video.cpu().numpy(), hp["batch"], np.random.default_rng(0)))
    sched = cosine_warmup(opt, n_batches * hp["epochs"], hp["warmup"])
    hist = []
    for epoch in range(1, hp["epochs"] + 1):
        t0, acc, nb = time.time(), {"loss": 0.0, "l_vid": 0.0, "l_evt": 0.0}, 0
        model.train()
        for cap in one_per_video_batches(ds.cap_video.cpu().numpy(), hp["batch"], rng):
            cap = torch.as_tensor(cap, device=ds.device)
            vidx = ds.cap_video[cap]
            perm = derangement(len(vidx), gen).to(ds.device) if shuffle_z and len(vidx) > 1 else None
            b = ds.batch(vidx, perm)
            with autocast(ds.device):
                C = model(b["E"], b["kmask"], b["segs"], b["Z"], b["t_sec"], b["duration"])
            q = torch.nn.functional.normalize(ds.q[cap].float(), dim=-1)
            s = segment_scores(q, C, b["kmask"])
            Cg, mg = with_global(C, b["kmask"], b["eg"], b["eg_ok"])
            S = lse_scores(segment_scores(q, Cg, mg), mg, hp["tau_s"])
            ar = torch.arange(len(cap), device=ds.device)
            neg_right = ds.same[cap][:, vidx] & (ar[:, None] != ar[None, :])    # another right video in the batch: not a negative
            l_vid = video_loss(S.masked_fill(neg_right, float("-inf")), hp["tau"])
            l_evt, _ = event_loss(s[ar, ar], ds.pos[cap], hp["tau"])
            loss = l_vid + lam * l_evt
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
            nb += 1
            for k, v in (("loss", loss), ("l_vid", l_vid), ("l_evt", l_evt)):
                acc[k] += float(v.detach())
        row = {"epoch": epoch, **{k: v / nb for k, v in acc.items()}, "alpha": float(model.alpha.detach().cpu()),
               "lr": sched.get_last_lr()[0], "sec": round(time.time() - t0, 1)}
        hist.append(row)
        log(f"[cf {ds.split}] epoch {epoch:2d} | loss {row['loss']:.4f} (vid {row['l_vid']:.4f}, evt {row['l_evt']:.4f}) | alpha {row['alpha']:+.4f}")
    if any(not math.isfinite(r["loss"]) for r in hist):
        raise FloatingPointError("Context Fusion loss is not finite")
    return model, {"history": hist, "n_params": n_params(model), "lam": lam, "config": cfg.to_dict()}


def cross_fit(index, global_index, feats: dict, durations: dict, captions: list[dict], q: np.ndarray, folds: dict, hp: dict,
              seed: int, device, shuffle_z: bool = False, log=print):
    """``(SegmentIndex of the fused vectors of every video + c_g, report)``: each fold's vectors come from the model of the other."""
    split = {f: make_split(index, global_index, feats, durations, captions, q, [v for v, x in folds.items() if x == f], f, device)
             for f in ("A", "B")}
    parts, rep = [], {}
    for train, apply in (("A", "B"), ("B", "A")):
        model, r = train_fold(split[train], hp, seed, device, shuffle_z, log)
        vecs = fuse(model, split[apply], shuffle_z, seed)
        parts.append(build_index(split[apply], vecs, use_global=True, name=f"{index.name}+cf"))
        rep[f"model {train} -> fold {apply}"] = {**r, "train videos": split[train].n_videos, "train queries": len(split[train].caption_id),
                                                "queries with an event positive": int((split[train].pos >= 0).sum())}
        del model
    return merge(*parts, name=f"{index.name}+cf"), rep
