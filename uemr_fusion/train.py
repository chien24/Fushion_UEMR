"""Train one Context Fusion (plan P6 §C2-C5, runs D/E/F) on one partition of ``fusion_cache``.

    python -m uemr_fusion.train --cache <fusion_cache> --partition uniav_pred --out <runs>/D3 [--seed 0]
        # defaults = frozen config D3 (tau_s 0.05, lam 0.5, c_g on); ablations: --tau-s, --lam 0, --no-global,
        # --no-cross, --no-self, --delta 0, --shuffle-z
    python -m uemr_fusion.train ... --overfit 32 --steps 200          # sanity check C6.6 (train R@1 > 90 %)

Every epoch: one pass over the trainsplit captions in batches with one caption per video, then dev R@1 (773
captions x 104 dev videos, agg max). Early stopping on dev R@1 (patience 5) keeps ``best.pt``. Then val is
scored once (``val.json``, ``ranks.npz``): galleries 394 and 498, column (b) from the same cache, R@1 by GT length
and by near-duplicate count, bootstrap (c) - (b) by video. ``history.jsonl`` has the losses, dev metrics and
alpha of every epoch. A finished run (``val.json`` present) is skipped unless ``--force``; an interrupted one
starts again from scratch (a run takes minutes).
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch

from .data import FusionSplit, derangement, one_per_video_batches
from .evaluate import autocast, final_eval, split_metrics
from .losses import event_loss, lse_scores, segment_scores, video_loss, with_global
from .model import ContextFusion, FusionConfig, n_params


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--cache", required=True, help="fusion_cache directory")
    p.add_argument("--partition", default="uniav_pred")
    p.add_argument("--out", required=True, help="run directory")
    p.add_argument("--tau", type=float, default=0.05, help="InfoNCE temperature")
    p.add_argument("--learn-tau", action="store_true")
    # defaults = the frozen P6 config (D3, NHAN_XET_KET_QUA_P6.md §4): tau_s 0.05, lam 0.5, c_g on
    p.add_argument("--tau-s", type=float, default=0.05, help="LSE temperature of the training video score")
    p.add_argument("--lam", type=float, default=0.5, help="weight of L_evt (0 = L_vid only)")
    p.add_argument("--no-cross", dest="use_cross", action="store_false")
    p.add_argument("--no-self", dest="use_self", action="store_false")
    p.add_argument("--delta", type=float, default=5.0)
    p.add_argument("--shuffle-z", action="store_true", help="control: Z of another video of the batch")
    p.add_argument("--use-global", action="store_true", default=True, help="add c_g = e_g to every video's vectors")
    p.add_argument("--no-global", dest="use_global", action="store_false")
    p.add_argument("--d-h", type=int, default=256, help="width of the fusion (d -> d_h -> d)")
    p.add_argument("--alpha-lr", type=float, default=1e-3,
                   help="lr of alpha (the size of the move of e; with --lr it would reach about 0.006 per epoch)")
    p.add_argument("--heads", type=int, default=8)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--ff", type=int, default=2048)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--wd", type=float, default=0.01)
    p.add_argument("--warmup", type=float, default=0.05, help="share of the steps")
    p.add_argument("--batch", type=int, default=128)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--patience", type=int, default=5)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--dup-csv", default=None, help="omni_cache/<tag>/analysis/gt_per_query.csv")
    p.add_argument("--overfit", type=int, default=0, metavar="N", help="sanity check on N train videos")
    p.add_argument("--steps", type=int, default=200, help="steps of --overfit")
    p.add_argument("--no-val", action="store_true", help="skip the final val scoring")
    p.add_argument("--force", action="store_true")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args(argv)


def optimizer(model, lr: float, wd: float, alpha=None, alpha_lr: float | None = None):
    """AdamW; no weight decay on alpha, biases, norms and embeddings' 1-d tensors (decay would pull alpha to 0).
    ``alpha`` gets its own group with ``alpha_lr``."""
    rest = [p for p in model.parameters() if p.requires_grad and p is not alpha]
    groups = [{"params": [p for p in rest if p.ndim >= 2], "weight_decay": wd},
              {"params": [p for p in rest if p.ndim < 2], "weight_decay": 0.0}]
    if alpha is not None:
        groups.append({"params": [alpha], "weight_decay": 0.0, "lr": alpha_lr or lr})
    return torch.optim.AdamW(groups, lr=lr)


def cosine_warmup(opt, total: int, warmup: float):
    w = max(1, int(round(warmup * total)))
    return torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: (s + 1) / w if s < w else 0.5 * (1 + math.cos(math.pi * min(1.0, (s - w) / max(1, total - w)))))


def train_step(model, ds: FusionSplit, cap: torch.Tensor, a, tau, gen: torch.Generator) -> dict:
    """Losses of one batch of captions (distinct videos)."""
    vidx = ds.cap_video[cap]
    perm = derangement(len(vidx), gen).to(ds.device) if a.shuffle_z and len(vidx) > 1 else None
    b = ds.batch(vidx, perm)
    with autocast(ds.device):
        C = model(b["E"], b["kmask"], b["segs"], b["Z"], b["t_sec"], b["duration"])
    q = torch.nn.functional.normalize(ds.q[cap].float(), dim=-1)
    s = segment_scores(q, C, b["kmask"])                                         # [B, B, K]
    if a.use_global:
        Cg, mg = with_global(C, b["kmask"], b["eg"], b["eg_ok"])
        S = lse_scores(segment_scores(q, Cg, mg), mg, a.tau_s)
    else:
        S = lse_scores(s, b["kmask"], a.tau_s)
    l_vid = video_loss(S, tau)
    ar = torch.arange(len(cap), device=ds.device)
    l_evt, n_evt = event_loss(s[ar, ar], ds.pos[cap], tau)
    return {"loss": l_vid + a.lam * l_evt, "l_vid": l_vid, "l_evt": l_evt, "n_evt": n_evt}


def build(a, d: int):
    cfg = FusionConfig(d=d, d_h=a.d_h, heads=a.heads, delta=a.delta, use_cross=a.use_cross, use_self=a.use_self,
                       dropout=a.dropout, ff=a.ff)
    model = ContextFusion(cfg).to(a.device)
    log_inv_tau = torch.nn.Parameter(torch.tensor(math.log(1 / a.tau), device=a.device), requires_grad=a.learn_tau)
    return cfg, model, log_inv_tau


def run_overfit(a) -> dict:
    """Plan C6.6: N trainsplit videos, ``--steps`` steps, then R@1 of their captions against those N videos."""
    torch.manual_seed(a.seed)
    rng = np.random.default_rng(a.seed)
    full = FusionSplit(a.cache, "train", a.partition, a.device)
    ds = full.subset(rng.choice(full.n_videos, size=min(a.overfit, full.n_videos), replace=False).tolist())
    cfg, model, log_inv_tau = build(a, ds.d)
    opt = optimizer(model, a.lr, a.wd, model.alpha, a.alpha_lr)
    gen = torch.Generator().manual_seed(a.seed)
    before = split_metrics(model, ds, a.use_global, a.shuffle_z)
    step, losses = 0, []
    while step < a.steps:
        for cap in one_per_video_batches(ds.cap_video.cpu().numpy(), a.batch, rng):
            out = train_step(model, ds, torch.as_tensor(cap, device=ds.device), a, 1 / log_inv_tau.exp(), gen)
            opt.zero_grad(set_to_none=True)
            out["loss"].backward()
            opt.step()
            losses.append(float(out["loss"]))
            step += 1
            if step >= a.steps:
                break
    after = split_metrics(model, ds, a.use_global, a.shuffle_z)
    res = {"videos": ds.n_videos, "captions": len(ds.caption_id), "steps": step, "R@1 before": before["R@1"],
           "R@1 after": after["R@1"], "loss first 10": float(np.mean(losses[:10])),
           "loss last 10": float(np.mean(losses[-10:])),
           "alpha": float(model.alpha.detach().cpu()), "pass (> 90)": after["R@1"] > 90}
    print("[overfit]", json.dumps(res))
    out_dir = Path(a.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "overfit.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
    return res


def main(argv=None) -> int:
    a = parse_args(argv)
    if a.overfit:
        run_overfit(a)
        return 0
    out = Path(a.out)
    if (out / "val.json").is_file() and not a.force:
        print(f"[train] {out} is finished (val.json), skipped; --force to redo")
        return 0
    out.mkdir(parents=True, exist_ok=True)
    if a.partition == "global" and a.lam:
        print("[train] global has one vector per video: L_evt off (lam 0)")
        a.lam = 0.0
    torch.manual_seed(a.seed)
    rng = np.random.default_rng(a.seed)
    gen = torch.Generator().manual_seed(a.seed)

    train = FusionSplit(a.cache, "train", a.partition, a.device)
    dev = FusionSplit(a.cache, "dev", a.partition, a.device)
    cfg, model, log_inv_tau = build(a, train.d)
    params = list(model.parameters()) + ([log_inv_tau] if a.learn_tau else [])
    opt = optimizer(torch.nn.ParameterList(params), a.lr, a.wd, model.alpha, a.alpha_lr)
    n_batches = len(one_per_video_batches(train.cap_video.cpu().numpy(), a.batch, np.random.default_rng(0)))
    sched = cosine_warmup(opt, n_batches * a.epochs, a.warmup)
    source = Path(a.cache) / "SOURCE.json"
    config = {"args": vars(a), "model": cfg.to_dict(), "n_params": n_params(model),
              "train": {"videos": train.n_videos, "captions": len(train.caption_id),
                        "captions with an event positive": int((train.pos >= 0).sum())},
              "dev": {"videos": dev.n_videos, "captions": len(dev.caption_id)},
              "cache_source": json.loads(source.read_text(encoding="utf-8")).get("sources") if source.is_file() else None,
              "started": time.strftime("%Y-%m-%d %H:%M:%S")}
    (out / "config.json").write_text(json.dumps(config, indent=1), encoding="utf-8")
    print(f"[train] {a.partition}: {train.n_videos} videos / {len(train.caption_id)} captions "
          f"({config['train']['captions with an event positive']} with tIoU >= 0.5), dev {dev.n_videos} / "
          f"{len(dev.caption_id)} | {n_params(model) / 1e6:.1f}M params | {n_batches} batches/epoch")

    dev0 = split_metrics(None, dev, a.use_global)
    print(f"[train] dev R@1 without fusion (column b): {dev0['R@1']:.2f}")
    hist = out / "history.jsonl"
    hist.write_text(json.dumps({"epoch": 0, "dev": dev0, "alpha": 0.0, "note": "column (b), no fusion"}) + "\n")
    best, best_epoch, step = -1.0, 0, 0
    for epoch in range(1, a.epochs + 1):
        t0, acc = time.time(), {"loss": 0.0, "l_vid": 0.0, "l_evt": 0.0, "n_evt": 0}
        batches = one_per_video_batches(train.cap_video.cpu().numpy(), a.batch, rng)
        model.train()
        for cap in batches:
            o = train_step(model, train, torch.as_tensor(cap, device=a.device), a, 1 / log_inv_tau.exp(), gen)
            opt.zero_grad(set_to_none=True)
            o["loss"].backward()
            opt.step()
            sched.step()
            step += 1
            for k in acc:
                acc[k] += float(o[k].detach()) if torch.is_tensor(o[k]) else o[k]
        m = split_metrics(model, dev, a.use_global, a.shuffle_z)
        alpha = float(model.alpha.detach().cpu())
        row = {"epoch": epoch, "step": step, **{k: v / len(batches) for k, v in acc.items() if k != "n_evt"},
               "evt captions/batch": acc["n_evt"] / len(batches), "dev": m, "alpha": alpha,
               "tau": float(1 / log_inv_tau.detach().exp()), "lr": sched.get_last_lr()[0], "sec": round(time.time() - t0, 1)}
        with open(hist, "a", encoding="utf-8") as f:
            f.write(json.dumps(row) + "\n")
        mark = ""
        if m["R@1"] > best:
            best, best_epoch, mark = m["R@1"], epoch, " *"
            torch.save({"model": model.state_dict(), "config": cfg.to_dict(), "args": vars(a), "epoch": epoch,
                        "dev": m, "tau": row["tau"]}, out / "best.pt")
        print(f"[train] epoch {epoch:2d} | loss {row['loss']:.4f} (vid {row['l_vid']:.4f}, evt {row['l_evt']:.4f}) "
              f"| dev R@1 {m['R@1']:.2f} R@5 {m['R@5']:.2f} MnR {m['MnR']:.1f} | alpha {alpha:+.4f} "
              f"| {row['sec']:.0f} s{mark}", flush=True)
        if epoch - best_epoch >= a.patience:
            print(f"[train] early stop: no dev gain for {a.patience} epochs (best {best:.2f} at epoch {best_epoch})")
            break

    if a.no_val:
        return 0
    ck = torch.load(out / "best.pt", weights_only=False, map_location=a.device)
    model.load_state_dict(ck["model"])
    val = FusionSplit(a.cache, "val", a.partition, a.device)
    report, arrays = final_eval(model, val, dev, a.use_global, a.shuffle_z, a.dup_csv)
    report.update({"run": out.name, "partition": a.partition, "best_epoch": best_epoch, "dev_best": ck["dev"],
                   "dev_without_fusion": dev0, "args": vars(a)})
    np.savez(out / "ranks.npz", **arrays)
    (out / "val.json").write_text(json.dumps(report, indent=1, default=float), encoding="utf-8")
    for gal in ("val394", "all498"):
        b, c = report[f"{gal} (b)"], report[f"{gal} (c)"]
        bs = report[f"{gal} bootstrap (c)-(b) by video"]
        print(f"[val] {gal}: R@1 (b) {b['R@1']:.2f} -> (c) {c['R@1']:.2f} | (c)-(b) {bs['diff']:+.2f} "
              f"[{bs['ci_low']:+.2f}, {bs['ci_high']:+.2f}] by video | VCMR {b['VCMR R@1 (tIoU.5)']:.2f} -> "
              f"{c['VCMR R@1 (tIoU.5)']:.2f} | alpha {report['alpha']:+.4f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
