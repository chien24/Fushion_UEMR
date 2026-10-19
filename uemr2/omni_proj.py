"""Stage 1 of docs/UEMR_VIEC_TIEP_THEO.md: retrain only Athena's ``omni_proj`` toward UEMR's Omni vectors.

Athena is frozen: the input is q (512, ``athena_q.span_q``), computed once. h = L2(omni_proj(q)), 3584-d.
Loss, after the teacher block of UniAV-fixed ``libs/modeling/event_archs.py:483-498`` (commit 48918c5):

* AV term: InfoNCE h_i -> bank of every training segment's Omni vector, positive = the segment's own vector
  (``event_archs.py:493-496``). Change asked for UEMR 2: the other segments of the same video with
  tIoU >= ``SAME_VIDEO_TIOU`` to segment i are left out of the denominator (they show the same step).
* Caption term (GT segments only): cross-entropy of h against the Omni text vectors of the train caption pool
  with soft targets ``alpha * softmax(<c_g, c_j> / tau) + (1 - alpha) * [j = g]``, the similarity taken in the
  InternVideo2 caption space (``event_archs.py:355-385, 488-492``; alpha 0.5, tau 0.02).
* One learnable temperature for both terms: init log(1/0.07), exp clamped at 100 (``event_archs.py:293, 495``).
* No jittered copy of the spans (the R6 / R3 segments play that role).
"""

from __future__ import annotations

import copy
import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from uemr_fusion.data import tiou_matrix

SAME_VIDEO_TIOU = 0.3
SOFT_ALPHA, SOFT_TAU = 0.5, 0.02


class OmniProj(nn.Module):
    """``omni_proj`` + ``omni_logit_scale`` with Athena's names and layout (``event_archs.py:290-293``)."""

    def __init__(self, d_in: int = 512, d_out: int = 3584):
        super().__init__()
        self.omni_proj = nn.Sequential(nn.Linear(d_in, d_in * 2), nn.GELU(), nn.Linear(d_in * 2, d_out))
        self.omni_logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

    def forward(self, q: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.omni_proj(q), dim=-1)

    def scale(self) -> torch.Tensor:
        return self.omni_logit_scale.exp().clamp(max=100)


def av_loss(h, rows, bank, bank_vid, bank_seg, scale, thr: float = SAME_VIDEO_TIOU):
    """InfoNCE of ``h`` [B, D] (rows ``rows`` [B] of the bank) against the whole ``bank`` [N, D]."""
    logits = scale * h @ bank.t()
    drop = (bank_vid[None] == bank_vid[rows][:, None]) & (tiou_matrix(bank_seg[None], bank_seg[rows]) >= thr)
    drop[torch.arange(len(rows), device=rows.device), rows] = False
    return F.cross_entropy(logits.masked_fill(drop, float("-inf")), rows)


def soft_targets(g, pool_iv2, iv2_ok, alpha: float = SOFT_ALPHA, tau: float = SOFT_TAU):
    """Targets [B, P] for pool indices ``g``; a caption without an InternVideo2 vector gets a one-hot target."""
    sim = (pool_iv2[g] @ pool_iv2.t() / tau).masked_fill(~iv2_ok[None], float("-inf"))
    has = iv2_ok[g][:, None]
    tgt = torch.where(has, alpha * F.softmax(sim, dim=-1), torch.zeros_like(sim))
    tgt[torch.arange(len(g), device=g.device), g] += 1 - alpha * has[:, 0].float()
    return tgt


def txt_loss(h, g, pool_t, pool_iv2, iv2_ok, scale):
    tgt = soft_targets(g, pool_iv2, iv2_ok)
    return -(tgt * F.log_softmax(scale * h @ pool_t.t(), dim=-1)).sum(-1).mean()


@torch.no_grad()
def retrieval(text, clips, pos) -> dict:
    """Text -> clip ranks (strict: 1 + clips scored higher than the true one, as A-Z)."""
    s = text.float() @ clips.float().t()
    rank = 1 + (s > s[torch.arange(len(pos), device=s.device), pos][:, None]).sum(1)
    r = rank.cpu().numpy()
    return {"R@1": 100 * float((r <= 1).mean()), "R@5": 100 * float((r <= 5).mean()),
            "R@10": 100 * float((r <= 10).mean()), "MedR": float(np.median(r)), "n": int(len(r))}


def train(data: dict, dev: dict, w_txt: float, epochs: int = 40, batch: int = 256, lr: float = 1e-3,
          wd: float = 1e-4, seed: int = 0, device: str = "cuda", log=print):
    """Train one variant; keep the epoch with the best dev R@1 (text -> h of the dev GT clips).

    ``data``: q [M, 512], e [M, D] (bank, L2), vid [M], seg [M, 2], cap [M] (pool index, -1 = not a GT segment),
    pool_t [P, D] (Omni text, L2), pool_iv2 [P, 512] (L2), iv2_ok [P].
    ``dev``: q [C, 512], e [C, D], text [N, D], pos [N] (clip of each caption).
    Returns ``(best state_dict, best row, history)``.
    """
    torch.manual_seed(seed)
    d = {k: v.to(device) for k, v in data.items()}
    dv = {k: v.to(device) for k, v in dev.items()}
    bank = d["e"].float()
    model = OmniProj(d["q"].shape[1], bank.shape[1]).to(device)
    decay = [p for p in model.parameters() if p.ndim >= 2]
    rest = [p for p in model.parameters() if p.ndim < 2]
    opt = torch.optim.AdamW([{"params": decay, "weight_decay": wd}, {"params": rest, "weight_decay": 0.0}], lr=lr)
    m = len(bank)
    per_ep = math.ceil(m / batch)
    total, warm = epochs * per_ep, per_ep
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: (s + 1) / warm if s < warm else 0.5 * (1 + math.cos(math.pi * (s - warm) / max(1, total - warm))))
    gen = torch.Generator().manual_seed(seed)
    best, best_row, hist = None, None, []
    for ep in range(1, epochs + 1):
        model.train()
        acc = {"av": 0.0, "txt": 0.0}
        for b in torch.randperm(m, generator=gen).split(batch):
            b = b.to(device)
            h = model(d["q"][b].float())
            s = model.scale()
            la = av_loss(h, b, bank, d["vid"], d["seg"], s)
            loss = la
            g = d["cap"][b]
            if w_txt > 0 and (g >= 0).any():
                lt = txt_loss(h[g >= 0], g[g >= 0], d["pool_t"].float(), d["pool_iv2"].float(), d["iv2_ok"], s)
                loss = loss + w_txt * lt
                acc["txt"] += float(lt.detach())
            acc["av"] += float(la.detach())
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
        model.eval()
        with torch.no_grad():
            hd = model(dv["q"].float())
            row = {"epoch": ep, "loss_av": acc["av"] / per_ep, "loss_txt": acc["txt"] / per_ep,
                   "scale": float(model.scale()), "dev_cos_h_e": float((hd * dv["e"].float()).sum(1).mean()),
                   **{f"dev {k}": v for k, v in retrieval(dv["text"], hd, dv["pos"]).items()}}
        hist.append(row)
        log(" | ".join(f"{k} {v:.4g}" if isinstance(v, float) else f"{k} {v}" for k, v in row.items()))
        if best_row is None or row["dev R@1"] > best_row["dev R@1"]:
            best, best_row = copy.deepcopy(model.state_dict()), row
    return best, best_row, hist


def match_captions(cap_video, gt_seg, video_segs, thr: float = 0.5) -> np.ndarray:
    """Row (within its video) of the GT segment of every caption: argmax tIoU, -1 below ``thr``.

    The ``gt`` partition is clamped to the mp4, the caption keeps its raw annotation span, hence tIoU not equality.
    """
    out = np.full(len(cap_video), -1, np.int64)
    for i, (v, g) in enumerate(zip(cap_video, gt_seg)):
        t = tiou_matrix(torch.as_tensor(video_segs[v], dtype=torch.float32), torch.as_tensor(g, dtype=torch.float32))
        if len(t) and float(t.max()) >= thr:
            out[i] = int(t.argmax())
    return out
