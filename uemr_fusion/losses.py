"""Training scores and losses (UEMR_FINAL.md §4D, §6; plan P6 §C2).

    s_vi     = cos(q, c_vi)                                  segment scores, padding = -inf
    S_ts     = tau_s * log mean_i exp(s_i / tau_s)            video score while training (normalised LSE)
    L_vid    = InfoNCE text -> video over the batch videos (one caption per video, so the diagonal is right)
    L_evt    = softmax over the segments of the caption's own video, positive i+ = argmax tIoU (>= 0.5)
"""

from __future__ import annotations

import torch
from torch.nn import functional as F


def segment_scores(q: torch.Tensor, C: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """``q`` [Q, d], ``C`` [V, K, d], ``mask`` [V, K] -> ``s`` [Q, V, K] with -inf on padding."""
    return torch.einsum("qd,vkd->qvk", q, C).masked_fill(~mask[None], float("-inf"))


def lse_scores(s: torch.Tensor, mask: torch.Tensor, tau_s: float) -> torch.Tensor:
    """``tau_s * (logsumexp(s / tau_s) - log |C(V)|)`` -> [Q, V]; the same as ``SegmentIndex``'s ``lse``."""
    count = mask.sum(-1).clamp(min=1).to(s.dtype)
    return tau_s * (torch.logsumexp(s / tau_s, -1) - torch.log(count)[None])


def video_loss(S: torch.Tensor, tau: torch.Tensor | float) -> torch.Tensor:
    """InfoNCE text -> video: ``S`` [B, B], caption i belongs to video i."""
    return F.cross_entropy(S / tau, torch.arange(len(S), device=S.device))


def event_loss(s_own: torch.Tensor, pos: torch.Tensor, tau: torch.Tensor | float) -> tuple[torch.Tensor, int]:
    """``s_own`` [B, K] scores of each caption against its own video's segments (-inf on padding), ``pos`` [B]
    (-1 = no segment with tIoU >= 0.5). Returns (mean loss over the captions with a positive, their count)."""
    ok = pos >= 0
    n = int(ok.sum())
    if n == 0:
        return s_own.new_zeros(()), 0
    return F.cross_entropy(s_own[ok] / tau, pos[ok]), n


def with_global(C: torch.Tensor, mask: torch.Tensor, eg: torch.Tensor, eg_ok: torch.Tensor):
    """Append ``c_g = e_g`` (not fused) as one more vector of every video (Exp 2, ``use_global``)."""
    g = F.normalize(eg.float(), dim=-1)[:, None]
    return torch.cat((C, g), 1), torch.cat((mask, eg_ok[:, None]), 1)
