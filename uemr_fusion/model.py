"""Context Fusion (UEMR_FINAL.md §4C, §13; plan P6 §C1), batched over videos and segments.

    step 1  r_i = MHA(W_Q e_i, kv = z_proj(Z_j) + rel_emb(rel_ij)), j in the window [ts_i - delta, te_i + delta]
    step 2  u   = TransformerEncoderLayer(e + r + p), p_i = order_emb(i) + MLP([centre_i, length_i] / duration)
    step 3  c_i = L2norm(e_i + alpha * W_u u_i),  alpha = 0 at init

Step 1 never builds the ``[B, K, T, d]`` key/value tensor (d = 3584): the projections are linear, so
``W_K (z_j + rel_r) = W_K z_j + W_K rel_r`` and the logits / outputs split into a ``Z`` part shared by every
segment of a video and a 3-row ``rel`` part. ``tests/test_model.py`` checks this against the plain formula.

Windows use ``t_sec`` (the centre time of each Z row), clipped to ``[0, duration]`` first (``t_sec`` of
``wii9jNiNl9Y`` runs past the mp4). Every window keeps at least the row nearest the segment centre.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import torch
from torch import nn
from torch.nn import functional as F

BEFORE, INSIDE, AFTER = 0, 1, 2


@dataclass
class FusionConfig:
    d: int                       # Omni embedding size (read from the cache)
    d_z: int = 1024              # UniAV level-0 Z channels (V 512 + A 512)
    heads: int = 8
    delta: float = 5.0           # context window, seconds on each side
    use_cross: bool = True       # step 1
    use_self: bool = True        # step 2
    dropout: float = 0.1         # inside the TransformerEncoderLayer
    ff: int = 2048               # its feed-forward width (PyTorch default)
    max_pos: int = 64            # order embeddings (K_max is 30)
    pos_hidden: int = 256

    def to_dict(self) -> dict:
        return asdict(self)


def windows(segs: torch.Tensor, t_sec: torch.Tensor, duration: torch.Tensor, delta: float):
    """``(in_window [B, K, T] bool, rel_id [B, K, T] long)``.

    ``rel_id`` = 0 before the segment, 1 inside ``[ts, te]``, 2 after it. A window with no row (a short segment
    between two rows of a long video, delta = 0) gets the row nearest the segment centre.
    """
    t = torch.minimum(t_sec.clamp(min=0), duration[:, None])[:, None, :]            # [B, 1, T]
    ts, te = segs[..., 0:1], segs[..., 1:2]                                         # [B, K, 1]
    win = (t >= ts - delta) & (t <= te + delta)
    near = (t - 0.5 * (ts + te)).abs().argmin(-1, keepdim=True)                      # [B, K, 1]
    win = win.scatter(-1, near, True)
    rel = (t >= ts).long() + (t > te).long()
    return win, rel


class ContextFusion(nn.Module):
    def __init__(self, cfg: FusionConfig):
        super().__init__()
        self.cfg = cfg
        d, h = cfg.d, cfg.heads
        assert d % h == 0, f"d={d} is not divisible by heads={h}"
        if cfg.use_cross:
            self.z_proj = nn.Linear(cfg.d_z, d)
            self.rel_emb = nn.Embedding(3, d)
            self.q = nn.Linear(d, d)
            self.k = nn.Linear(d, d)
            self.v = nn.Linear(d, d)
            self.o = nn.Linear(d, d)
        self.order_emb = nn.Embedding(cfg.max_pos, d)
        self.pos_mlp = nn.Sequential(nn.Linear(2, cfg.pos_hidden), nn.GELU(), nn.Linear(cfg.pos_hidden, d))
        if cfg.use_self:
            self.inter = nn.TransformerEncoderLayer(d, h, cfg.ff, cfg.dropout, batch_first=True)
        self.out = nn.Linear(d, d)
        self.alpha = nn.Parameter(torch.zeros(1))

    # ------------------------------------------------------------------ #
    def cross(self, e, segs, z, t_sec, duration):
        """Step 1: ``r`` [B, K, d]."""
        B, K, d = e.shape
        T, H = z.shape[1], self.cfg.heads
        dh = d // H
        win, rel = windows(segs, t_sec, duration, self.cfg.delta)
        zp = self.z_proj(z)                                                          # [B, T, d]
        kz = self.k(zp).view(B, T, H, dh)
        vz = self.v(zp).view(B, T, H, dh)
        kr = F.linear(self.rel_emb.weight, self.k.weight).view(3, H, dh)             # bias already in kz
        vr = F.linear(self.rel_emb.weight, self.v.weight).view(3, H, dh)
        q = self.q(e).view(B, K, H, dh)
        logit = torch.einsum("bkhd,bthd->bhkt", q, kz)
        lr = torch.einsum("bkhd,rhd->bhkr", q, kr)                                   # [B, H, K, 3]
        logit = logit + torch.gather(lr, -1, rel[:, None].expand(B, H, K, T))
        logit = (logit / math.sqrt(dh)).masked_fill(~win[:, None], float("-inf"))
        a = logit.softmax(-1)
        out = torch.einsum("bhkt,bthd->bkhd", a, vz)
        ar = torch.einsum("bhkt,bktr->bhkr", a, F.one_hot(rel, 3).to(a.dtype))       # attention mass per rel id
        out = out + torch.einsum("bhkr,rhd->bkhd", ar, vr)
        return self.o(out.reshape(B, K, d))

    def positions(self, segs, duration, K):
        dur = duration.clamp(min=1e-6)[:, None]
        feats = torch.stack(((segs[..., 0] + segs[..., 1]) / 2 / dur, (segs[..., 1] - segs[..., 0]) / dur), -1)
        order = torch.arange(K, device=segs.device).clamp(max=self.cfg.max_pos - 1)
        return self.order_emb(order)[None] + self.pos_mlp(feats)

    def forward(self, E, kmask, segs, Z, t_sec, duration):
        """``E`` [B, K, d] (padded), ``kmask`` [B, K] True = real segment, ``segs`` [B, K, 2] seconds,
        ``Z`` [B, T, d_z], ``t_sec`` [B, T], ``duration`` [B]. Returns ``c`` [B, K, d] float32, L2-normalised."""
        e = E.float()
        x = e + self.positions(segs.float(), duration.float(), e.shape[1])
        if self.cfg.use_cross:
            x = x + self.cross(e, segs.float(), Z.float(), t_sec.float(), duration.float())
        u = self.inter(x, src_key_padding_mask=~kmask) if self.cfg.use_self else x
        c = F.normalize(e + self.alpha * self.out(u).float(), dim=-1)
        return c.masked_fill(~kmask[..., None], 0.0)   # padding rows exactly 0, never NaN in a backward pass


def n_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
