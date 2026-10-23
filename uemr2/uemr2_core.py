"""UEMR 2 core: align Athena segment vectors into the Omni ft space, then score with MaxSim.

Spec: project doc UEMR_ALIGN_FINAL.md (sections 3-5). Every formula here carries the number of the
section that defines it. This module only does the math; reading caches and writing results is the
job of the notebook (agent doc claude/UEMR2_AGENT_AZ.md).

Notation (one row per segment, all vectors L2-normalised unless said otherwise)
  e   (M, 3584)  Omni ft audio+video vector of the segment            (frozen, cached by UEMR)
  q   (M, 512)   Athena span vector of the same segment               (frozen, clean Athena state_dict)
  h   (M, 3584)  aligned vector  h = normalize(P_phi(q))              (this module trains P_phi)
  t   (C, 3584)  Omni ft text vector of a caption / query             (frozen, cached by UEMR)
  u   (C, 512)   Athena query vector  normalize(W_clip normalize(f_IV2(text) - mu))  (frozen)
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


# --------------------------------------------------------------------------- 3.1 projector
class AlignProjector(nn.Module):
    """P_phi: 512 -> 1024 -> 3584, same shape as Athena's omni_proj (event_archs.py:290-292)."""

    def __init__(self, d_in: int = 512, d_hidden: int = 1024, d_out: int = 3584, tau0: float = 0.07):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d_in, d_hidden), nn.GELU(), nn.Linear(d_hidden, d_out))
        self.log_scale_av = nn.Parameter(torch.tensor(math.log(1 / tau0)))
        self.log_scale_txt = nn.Parameter(torch.tensor(math.log(1 / tau0)))

    def forward(self, q: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.net(q.float()), dim=-1)

    @staticmethod
    def scale(log_scale: torch.Tensor) -> torch.Tensor:
        return log_scale.exp().clamp(max=100.0)


# --------------------------------------------------------------------------- 3.2 losses
def loss_av(h: torch.Tensor, rows: torch.Tensor, bank: torch.Tensor, scale: torch.Tensor,
            neg_mask: torch.Tensor | None = None) -> torch.Tensor:
    """L_av: InfoNCE of each aligned vector against the whole bank of Omni AV vectors.

    h (B, d) aligned vectors of the batch, rows (B,) index of each row's own target in bank,
    bank (N, d) every training target e, neg_mask (B, N) True where a bank entry must not act as a
    negative (same video and tIoU >= 0.5 with the row's segment; its own target stays).
    """
    logit = scale * h @ bank.t()
    if neg_mask is not None:
        keep_own = F.one_hot(rows, bank.shape[0]).bool()
        logit = logit.masked_fill(neg_mask & ~keep_own, -1e4)
    return F.cross_entropy(logit, rows)


def soft_targets(gt: torch.Tensor, iv2_pool: torch.Tensor, alpha: float = 0.5, tau_soft: float = 0.02) -> torch.Tensor:
    """Soft caption targets of Athena's pool loss (event_archs.py:376-378):
    y = (1 - alpha) * onehot(g) + alpha * softmax(<v_g, v_k> / tau_soft) over the pool, v = IV2 caption vectors."""
    sim = iv2_pool[gt] @ iv2_pool.t() / tau_soft
    y = alpha * F.softmax(sim, dim=-1)
    y[torch.arange(len(gt), device=gt.device), gt] += 1 - alpha
    return y


def loss_txt(h: torch.Tensor, gt: torch.Tensor, txt_pool: torch.Tensor, iv2_pool: torch.Tensor,
             scale: torch.Tensor, alpha: float = 0.5, tau_soft: float = 0.02) -> torch.Tensor:
    """L_txt: cross-entropy of each aligned vector over the Omni text pool of unique train captions,
    with soft targets. gt (B,) pool index of the row's caption, -1 if the segment has no caption (row skipped)."""
    ok = gt >= 0
    if not ok.any():
        return h.sum() * 0
    h, gt = h[ok], gt[ok]
    with torch.no_grad():
        y = soft_targets(gt, iv2_pool, alpha, tau_soft)
    logit = scale * h @ txt_pool.t()
    return -(y * F.log_softmax(logit, dim=-1)).sum(-1).mean()


# --------------------------------------------------------------------------- 3.3 training
@dataclass
class TrainData:
    q: torch.Tensor          # (N, 512)  Athena span vectors of the training segments
    e: torch.Tensor          # (N, 3584) Omni ft AV targets of the same segments (the bank)
    cap: torch.Tensor        # (N,) pool index of the caption, -1 when none (tIoU < 0.5 with every GT)
    vid: torch.Tensor        # (N,) video index
    segs: torch.Tensor       # (N, 2) seconds
    txt_pool: torch.Tensor   # (P, 3584) Omni ft text of the unique train captions
    iv2_pool: torch.Tensor   # (P, 512)  IV2 caption vectors of the same captions (centred, L2)


def tiou(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Pairwise temporal IoU between segments a (n, 2) and b (m, 2) -> (n, m)."""
    inter = (torch.minimum(a[:, None, 1], b[None, :, 1]) - torch.maximum(a[:, None, 0], b[None, :, 0])).clamp(min=0)
    union = (a[:, 1] - a[:, 0])[:, None] + (b[:, 1] - b[:, 0])[None] - inter
    return inter / union.clamp(min=1e-6)


def false_negative_pairs(d: TrainData, thr: float = 0.5) -> torch.Tensor:
    """(K, 2) pairs (row, bank entry) of the same video whose segments overlap with tIoU >= thr, row != entry.
    Computed once; these entries are not negatives for that row in L_av (section 3.2)."""
    pairs = []
    vid = d.vid.cpu()
    segs = d.segs.cpu().float()
    for v in torch.unique(vid).tolist():
        ix = (vid == v).nonzero(as_tuple=True)[0]
        m = tiou(segs[ix], segs[ix]) >= thr
        m.fill_diagonal_(False)
        a, b = m.nonzero(as_tuple=True)
        if len(a):
            pairs.append(torch.stack((ix[a], ix[b]), 1))
    return torch.cat(pairs) if pairs else torch.zeros(0, 2, dtype=torch.long)


def batch_mask(pairs: torch.Tensor, rows: torch.Tensor, n_bank: int) -> torch.Tensor:
    """(B, N) mask of the false negatives of the batch rows, from the precomputed pairs."""
    pos = torch.full((n_bank,), -1, dtype=torch.long, device=rows.device)
    pos[rows] = torch.arange(len(rows), device=rows.device)
    p = pairs.to(rows.device)
    p = p[pos[p[:, 0]] >= 0]
    m = torch.zeros(len(rows), n_bank, dtype=torch.bool, device=rows.device)
    m[pos[p[:, 0]], p[:, 1]] = True
    return m


def train_projector(d: TrainData, *, seed: int = 0, epochs: int = 30, batch: int = 512, lr: float = 1e-3,
                    wd: float = 1e-4, lam_txt: float = 1.0, alpha: float = 0.5, tau_soft: float = 0.02,
                    dev_eval=None, patience: int = 5, device: str = 'cuda', log=print) -> tuple[AlignProjector, dict]:
    """Train P_phi with L = L_av + lam_txt * L_txt (section 3.2). Athena and Omni stay frozen.

    dev_eval(model) -> float is the dev criterion (higher is better); the best epoch is kept.
    """
    torch.manual_seed(seed); np.random.seed(seed)
    dev = torch.device(device if torch.cuda.is_available() or device == 'cpu' else 'cpu')
    model = AlignProjector(d.q.shape[1], 1024, d.e.shape[1]).to(dev)
    bank = F.normalize(d.e.float(), dim=-1).to(dev)
    txt_pool = F.normalize(d.txt_pool.float(), dim=-1).to(dev)
    iv2_pool = F.normalize(d.iv2_pool.float(), dim=-1).to(dev)
    dd = TrainData(d.q.to(dev), bank, d.cap.to(dev), d.vid.to(dev), d.segs.float().to(dev), txt_pool, iv2_pool)
    fn_pairs = false_negative_pairs(d).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    steps = epochs * math.ceil(len(d.q) / batch)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps)
    best, best_state, hist, bad = -1e9, None, [], 0
    g = torch.Generator().manual_seed(seed)
    for ep in range(epochs):
        model.train()
        perm = torch.randperm(len(d.q), generator=g).to(dev)
        tot = {'av': 0.0, 'txt': 0.0, 'n': 0}
        for i in range(0, len(perm), batch):
            rows = perm[i:i + batch]
            h = model(dd.q[rows])
            la = loss_av(h, rows, bank, model.scale(model.log_scale_av), batch_mask(fn_pairs, rows, len(bank)))
            lt = loss_txt(h, dd.cap[rows], txt_pool, iv2_pool, model.scale(model.log_scale_txt), alpha, tau_soft)
            loss = la + lam_txt * lt
            opt.zero_grad(); loss.backward(); opt.step(); sched.step()
            tot['av'] += la.item() * len(rows); tot['txt'] += lt.item() * len(rows); tot['n'] += len(rows)
        model.eval()
        crit = dev_eval(model) if dev_eval is not None else -tot['av'] / tot['n']
        hist.append({'epoch': ep + 1, 'L_av': tot['av'] / tot['n'], 'L_txt': tot['txt'] / tot['n'], 'dev': crit,
                     'tau_av': 1 / model.scale(model.log_scale_av).item(),
                     'tau_txt': 1 / model.scale(model.log_scale_txt).item()})
        log(hist[-1])
        if crit > best:
            best, bad = crit, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                break
    model.load_state_dict(best_state)
    return model.cpu().eval(), {'history': hist, 'best_dev': best,
                                'best_epoch': max(hist, key=lambda r: r['dev'])['epoch']}


@torch.no_grad()
def align(model: AlignProjector, q: torch.Tensor, batch: int = 8192) -> torch.Tensor:
    return torch.cat([model(q[i:i + batch]) for i in range(0, len(q), batch)]) if len(q) else q.new_zeros(0, 3584)


# --------------------------------------------------------------------------- 4 scoring
def seg_scores(t: np.ndarray, x: np.ndarray, chunk: int = 512) -> np.ndarray:
    """Cosine of every query with every segment: (C, d) x (M, d) -> (C, M), float32."""
    out = np.empty((len(t), len(x)), np.float32)
    xt = np.ascontiguousarray(x.T.astype(np.float32))
    for i in range(0, len(t), chunk):
        out[i:i + chunk] = t[i:i + chunk].astype(np.float32) @ xt
    return out


@dataclass
class ZStats:
    mu: float
    sd: float

    def __call__(self, s: np.ndarray) -> np.ndarray:
        return (s - self.mu) / self.sd


def zstats(s: np.ndarray) -> ZStats:
    """Mean and std of a (C, M) score matrix on dev, used to z-score the IV2 baseline (section 4.3)."""
    return ZStats(float(s.mean()), float(s.std() + 1e-8))


def fuse(*parts: tuple[float, np.ndarray]) -> np.ndarray:
    """s_i = sum_k w_k * S_k (section 4.2). parts are (weight, (C, M) matrix)."""
    out = None
    for w, s in parts:
        if w == 0:
            continue
        out = w * s if out is None else out + w * s
    return out


def maxsim(s: np.ndarray, seg_vid: np.ndarray, n_vid: int) -> tuple[np.ndarray, np.ndarray]:
    """Video score = max over its segments (section 4.4). Returns (C, V) scores and (C, V) argmax segment
    index (global row in s), -1 for a video with no segment."""
    C = s.shape[0]
    score = np.full((C, n_vid), -np.inf, np.float32)
    arg = np.full((C, n_vid), -1, np.int64)
    order = np.argsort(seg_vid, kind='stable')
    sv = seg_vid[order]
    starts = np.searchsorted(sv, np.arange(n_vid))
    ends = np.searchsorted(sv, np.arange(n_vid), side='right')
    for v in range(n_vid):
        if ends[v] > starts[v]:
            idx = order[starts[v]:ends[v]]
            blk = s[:, idx]
            j = blk.argmax(1)
            score[:, v] = blk[np.arange(C), j]
            arg[:, v] = idx[j]
    return score, arg


def evaluate(score: np.ndarray, arg: np.ndarray, cap_vid: np.ndarray, cap_gt: np.ndarray,
             segs: np.ndarray) -> dict:
    """Strict rank = 1 + number of videos scoring strictly higher than the right one (section 5).
    VCMR R@1: right video ranked first and its best segment has tIoU >= 0.5 with the caption's GT segment."""
    C = score.shape[0]
    right = score[np.arange(C), cap_vid]
    rank = 1 + (score > right[:, None]).sum(1)
    best = arg[np.arange(C), cap_vid]
    a = segs[best]
    inter = np.clip(np.minimum(a[:, 1], cap_gt[:, 1]) - np.maximum(a[:, 0], cap_gt[:, 0]), 0, None)
    union = (a[:, 1] - a[:, 0]) + (cap_gt[:, 1] - cap_gt[:, 0]) - inter
    hit = inter / np.maximum(union, 1e-6) >= 0.5
    return {'rank': rank, 'moment_hit': hit,
            'R@1': 100 * float((rank <= 1).mean()), 'R@5': 100 * float((rank <= 5).mean()),
            'R@10': 100 * float((rank <= 10).mean()), 'MdR': float(np.median(rank)), 'MnR': float(rank.mean()),
            'VCMR@1': 100 * float(((rank <= 1) & hit).mean()),
            'mom@.5|top1': 100 * float(hit[rank <= 1].mean()) if (rank <= 1).any() else float('nan')}


def run(t_parts: list[tuple[float, np.ndarray]], seg_vid, n_vid, cap_vid, cap_gt, segs) -> dict:
    """Fuse score matrices, MaxSim, evaluate."""
    score, arg = maxsim(fuse(*t_parts), seg_vid, n_vid)
    return evaluate(score, arg, cap_vid, cap_gt, segs)


# --------------------------------------------------------------------------- 5 statistics
def paired_bootstrap(a: np.ndarray, b: np.ndarray, cap_vid: np.ndarray, n: int = 1000, seed: int = 0) -> dict:
    """A - B with a 95% CI, resampling the query videos with replacement (all captions of a drawn video
    come along), as in BAO_CAO_UEMR_YC2.md section 3. a, b (C,) per-caption values (e.g. 100 * (rank <= 1))."""
    rng = np.random.default_rng(seed)
    vids = np.unique(cap_vid)
    by = [np.flatnonzero(cap_vid == v) for v in vids]
    diff = a.astype(np.float64) - b.astype(np.float64)
    sums = np.array([diff[ix].sum() for ix in by]); cnts = np.array([len(ix) for ix in by])
    bs = np.empty(n)
    for k in range(n):
        pick = rng.integers(0, len(vids), len(vids))
        bs[k] = sums[pick].sum() / cnts[pick].sum()
    lo, hi = np.percentile(bs, [2.5, 97.5])
    return {'diff': float(diff.mean()), 'lo': float(lo), 'hi': float(hi)}


def select_beta(grid, make_parts, seg_vid, n_vid, cap_vid, cap_gt, segs, key: str = 'R@1') -> tuple[float, list]:
    """Pick the weight on dev: highest key, ties -> the smaller weight (closer to UEMR)."""
    rows = []
    for b in sorted(grid):
        r = run(make_parts(b), seg_vid, n_vid, cap_vid, cap_gt, segs)
        rows.append({'beta': b, **{k: v for k, v in r.items() if k not in ('rank', 'moment_hit')}})
    best = max(rows, key=lambda r: (round(r[key], 6), -r['beta']))
    return best['beta'], rows
