"""One (split, partition) of ``fusion_cache`` on the device, the event positives, and the caption sampler.

The whole split fits in GPU memory (train R6: about 1.000 x 256 x 1024 Z in fp16 = 0.5 GB), so a batch is a
gather of padded tensors, with no DataLoader.
"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

EVT_TIOU = 0.5


def tiou_matrix(segs: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
    """tIoU of segments ``[..., K, 2]`` with one GT ``[..., 2]`` -> ``[..., K]``."""
    inter = (torch.minimum(segs[..., 1], gt[..., None, 1]) - torch.maximum(segs[..., 0], gt[..., None, 0])).clamp(min=0)
    union = torch.maximum(segs[..., 1], gt[..., None, 1]) - torch.minimum(segs[..., 0], gt[..., None, 0])
    return torch.where(union > 0, inter / union.clamp(min=1e-9), torch.zeros_like(inter))


def positives(segs: torch.Tensor, kmask: torch.Tensor, cap_video: torch.Tensor, gt_seg: torch.Tensor,
              thr: float = EVT_TIOU) -> torch.Tensor:
    """``i+ = argmax_i tIoU(seg_i, g)`` of every caption's own video, or -1 when that tIoU < ``thr``."""
    t = tiou_matrix(segs[cap_video], gt_seg).masked_fill(~kmask[cap_video], -1.0)    # [N, K]
    best, idx = t.max(-1)
    return torch.where(best >= thr, idx, torch.full_like(idx, -1))


class FusionSplit:
    """Padded tensors of one split / partition: ``E`` [V, K, d] fp16, ``kmask``, ``segs``, ``Z``, ``t_sec``,
    ``duration``, ``eg`` (+ ``eg_ok``); captions ``q`` [N, d], ``cap_video`` [N], ``gt_seg`` [N, 2], ``pos`` [N]."""

    def __init__(self, cache_dir, split: str, partition: str, device="cpu"):
        part = torch.load(Path(cache_dir) / f"{split}_{partition}.pt", weights_only=False)
        z = torch.load(Path(cache_dir) / f"{split}_Z.pt", weights_only=False)
        self.split, self.partition, self.d = split, partition, int(part["d"])
        self.video_ids = list(part["video_ids"])
        V, K = len(self.video_ids), max(len(e) for e in part["E"])
        self.E = torch.zeros(V, K, self.d, dtype=torch.float16)
        self.segs = torch.zeros(V, K, 2)
        self.kmask = torch.zeros(V, K, dtype=torch.bool)
        for i, (e, s) in enumerate(zip(part["E"], part["segs"])):
            self.E[i, :len(e)], self.segs[i, :len(e)], self.kmask[i, :len(e)] = e, s, True
        self.Z = z["Z"][part["z_index"]]
        self.t_sec = z["t_sec"][part["z_index"]].float()
        self.duration = part["duration"].float()
        self.eg, self.eg_ok = part["eg"], part["eg_ok"]
        caps = part["captions"]
        self.caption_id, self.sentence = list(caps["caption_id"]), list(caps["sentence"])
        self.q = caps["q"]
        self.cap_video = caps["video_idx"].long()
        self.gt_seg = caps["gt_seg"].float()
        self.pos = positives(self.segs, self.kmask, self.cap_video, self.gt_seg)
        self.to(device)

    def to(self, device) -> "FusionSplit":
        for k in ("E", "segs", "kmask", "Z", "t_sec", "duration", "eg", "eg_ok", "q", "cap_video", "gt_seg", "pos"):
            setattr(self, k, getattr(self, k).to(device))
        self.device = torch.device(device)
        return self

    @property
    def n_videos(self) -> int:
        return len(self.video_ids)

    def batch(self, vidx: torch.Tensor, z_perm: torch.Tensor | None = None) -> dict:
        """Tensors of videos ``vidx``, K trimmed to the longest of them. ``z_perm``: positions in ``vidx`` whose
        Z (with its ``t_sec``) each video gets instead of its own (the shuffled-Z control)."""
        k = int(self.kmask[vidx].sum(1).max())
        zi = vidx if z_perm is None else vidx[z_perm]
        return {"E": self.E[vidx, :k], "kmask": self.kmask[vidx, :k], "segs": self.segs[vidx, :k],
                "Z": self.Z[zi], "t_sec": self.t_sec[zi], "duration": self.duration[vidx],
                "eg": self.eg[vidx], "eg_ok": self.eg_ok[vidx]}

    def subset(self, videos) -> "FusionSplit":
        """The same split restricted to video indices ``videos`` (and their captions), e.g. the overfit check."""
        new = object.__new__(FusionSplit)
        keep = torch.as_tensor(sorted(videos), device=self.E.device)
        remap = torch.full((self.n_videos,), -1, dtype=torch.long, device=self.E.device)
        remap[keep] = torch.arange(len(keep), device=self.E.device)
        cap = (remap[self.cap_video] >= 0).nonzero().squeeze(1)
        new.split, new.partition, new.d, new.device = self.split, self.partition, self.d, self.device
        new.video_ids = [self.video_ids[int(i)] for i in keep]
        for k in ("E", "segs", "kmask", "Z", "t_sec", "duration", "eg", "eg_ok"):
            setattr(new, k, getattr(self, k)[keep])
        for k in ("q", "gt_seg", "pos"):
            setattr(new, k, getattr(self, k)[cap])
        new.cap_video = remap[self.cap_video[cap]]
        new.caption_id = [self.caption_id[int(i)] for i in cap]
        new.sentence = [self.sentence[int(i)] for i in cap]
        return new

    def text(self) -> dict:
        """Captions in the ``omni_retrieval.evaluate.load_text`` layout (numpy), for ``video_ranks``."""
        gt = self.gt_seg.cpu().numpy()
        return {"emb": self.q.float().cpu().numpy(), "video_id": np.array(self.video_ids)[self.cap_video.cpu().numpy()],
                "gt_ts": gt[:, 0], "gt_te": gt[:, 1], "caption_id": np.array(self.caption_id)}


def one_per_video_batches(cap_video, batch_size: int, rng: np.random.Generator) -> list[np.ndarray]:
    """Every caption exactly once, in batches with at most one caption per video (plan §C3).

    Each batch takes one caption from the videos with the most captions left (random tie-break), so no video
    is left with a long tail of captions that could only go into tiny batches. Batch order is shuffled.
    """
    cap_video = np.asarray(cap_video)
    left = defaultdict(list)
    for i in rng.permutation(len(cap_video)):
        left[int(cap_video[i])].append(int(i))
    batches = []
    while left:
        vids = np.array(list(left))
        tie = rng.permutation(len(vids))
        order = np.lexsort((tie, -np.array([len(left[v]) for v in vids])))
        take = vids[order[:batch_size]]
        batches.append(np.array([left[int(v)].pop() for v in take]))
        for v in take:
            if not left[int(v)]:
                del left[int(v)]
    return [batches[i] for i in rng.permutation(len(batches))]


def derangement(n: int, generator: torch.Generator) -> torch.Tensor:
    """A permutation with no fixed point (n >= 2): every video gets another video's Z."""
    p = torch.randperm(n, generator=generator)
    out = torch.empty(n, dtype=torch.long)
    out[p] = p.roll(-1)
    return out
