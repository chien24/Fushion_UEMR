"""Synthetic checks of uemr2_core: run with  python -m pytest -q test_uemr2_core.py"""
import numpy as np
import torch
from torch.nn import functional as F

from uemr2 import uemr2_core as U


def nrm(x):
    return F.normalize(torch.as_tensor(x, dtype=torch.float32), dim=-1)


def make_world(n_vid=40, seg_per_vid=6, seed=0, d_q=512, d_o=3584):
    """Each segment has a latent topic z; q, e and the caption text are noisy views of z."""
    g = torch.Generator().manual_seed(seed)
    n = n_vid * seg_per_vid
    z = torch.randn(n, 64, generator=g)
    Wq, We, Wt = (torch.randn(64, d, generator=g) for d in (d_q, d_o, d_o))
    q = nrm(z @ Wq + 0.5 * torch.randn(n, d_q, generator=g))
    e = nrm(z @ We + 0.5 * torch.randn(n, d_o, generator=g))
    t = nrm(z @ Wt + 0.5 * torch.randn(n, d_o, generator=g))
    vid = torch.arange(n_vid).repeat_interleave(seg_per_vid)
    start = torch.arange(seg_per_vid).repeat(n_vid).float() * 10
    segs = torch.stack((start, start + 10), 1)
    iv2 = nrm(z @ torch.randn(64, 512, generator=g))
    return q, e, t, vid, segs, iv2


def test_soft_targets_sum_to_one():
    iv2 = nrm(torch.randn(20, 512))
    y = U.soft_targets(torch.tensor([3, 7]), iv2)
    assert torch.allclose(y.sum(-1), torch.ones(2), atol=1e-5)
    assert (y[0, 3] >= 0.5) and (y[1, 7] >= 0.5)


def test_false_negative_pairs():
    segs = torch.tensor([[0., 10.], [1., 10.], [20., 30.], [0., 10.]])
    d = U.TrainData(q=torch.zeros(4, 512), e=torch.zeros(4, 8), cap=torch.zeros(4, dtype=torch.long),
                    vid=torch.tensor([0, 0, 0, 1]), segs=segs, txt_pool=torch.zeros(1, 8), iv2_pool=torch.zeros(1, 512))
    p = {tuple(x) for x in U.false_negative_pairs(d).tolist()}
    assert p == {(0, 1), (1, 0)}           # same video, tIoU 0.9; video 1 is another video
    m = U.batch_mask(U.false_negative_pairs(d), torch.tensor([1, 2]), 4)
    assert m.tolist() == [[True, False, False, False], [False, False, False, False]]


def test_training_aligns_and_beats_random():
    q, e, t, vid, segs, iv2 = make_world(seed=0)
    n = len(q)
    tr = vid < 30
    cap = torch.where(tr, torch.arange(n), torch.full((n,), -1))      # caption index = own row in the pool
    d = U.TrainData(q=q[tr], e=e[tr], cap=cap[tr], vid=vid[tr], segs=segs[tr], txt_pool=t[tr], iv2_pool=iv2[tr])
    te = ~tr

    def dev_eval(m):   # mean cosine between aligned vector and its Omni AV target, held-out videos
        return float((U.align(m, q[te]) * e[te]).sum(-1).mean())

    before = dev_eval(U.AlignProjector().eval())
    model, info = U.train_projector(d, epochs=15, batch=64, device='cpu', dev_eval=dev_eval, log=lambda *_: None)
    after = info['best_dev']
    assert after > before + 0.2, (before, after)
    # text -> segment on held-out videos: aligned vectors must retrieve far above chance
    h = U.align(model, q[te]).numpy()
    s = U.seg_scores(t[te].numpy(), h)
    top1 = (s.argmax(1) == np.arange(len(s))).mean()
    assert top1 > 5 / len(s), top1


def test_maxsim_evaluate_and_beta_zero():
    # 3 videos, 2 segments each, 2 captions
    seg_vid = np.array([0, 0, 1, 1, 2, 2])
    segs = np.array([[0, 5], [5, 10], [0, 5], [5, 10], [0, 5], [5, 10]], float)
    a = np.array([[0.1, 0.9, 0.3, 0.2, 0.0, 0.0],      # caption 0: video 0 seg 1 best
                  [0.5, 0.1, 0.2, 0.6, 0.7, 0.1]])     # caption 1: video 2 beats its video 1
    b = np.array([[0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                  [0.0, 0.0, 0.0, 0.9, 0.0, 0.0]])
    cap_vid = np.array([0, 1]); cap_gt = np.array([[5, 10], [5, 10]], float)
    r0 = U.run([(1.0, a), (0.0, b)], seg_vid, 3, cap_vid, cap_gt, segs)
    assert r0['rank'].tolist() == [1, 2] and r0['R@1'] == 50.0 and r0['VCMR@1'] == 50.0
    r1 = U.run([(1.0, a), (0.5, b)], seg_vid, 3, cap_vid, cap_gt, segs)
    assert r1['rank'].tolist() == [1, 1] and r1['VCMR@1'] == 100.0
    base = U.run([(1.0, a)], seg_vid, 3, cap_vid, cap_gt, segs)
    assert np.array_equal(base['rank'], r0['rank'])      # beta = 0 is exactly UEMR


def test_strict_rank_ties_count_in_favour():
    s = np.array([[0.5, 0.5, 0.4]]); seg_vid = np.array([0, 1, 2]); segs = np.array([[0, 1]] * 3, float)
    r = U.run([(1.0, s)], seg_vid, 3, np.array([1]), np.array([[0, 1]], float), segs)
    assert r['rank'].tolist() == [1]                      # a tie does not push the right video down


def test_bootstrap_ci_and_determinism():
    rng = np.random.default_rng(1)
    cap_vid = np.repeat(np.arange(100), 8)
    a = (rng.random(800) < 0.35) * 100.0
    b = (rng.random(800) < 0.25) * 100.0
    r1 = U.paired_bootstrap(a, b, cap_vid); r2 = U.paired_bootstrap(a, b, cap_vid)
    assert r1 == r2 and r1['lo'] < r1['diff'] < r1['hi']
    same = U.paired_bootstrap(a, a, cap_vid)
    assert same['lo'] == same['hi'] == 0.0


def test_select_beta_prefers_smaller_on_tie():
    seg_vid = np.array([0, 1]); segs = np.array([[0, 1], [0, 1]], float)
    a = np.array([[0.9, 0.1]]); b = np.zeros_like(a)
    beta, rows = U.select_beta([0, 0.5, 1], lambda w: [(1.0, a), (w, b)], seg_vid, 2,
                               np.array([0]), np.array([[0, 1]], float), segs)
    assert beta == 0 and len(rows) == 3
