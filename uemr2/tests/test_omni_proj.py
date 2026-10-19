"""Stage-1 loss pieces and a tiny end-to-end train (CPU). Run from the Fushion root: python -m pytest uemr2/tests"""

import numpy as np
import torch
import torch.nn.functional as F

from uemr2.omni_proj import SOFT_ALPHA, OmniProj, av_loss, match_captions, retrieval, soft_targets, train


def test_av_loss_drops_overlapping_same_video_segments_only():
    torch.manual_seed(0)
    bank = F.normalize(torch.randn(4, 8), dim=-1)
    vid = torch.tensor([0, 0, 0, 1])
    seg = torch.tensor([[0.0, 10.0], [1.0, 10.0], [20.0, 30.0], [0.0, 10.0]])   # tIoU(0, 1) = 0.9; (0, 3) other video
    h = F.normalize(torch.randn(1, 8), dim=-1)
    rows = torch.tensor([0])
    got = av_loss(h, rows, bank, vid, seg, torch.tensor(10.0))
    logits = 10.0 * h @ bank[[0, 2, 3]].t()
    assert torch.allclose(got, F.cross_entropy(logits, torch.tensor([0])))
    # threshold above 0.9: nothing is dropped
    full = F.cross_entropy(10.0 * h @ bank.t(), rows)
    assert torch.allclose(av_loss(h, rows, bank, vid, seg, torch.tensor(10.0), thr=0.95), full)


def test_soft_targets_rows_sum_to_one_and_one_hot_without_iv2():
    pool = F.normalize(torch.randn(5, 4), dim=-1)
    ok = torch.tensor([True, True, True, False, True])
    t = soft_targets(torch.tensor([0, 3]), pool, ok)
    assert torch.allclose(t.sum(1), torch.ones(2))
    assert t[0, 0] >= 1 - SOFT_ALPHA and t[0, 3] == 0            # no mass on a caption without a vector
    assert torch.equal(t[1], F.one_hot(torch.tensor(3), 5).float())


def test_retrieval_strict_rank():
    text = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    clips = torch.tensor([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    r = retrieval(text, clips, torch.tensor([1, 2]))               # caption 0 ties with clip 0: still rank 1
    assert r["R@1"] == 100.0 and r["MedR"] == 1.0


def test_match_captions_clamped_gt():
    segs = {0: np.array([[0.0, 5.0], [5.0, 9.5]], np.float32)}  # second GT clamped from 10 to 9.5 s
    assert match_captions([0, 0], np.array([[5.0, 10.0], [30.0, 40.0]]), segs).tolist() == [1, -1]


def test_train_learns_a_linear_teacher():
    g = torch.Generator().manual_seed(0)
    W = torch.randn(16, 24, generator=g)
    q = F.normalize(torch.randn(300, 16, generator=g), dim=-1)
    e = F.normalize(q @ W, dim=-1)
    vid = torch.arange(300) // 3
    seg = torch.stack([torch.arange(300) % 3 * 10.0, torch.arange(300) % 3 * 10.0 + 5], 1)
    cap = torch.where(torch.arange(300) % 3 == 0, torch.arange(300) // 3, torch.full((300,), -1))
    pool_t = e[::3]
    data = dict(q=q, e=e, vid=vid, seg=seg, cap=cap, pool_t=pool_t, pool_iv2=F.normalize(torch.randn(100, 8, generator=g), dim=-1),
                iv2_ok=torch.ones(100, dtype=torch.bool))
    qd = F.normalize(torch.randn(50, 16, generator=g), dim=-1)
    ed = F.normalize(qd @ W, dim=-1)
    dev = dict(q=qd, e=ed, text=ed, pos=torch.arange(50))
    state, best, hist = train(data, dev, w_txt=1.0, epochs=15, batch=64, lr=3e-3, device="cpu", log=lambda s: None)
    assert best["dev R@1"] >= 80 and hist[-1]["dev_cos_h_e"] > hist[0]["dev_cos_h_e"]
    assert set(state) == {"omni_proj.0.weight", "omni_proj.0.bias", "omni_proj.2.weight", "omni_proj.2.bias",
                          "omni_logit_scale"}
    m = OmniProj(16, 24)
    m.load_state_dict(state)
