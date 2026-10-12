"""Context Fusion module (plan P6 §C6.1-C6.3). CPU, tiny d."""

import math

import torch
from torch.nn import functional as F

from uemr_fusion.model import ContextFusion, FusionConfig, windows


def small(**kw):
    return FusionConfig(d=16, heads=4, ff=32, pos_hidden=8, **kw)


def inputs(B=2, K=3, T=256, d=16, seed=0):
    g = torch.Generator().manual_seed(seed)
    dur = torch.tensor([100.0, 60.0][:B])
    segs = torch.tensor([[[0.0, 12.0], [40.0, 55.0], [90.0, 100.0]],
                         [[3.0, 9.0], [20.0, 41.0], [50.0, 58.0]]])[:B, :K]
    t = (torch.arange(T) + 0.5)[None] * torch.tensor([[110.0], [60.0]])[:B] / T     # video 0: n > duration
    return {"E": F.normalize(torch.randn(B, K, d, generator=g), dim=-1), "kmask": torch.ones(B, K, dtype=torch.bool),
            "segs": segs, "Z": torch.randn(B, T, 1024, generator=g), "t_sec": t, "duration": dur}


def test_windows_rows_rel_and_clipping():
    t = (torch.arange(10.0) + 0.5)[None]                     # rows at 0.5 .. 9.5 s
    dur = torch.tensor([8.0])                                # rows 8.5 and 9.5 lie past the mp4 -> 8.0
    segs = torch.tensor([[[0.0, 2.0], [7.0, 8.0], [4.1, 4.2]]])
    win, rel = windows(segs, t, dur, delta=1.0)
    assert win[0, 0].nonzero().flatten().tolist() == [0, 1, 2]           # [-1, 3]
    assert rel[0, 0, :3].tolist() == [1, 1, 2]                           # 2.5 is after te = 2
    assert win[0, 1].nonzero().flatten().tolist() == [6, 7, 8, 9]        # [6, 9], clipped rows included
    assert rel[0, 1, 6:].tolist() == [0, 1, 1, 1]
    win0, _ = windows(segs, t, dur, delta=0.0)
    assert win0[0, 2].nonzero().flatten().tolist() == [4]                # empty window -> row nearest 4.15 s
    assert win0.any(-1).all()                                            # never an empty window


def naive_cross(m, e, segs, z, t, dur):
    """Plain MHA with the key/value tensor built explicitly, one segment at a time (pseudocode of §13)."""
    B, K, d = e.shape
    H = m.cfg.heads
    dh = d // H
    win, rel = windows(segs, t, dur, m.cfg.delta)
    out = torch.zeros(B, K, d)
    for b in range(B):
        for i in range(K):
            rows = win[b, i].nonzero().flatten()
            kv = m.z_proj(z[b, rows]) + m.rel_emb(rel[b, i, rows])
            q = m.q(e[b, i]).view(H, dh)
            k, v = m.k(kv).view(-1, H, dh), m.v(kv).view(-1, H, dh)
            a = (torch.einsum("hd,nhd->hn", q, k) / math.sqrt(dh)).softmax(-1)
            out[b, i] = m.o(torch.einsum("hn,nhd->hd", a, v).reshape(d))
    return out


def test_batched_cross_attention_equals_plain_formula():
    torch.manual_seed(0)
    m = ContextFusion(small(delta=5.0)).eval()
    x = inputs()
    with torch.no_grad():
        fast = m.cross(x["E"], x["segs"], x["Z"], x["t_sec"], x["duration"])
        ref = naive_cross(m, x["E"], x["segs"], x["Z"], x["t_sec"], x["duration"])
    torch.testing.assert_close(fast, ref, atol=1e-5, rtol=1e-4)


def test_alpha_zero_is_identity_and_only_alpha_gets_gradient():
    torch.manual_seed(0)
    m = ContextFusion(small())
    x = inputs()
    x["kmask"][1, 2] = False
    c = m(**x)
    torch.testing.assert_close(c[x["kmask"]], F.normalize(x["E"], dim=-1)[x["kmask"]], atol=1e-6, rtol=0)
    assert (c[1, 2] == 0).all()                                          # padding row
    (c[x["kmask"]] * torch.randn_like(c[x["kmask"]])).sum().backward()
    assert m.alpha.grad.abs().item() > 0
    assert m.out.weight.grad.abs().max().item() == 0                     # ReZero: opens up once alpha moves
    assert m.z_proj.weight.grad.abs().max().item() == 0


def test_padding_does_not_leak():
    torch.manual_seed(0)
    m = ContextFusion(small()).eval()
    with torch.no_grad():
        m.alpha.fill_(0.7)
        x = inputs()
        x["kmask"][0, 2] = False                                         # video 0 has 2 segments, video 1 has 3
        x["E"][0, 2] = 0
        both = m(**x)
        alone = m(**{k: v[:1, :2] if k in ("E", "kmask", "segs") else v[:1] for k, v in x.items()})
    torch.testing.assert_close(both[0, :2], alone[0], atol=1e-5, rtol=1e-4)


def test_ablation_flags_build_and_run():
    x = inputs()
    for flags in ({"use_cross": False}, {"use_self": False}, {"use_cross": False, "use_self": False}, {"delta": 0.0}):
        m = ContextFusion(small(**flags))
        names = {n for n, _ in m.named_parameters()}
        assert ("z_proj.weight" in names) == flags.get("use_cross", True)
        assert any(n.startswith("inter.") for n in names) == flags.get("use_self", True)
        with torch.no_grad():
            m.alpha.fill_(0.5)
        c = m(**x)
        assert c.shape == x["E"].shape and torch.isfinite(c).all()
        torch.testing.assert_close(c.norm(dim=-1), torch.ones(2, 3), atol=1e-5, rtol=0)
