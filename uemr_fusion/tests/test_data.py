"""Cache building, loading, positives and the sampler (plan P6 §B4, §C6.4-C6.5)."""

import json

import numpy as np
import pytest
import torch

from omni_retrieval.evaluate import evaluate
from uemr_fusion.build_cache import parity
from uemr_fusion.data import FusionSplit, derangement, one_per_video_batches, positives


def test_sampler_one_caption_per_video_and_full_coverage():
    rng = np.random.default_rng(0)
    cap_video = np.repeat(np.arange(9), [7, 1, 3, 5, 2, 2, 4, 1, 6])
    batches = one_per_video_batches(cap_video, 4, rng)
    seen = np.concatenate(batches)
    assert sorted(seen.tolist()) == list(range(len(cap_video)))           # every caption exactly once
    for b in batches:
        assert 1 <= len(b) <= 4
        assert len(set(cap_video[b].tolist())) == len(b)                  # no two captions of one video
    assert len(batches) == 8                                              # ceil(31 / 4); no tail of tiny batches
    again = one_per_video_batches(cap_video, 4, np.random.default_rng(0))
    assert [b.tolist() for b in again] == [b.tolist() for b in batches]   # seeded


def test_positives_hand_computed():
    segs = torch.tensor([[[0.0, 10.0], [10.0, 20.0], [20.0, 30.0]]])
    kmask = torch.tensor([[True, True, True]])
    gt = torch.tensor([[9.0, 21.0], [0.0, 30.0], [10.0, 20.0]])
    pos = positives(segs, kmask, torch.zeros(3, dtype=torch.long), gt)
    # [9, 21] vs [10, 20]: 10 / 12 -> segment 1; [0, 30]: best tIoU 1/3 < 0.5 -> none; exact match -> 1
    assert pos.tolist() == [1, -1, 1]
    kmask[0, 1] = False                                                    # a padding row is never positive
    assert positives(segs, kmask, torch.zeros(3, dtype=torch.long), gt).tolist() == [-1, -1, -1]


def test_derangement_has_no_fixed_point():
    g = torch.Generator().manual_seed(0)
    for n in range(2, 12):
        p = derangement(n, g)
        assert sorted(p.tolist()) == list(range(n)) and (p != torch.arange(n)).all()


def test_build_cache_files(fusion_cache):
    z = torch.load(fusion_cache / "val_Z.pt", weights_only=False)
    assert z["Z"].shape == (5, 256, 1024) and z["Z"].dtype == torch.float16 and z["t_sec"].shape == (5, 256)
    c = torch.load(fusion_cache / "train_uniav_pred.pt", weights_only=False)
    assert c["d"] == 16 and len(c["video_ids"]) == 8 and all(len(e) == 4 for e in c["E"])
    assert c["eg_ok"].all() and c["captions"]["q"].shape == (16, 16)
    s = c["segs"][0].numpy()
    assert (np.diff(s[:, 0]) > 0).all()                                    # sorted by time
    src = json.loads((fusion_cache / "SOURCE.json").read_text())
    assert any(k.endswith("segments.npz") for k in src["sources"]) and "val_gt" in src["files"]


def test_parity_with_column_b(fusion_cache, tmp_path):
    """MaxSim on the cached E over the val videos = omni_retrieval.evaluate on the gallery cache."""
    subset = json.loads((tmp_path / "subset_videos.json").read_text())["videos"]
    val = {r["video_id"] for r in subset if r["split"] == "val"}
    rows = evaluate(tmp_path / "omni_cache" / "ft", ["uniav_pred", "gt", "global"], "max", val)
    expected = {r["partition"]: round(r["R@1"], 2) for r in rows}
    out = parity(fusion_cache, expected)
    assert len(out) == 3
    with pytest.raises(RuntimeError):
        parity(fusion_cache, {"gt": expected["gt"] + 1.0})


def test_fusion_split(fusion_cache):
    ds = FusionSplit(fusion_cache, "train", "uniav_pred")
    assert ds.E.shape == (8, 4, 16) and ds.kmask.all() and ds.Z.shape == (8, 256, 1024)
    assert (ds.pos >= 0).all()                                             # captions sit exactly on a segment
    b = ds.batch(torch.tensor([3, 1]), torch.tensor([1, 0]))
    assert torch.equal(b["Z"][0], ds.Z[1]) and torch.equal(b["E"][0], ds.E[3])
    sub = ds.subset([1, 5])
    assert sub.video_ids == [ds.video_ids[1], ds.video_ids[5]] and len(sub.caption_id) == 4
    assert set(sub.cap_video.tolist()) == {0, 1}
    assert torch.equal(sub.q, ds.q[(ds.cap_video == 1) | (ds.cap_video == 5)])
    t = ds.text()
    assert t["emb"].shape == (16, 16) and set(t["video_id"]) == set(ds.video_ids)
    g = FusionSplit(fusion_cache, "dev", "global")
    assert g.E.shape == (3, 1, 16)


def test_attention_maps(fusion_cache):
    import torch
    from uemr_fusion.evaluate import attention_maps
    from uemr_fusion.model import ContextFusion, FusionConfig
    ds = FusionSplit(fusion_cache, "dev", "uniav_pred")
    m = ContextFusion(FusionConfig(d=ds.d, d_h=8, heads=4, ff=32, pos_hidden=8))
    maps = attention_maps(m, ds, [0, 2])
    assert [r["video_id"] for r in maps] == [ds.video_ids[0], ds.video_ids[2]]
    for r, v in zip(maps, (0, 2)):
        assert r["attn"].shape == (int(ds.kmask[v].sum()), len(r["t_sec"])) == (len(r["segs"]), 256)
        assert abs(r["attn"].sum(1) - 1).max() < 1e-5
    assert m.training and not m.keep_attn and m.last_attn is None        # state restored
