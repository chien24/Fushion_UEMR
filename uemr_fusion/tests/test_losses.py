"""Scores and losses (plan P6 §C2, §C6.5)."""

import math

import numpy as np
import torch

from omni_retrieval.search import SegmentIndex
from uemr_fusion.losses import event_loss, lse_scores, segment_scores, video_loss, with_global


def test_lse_matches_segment_index():
    rng = np.random.default_rng(0)
    C = torch.nn.functional.normalize(torch.tensor(rng.normal(size=(3, 4, 8)), dtype=torch.float32), dim=-1)
    mask = torch.tensor([[1, 1, 1, 1], [1, 1, 0, 0], [1, 0, 0, 0]], dtype=torch.bool)
    q = torch.nn.functional.normalize(torch.tensor(rng.normal(size=(5, 8)), dtype=torch.float32), dim=-1)
    S = lse_scores(segment_scores(q, C, mask), mask, 0.1)
    index = SegmentIndex(C[mask].numpy(), np.repeat(["a", "b", "c"], mask.sum(1).numpy()),
                         np.concatenate([np.arange(k) for k in mask.sum(1).tolist()]),
                         np.concatenate([np.arange(k) + 1 for k in mask.sum(1).tolist()]))
    ref, _ = index.score_matrix(q.numpy(), agg="lse", tau=0.1)
    np.testing.assert_allclose(S.numpy(), ref, atol=1e-5)
    # a small tau_s tends to the max (inference uses MaxSim)
    s = segment_scores(q, C, mask)
    np.testing.assert_allclose(lse_scores(s, mask, 1e-4).numpy(), s.max(-1).values.numpy(), atol=2e-3)


def test_event_loss_hand_computed():
    s = torch.tensor([[1.0, 0.0, float("-inf")], [0.3, 0.2, 0.1]])
    loss, n = event_loss(s, torch.tensor([0, -1]), 1.0)
    assert n == 1 and math.isclose(float(loss), math.log(1 + math.exp(-1)), rel_tol=1e-6)
    loss, n = event_loss(s, torch.tensor([-1, -1]), 0.05)
    assert n == 0 and float(loss) == 0.0


def test_video_loss_prefers_the_diagonal():
    S = torch.eye(4) * 0.5
    assert float(video_loss(S, 0.05)) < 1e-3
    assert float(video_loss(S.flip(0), 0.05)) > 5


def test_with_global_appends_one_vector():
    C = torch.zeros(2, 3, 4)
    mask = torch.tensor([[1, 1, 0], [1, 0, 0]], dtype=torch.bool)
    eg = torch.tensor([[2.0, 0, 0, 0], [0, 3.0, 0, 0]])
    Cg, mg = with_global(C, mask, eg, torch.tensor([True, False]))
    assert Cg.shape == (2, 4, 4) and torch.equal(Cg[0, 3], torch.tensor([1.0, 0, 0, 0]))
    assert mg.tolist() == [[True, True, False, True], [True, False, False, False]]
