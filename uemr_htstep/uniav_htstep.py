"""UniAV HT-Step (HanLi-Retriever 77fc4fa) for retrieval, without the video's article (plan HTSTEP_UEMR_OMNI_UNIAV.md §3.1).

``EventCaptionTransformer.forward`` takes ``cand`` = the steps of the GT article from the dataset, and ``article_cond``
makes every head attend to their text before the event, boundary, embed and omni span heads run. For a gallery video
that reads the answer of the query. Here the code of the model is not changed: a forward hook on ``article_head`` replaces
``cand`` with the steps of the model's own top-1 article at the point where ``forward`` is about to use it, so the
remaining forward, ``inference`` and ``test_cfg`` run unchanged. The article head sees the features before
``article_cond``, so it does not see any step text of the video.

``segments`` of ``inference`` and ``predict`` are in seconds; ``sec_to_steps`` / ``steps_to_sec`` are the two
conversions of ``libs/datasets/htstep.py:164`` and ``event_archs.py:360``.
"""

from __future__ import annotations

import sys

import numpy as np
import torch

MODES = ("pred", "gt")


def load(repo, ckpt, annotations, taxonomy, step_emb, feat_dir, omni_dir, device="cuda"):
    """``(model, dataset of val_seen)``: the model of ``train.py --eval``, in eval mode, on ``device``."""
    if str(repo) not in sys.path:
        sys.path.insert(0, str(repo))
    from libs.core import cfg
    from libs.datasets import make_dataset
    from libs.modeling import make_multimodal_meta_arch

    data_cfg = dict(cfg["dataset"], json_file=annotations, taxonomy_file=taxonomy, step_emb_file=step_emb,
                    feat_folder=feat_dir, omni_emb_folder=omni_dir, scale_factor=cfg["model"]["scale_factor"],
                    regression_range=cfg["model"]["regression_range"], backbone_arch=cfg["model"]["backbone_arch"])
    ds = make_dataset("htstep", False, ["val_seen"], **data_cfg)
    model = make_multimodal_meta_arch("EventCaptionTransformer", **cfg["model"]).to(device)
    model.set_step_pool(torch.from_numpy(ds.pool_emb).to(device), [ds.article_steps[a] for a in ds.articles])
    model.load_state_dict(torch.load(ckpt, map_location=device, weights_only=True)["state_dict"])
    return model.eval(), ds


@torch.inference_mode()
def predict(model, video: dict, mode: str = "pred", article: int | None = None) -> dict:
    """The dict of ``inference`` for one dataset item, plus ``article_top1`` and ``feat0`` (D, T) after ``article_cond``.

    ``mode='pred'``: the article is the model's top-1 (the main mode). ``mode='gt'``: the GT article of the dataset,
    only for the reproduction checks and the oracle row R6gt. ``article``: this article's steps instead (plan bổ sung
    §5.4, the random-article control); ``mode`` is then ignored.
    """
    if mode not in MODES:
        raise ValueError(f"mode must be one of {MODES}")
    x = dict(video, feats={k: v.to(model.device) for k, v in video["feats"].items()})
    top1, feat0 = {}, {}

    def on_article_head(_module, _inputs, logits):
        top1["i"] = int(logits[0].argmax())
        if article is not None:
            x["cand"] = torch.as_tensor(model.article_steps[int(article)], device=logits.device)
        elif mode == "pred":
            x["cand"] = torch.as_tensor(model.article_steps[top1["i"]], device=logits.device)

    hooks = [model.article_head.register_forward_hook(on_article_head),
             model.article_cond.register_forward_hook(lambda _m, _i, out: feat0.__setitem__("f", out[0][0]))]
    try:
        out = model([x])[0]
    finally:
        for h in hooks:
            h.remove()
    return {**out, "article_top1": top1["i"], "feat0": feat0["f"], "feat_stride": video["feat_stride"],
            "fps": video["fps"]}


def sec_to_steps(segs, feat_stride: float, fps: float, offset: float = 0.5):
    """Seconds -> level-0 steps, as ``HTStepDataset.__getitem__`` does for the GT segments."""
    return np.asarray(segs, dtype=np.float64) * fps / feat_stride - offset


def steps_to_sec(steps, feat_stride: float, fps: float):
    """Level-0 steps -> seconds, as ``EventCaptionTransformer.inference`` (the clamp to the video is left to it)."""
    return (np.asarray(steps, dtype=np.float64) * feat_stride + 0.5 * feat_stride) / fps


@torch.inference_mode()
def student_vectors(model, pred: dict, segs_sec) -> np.ndarray:
    """(N, 3584) fp32: the Omni span head on ``segs_sec`` (seconds), in the conditioning of ``pred`` (from ``predict``)."""
    segs = sec_to_steps(segs_sec, pred["feat_stride"], pred["fps"]).reshape(-1, 2)
    feat0 = pred["feat0"]
    out = model.omni_span(feat0, feat0.shape[1], torch.as_tensor(segs, dtype=feat0.dtype, device=feat0.device))
    return out.float().cpu().numpy()
