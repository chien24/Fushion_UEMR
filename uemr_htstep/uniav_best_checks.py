"""Plan chốt 01 §6: the table old UniAV vs new UniAV on the 510 gallery videos, and the segment-source rule, into
``results/uemr_final/uniav_best_checks.json``. Inputs are the outputs of ``run_uniav proposals / check / omni`` of each
checkpoint; the recall of the GT steps uses the θ rule of the dev runs (mean segment count = train steps per video).

    python -m uemr_htstep.uniav_best_checks --old D:/Học/KL/output/htstep_uemr/uniav --new D:/Học/KL/output/uemr_chot/uniav_best \
        --new-ckpt .../model_best.pth.tar --val-manifest val_omni_htstep.jsonl --out results/uemr_final/uniav_best_checks.json
"""

from __future__ import annotations

import argparse
import json
import os

import numpy as np

from omni_retrieval.partitions_uemr import best_tiou

from . import data as hd
from . import partitions as hp
from .run_uniav import RAW, sha256


def recall(raw_path: str, pr: dict, gt: dict) -> dict:
    """θ by the rule of the protocol on this UniAV's predicted-article proposals, then recall of the GT steps."""
    raw = {json.loads(line)["video_id"]: json.loads(line) for line in open(raw_path, encoding="utf-8") if line.strip()}
    kho = pr["data"]["gallery"]
    dur = {v: float(pr["data"]["durations"][v]) for v in kho}
    p = {**pr["params"]["r6_postprocess"], **{k: pr["params"]["r8"][k] for k in ("l_min", "l_fill")}, "theta_grid": pr["rules"]["theta"]["grid"]}
    rs = hp.resolve({v: raw[v]["pred"] for v in kho}, {v: raw[v]["gt"] for v in kho}, dur, gt, pr["resolved"]["train_steps_per_video"], p)
    bt = best_tiou({v: rs["parts"]["uniav_pred"][v] for v in kho}, {v: gt[v] for v in kho})
    return {"theta": rs["theta"], "mean_k": rs["mean_k"], **{f"recall tIoU {t}": 100 * float((bt >= t).mean()) for t in (0.3, 0.5, 0.7)}}


def row(d: str, pr: dict, gt: dict) -> dict:
    ch = json.load(open(os.path.join(d, "checks.json"), encoding="utf-8"))
    out = {"uniav_raw_sha256": sha256(os.path.join(d, RAW)), "mAP article GT": ch["map_gt"]["mAP"], "mAP article đoán": ch["map_pred"]["mAP"],
           "ArtAcc": ch["ArtAcc"], **recall(os.path.join(d, RAW), pr, gt)}
    om = os.path.join(d, "omni_checks.json")
    if os.path.exists(om):
        out["omni"] = json.load(open(om, encoding="utf-8"))
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    for flag in ("--old", "--new", "--new-ckpt", "--val-manifest", "--out"):
        p.add_argument(flag, required=True)
    p.add_argument("--protocol", default="configs/htstep_uemr_protocol.json")
    a = p.parse_args()
    pr = json.load(open(a.protocol, encoding="utf-8"))
    gt = hd.gt_partition(hd.queries(hd.read_jsonl(a.val_manifest), pr["data"]["gallery"]))
    old, new = row(a.old, pr, gt), row(a.new, pr, gt)
    # plan 01 §6: R6 from the UniAV with the higher predicted-article mAP on dev; h_i always from the new one
    source = "mới" if new["mAP article đoán"] >= old["mAP article đoán"] else "cũ"
    out = {"old": {"ckpt_sha256": pr["models"]["uniav"]["sha256"], **old}, "new": {"ckpt_sha256": sha256(a.new_ckpt), **new},
           "segment_source_rule": "đoạn R6 lấy từ UniAV có mAP article đoán cao hơn trên dev; hoà lấy mới; h_i luôn từ UniAV mới",
           "segment_source": source}
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, "w", encoding="utf-8", newline="\n") as f:
        json.dump(out, f, indent=1, ensure_ascii=False)
    print(json.dumps(out, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
