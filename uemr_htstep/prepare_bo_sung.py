"""Local, before the add-on protocol is locked (plan bổ sung §3, §4, §5.2, §7): the new segment rows, their hashes, the article
folds of Context Fusion, and the new clip count. No label of a query is read except where the plan says so (R3@K_GT, R7n
and the folds use the GT steps / articles by definition).

    python -m uemr_htstep.prepare_bo_sung --uniav-raw <uniav>/uniav_raw.jsonl --segments <main segments dir> \
        --main-source results/uemr_htstep/a4ca43db/segments_SOURCE.json --feats <bo_sung>/uniav/feats.npz \
        --val-manifest val_omni_htstep.jsonl --out <bo_sung>/segments
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from omni_retrieval.manifest import seg_key
from omni_retrieval.partitions_uemr import sha256

from . import bo_sung as bs
from . import data as hd
from .partitions import FILES

D_MIN_R4 = 2.0          # plan bổ sung §3
D_MIN_NOISE = 0.1       # only keeps a noisy segment from collapsing; the plan sets no floor for R7n
R6K_TARGETS = (12, 33)  # option RUN_R6K


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    for flag in ("--uniav-raw", "--segments", "--main-source", "--feats", "--val-manifest", "--out"):
        p.add_argument(flag, required=True)
    p.add_argument("--protocol", default="configs/htstep_uemr_protocol.json")
    a = p.parse_args()
    pr = json.loads(Path(a.protocol).read_text(encoding="utf-8"))
    kho = pr["data"]["gallery"]
    dur = {v: float(pr["data"]["durations"][v]) for v in kho}
    prm = {**pr["params"]["r6_postprocess"], **{k: pr["params"]["r8"][k] for k in ("l_min", "l_fill")}, "theta_grid": pr["rules"]["theta"]["grid"]}
    src = json.loads(Path(a.main_source).read_text(encoding="utf-8"))
    main = {}
    for name, f in FILES.items():
        path = Path(a.segments) / f"{f}.json"
        part = json.loads(path.read_text(encoding="utf-8"))
        # the hash of the file the main notebook wrote (``wj``: indent 1, UTF-8), whatever the layout of the local copy
        if hashlib.sha256(json.dumps({v: part[v] for v in kho}, indent=1, ensure_ascii=False).encode()).hexdigest() != src["files"][f]:
            raise SystemExit(f"CHECK {path} = segments of the main run FAIL")
        main[name] = part
    r6, gt = main["uniav_pred"], main["gt"]
    k_pred = {v: len(r6[v]) for v in kho}
    raw = {json.loads(line)["video_id"]: json.loads(line) for line in open(a.uniav_raw, encoding="utf-8") if line.strip()}
    with np.load(a.feats) as z:
        feats = {str(v): (f.astype(np.float32), t) for v, f, t in zip(z["video_id"], z["feat"], z["t_sec"])}

    rows, info = {}, {}
    rows["r3kgt"] = bs.r3_kgt(gt, dur)
    for s in bs.R4_SEEDS:
        rows[f"r4_s{s}"] = bs.r4(k_pred, dur, s, D_MIN_R4)
    rows["r5"], info["r5"] = bs.r5(feats, k_pred, dur, 0, 1.0)
    dirs = bs.noise_dirs(gt, 0)
    info["noise"] = {}
    for t in bs.NOISE_TARGETS:
        aa, got, part = bs.calibrate(gt, dur, dirs, t, D_MIN_NOISE)
        rows[bs.noise_name(t)] = part
        info["noise"][bs.noise_name(t)] = {"target": t, "a": aa, "mean paired tIoU": got}
    info["r6k"] = {}
    for t in R6K_TARGETS:
        th, k, part = bs.r6k({v: raw[v]["pred"] for v in kho}, dur, t, prm)
        rows[f"r6k{t}"] = part
        info["r6k"][f"r6k{t}"] = {"target K": t, "theta": th, "mean K": k}
    rows["r7fill"] = bs.r7fill(gt, dur, prm["l_min"], prm["l_fill"])

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    files = {}
    for name, part in rows.items():
        bad = [v for v in kho if not part.get(v) or any(te <= ts for ts, te in part[v])]
        if bad:
            raise SystemExit(f"CHECK {name}: every video has valid segments FAIL: {bad[:5]}")
        path = out / f"{name}.json"
        path.write_text(json.dumps({v: part[v] for v in kho}), encoding="utf-8", newline="\n")
        files[name] = sha256(path)
    main_keys = {seg_key(v, s[0], s[1]) for part in main.values() for v in kho for s in part[v]}
    new = {}
    for name, part in rows.items():
        keys = {seg_key(v, s[0], s[1]) for v in kho for s in part[v]}
        new[name] = {"segments": sum(len(part[v]) for v in kho), "vec/video": float(np.mean([len(part[v]) for v in kho])),
                     "new clips": len(keys - main_keys)}
    caps = hd.queries(hd.read_jsonl(a.val_manifest), kho)
    nq = {}
    for c in caps:
        nq[c["video_id"]] = nq.get(c["video_id"], 0) + 1
    folds = bs.article_folds({v: raw[v]["article_gt"] for v in kho}, nq, 0)
    fold_stat = {f: {"videos": sum(x == f for x in folds.values()), "queries": sum(nq.get(v, 0) for v, x in folds.items() if x == f),
                     "articles": len({raw[v]["article_gt"] for v, x in folds.items() if x == f})} for f in ("A", "B")}
    shared = {raw[v]["article_gt"] for v, x in folds.items() if x == "A"} & {raw[v]["article_gt"] for v, x in folds.items() if x == "B"}
    if shared:
        raise SystemExit(f"CHECK no article in both folds FAIL: {sorted(shared)[:5]}")
    summary = {"files": files, "rows": new, "params": {"d_min_r4": D_MIN_R4, "d_min_noise": D_MIN_NOISE, "r5_half_window": 1.0,
                                                       "r6k_targets": list(R6K_TARGETS), "seed": 0},
               "info": info, "folds": {"rule": "article GT, greedy by query count, seed 0", "stats": fold_stat,
                                       "A": sorted(v for v, x in folds.items() if x == "A"), "B": sorted(v for v, x in folds.items() if x == "B")}}
    (out / "prepared.json").write_text(json.dumps(summary, indent=1), encoding="utf-8", newline="\n")
    print(json.dumps({k: v for k, v in summary.items() if k != "folds"} | {"folds": fold_stat}, indent=1))


if __name__ == "__main__":
    main()
