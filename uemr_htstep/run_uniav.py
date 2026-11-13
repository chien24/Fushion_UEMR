"""UniAV HT-Step over the gallery, once (plan HTSTEP_UEMR_OMNI_UNIAV.md §3.1, §8 cells 6-7). Local CPU or Colab.

    python -m uemr_htstep.run_uniav proposals --uniav-repo R --ckpt model_best.pth.tar --annotations A --taxonomy T \
        --step-emb step_emb_iv2j.npz --feat-dir k400_feats --omni-dir omni_av_feats --videos gallery.txt --out DIR
    python -m uemr_htstep.run_uniav check --uniav-repo R --annotations A --taxonomy T --out DIR --hf-csv predictions_val_seen.csv
    python -m uemr_htstep.run_uniav student ...same as proposals... --segments DIR/segments --names r0 r2 ...
    python -m uemr_htstep.run_uniav omni ...same as proposals... --train-teacher <teacher set>/train

``proposals`` appends one line per video to ``DIR/uniav_raw.jsonl`` (a rerun skips the videos that are in it): the raw
proposals of both article modes in seconds, the article of the model, and the span-head / teacher cosines of the video.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
import time

import numpy as np

RAW = "uniav_raw.jsonl"


def sha256(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


def props_of(r: dict, pool_ids) -> list[dict]:
    segs, scores, labels = r["segments"].tolist(), r["scores"].tolist(), r["labels"].tolist()
    return [{"ts": round(s, 4), "te": round(e, 4), "conf": round(c, 6), "step": int(pool_ids[lab])}
            for (s, e), c, lab in zip(segs, scores, labels)]


def cmd_proposals(a, model, ds, uh) -> None:
    import torch

    row_of = {x["id"]: i for i, x in enumerate(ds.data_list)}
    videos = [v.strip() for v in open(a.videos, encoding="utf-8") if v.strip()]
    missing = [v for v in videos if v not in row_of]
    if missing:
        sys.exit(f"CHECK videos FAIL: {len(missing)} gallery videos are not in the UniAV dataset, e.g. {missing[:5]}")
    os.makedirs(a.out, exist_ok=True)
    path = os.path.join(a.out, RAW)
    done = {json.loads(line)["video_id"] for line in open(path, encoding="utf-8")} if os.path.exists(path) else set()
    todo = [v for v in videos if v not in done][: a.limit or None]
    t0 = time.time()
    with open(path, "a", encoding="utf-8", newline="\n") as f:
        for n, v in enumerate(todo, 1):
            item = ds[row_of[v]]
            g, p = uh.predict(model, item, "gt"), uh.predict(model, item, "pred")
            cos = torch.nn.functional.cosine_similarity(g["omni_student"].float(), g["omni_teacher"].float(), dim=-1)
            f.write(json.dumps({"video_id": v, "article_top1": p["article_top1"], "article_gt": int(item["article"]),
                                "omni_n": len(cos), "omni_cos_sum": float(cos.sum()),
                                "pred": props_of(p, ds.pool_ids), "gt": props_of(g, ds.pool_ids)}) + "\n")
            f.flush()
            if n % 25 == 0 or n == len(todo):
                dt = time.time() - t0
                print(f"[uniav] {n}/{len(todo)} | {dt / n:.2f} s/video | ETA {dt / n * (len(todo) - n) / 60:.1f} min", flush=True)
    print(f"{RAW}: {len(done) + len(todo)} videos, sha256 {sha256(path)}")


def write_csv(rows: list[dict], key: str, path: str) -> None:
    """The official HT-Step prediction format (``train_utils.write_predictions_csv``)."""
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write("video-id,t-start,t-end,label,score\n")
        for r in rows:
            for p in r[key]:
                f.write(f"{r['video_id']},{p['ts']:.3f},{p['te']:.3f},{p['step']},{p['conf']:.6f}\n")


def official_map(a, csv_path: str) -> dict:
    """``tools/eval_htstep.py`` on the videos that have predictions; ``{'mAP': .., 'mAP@0.3': ...}`` plus the raw output."""
    out = subprocess.run([sys.executable, "-X", "utf8", "tools/eval_htstep.py", os.path.abspath(csv_path), "--annotations",
                          os.path.abspath(a.annotations), "--taxonomy", os.path.abspath(a.taxonomy)],
                         cwd=a.uniav_repo, capture_output=True, text=True, encoding="utf-8", errors="replace")
    if out.returncode:
        sys.exit(f"eval_htstep.py failed: {out.stderr[-800:]}")
    tail = out.stdout.split("videos with predictions")[-1]
    return {**{k: float(v) for k, v in re.findall(r"(mAP(?:@[\d.]+)?) (\d+\.\d+)", tail.split("full")[0])},
            "R@1": float(re.search(r"R@1 (\d+\.\d+)", tail).group(1)), "stdout": out.stdout[-900:]}


def compare_csv(mine: str, reference: str) -> dict:
    """Share of the rows of the model card's ``predictions_val_seen.csv`` found in ours: same video and step, start and end
    within 0.1 s (a CPU / GPU rounding moves a score and swaps neighbours, so rows cannot be compared by position)."""
    import pandas as pd

    m, r = (pd.read_csv(p).sort_values("t-start") for p in (mine, reference))
    j = pd.merge_asof(r, m, on="t-start", by=["video-id", "label"], tolerance=0.1, direction="nearest", suffixes=("", "_m"))
    hit = j["t-end_m"].notna() & ((j["t-end"] - j["t-end_m"]).abs() <= 0.1)
    return {"rows": [len(m), len(r)], "same_videos": set(m["video-id"]) == set(r["video-id"]),
            "matched_pct": 100 * float(hit.mean()), "score_mean_abs": float((j["score"] - j["score_m"]).abs()[hit].mean()),
            "score_max_abs": float((j["score"] - j["score_m"]).abs()[hit].max())}


def cmd_check(a) -> None:
    rows = [json.loads(line) for line in open(os.path.join(a.out, RAW), encoding="utf-8")]
    n_omni = sum(r["omni_n"] for r in rows)
    checks = {"videos": len(rows), "uniav_raw_sha256": sha256(os.path.join(a.out, RAW)),
              "ArtAcc": 100 * float(np.mean([r["article_top1"] == r["article_gt"] for r in rows])),
              "OmniCos": 100 * sum(r["omni_cos_sum"] for r in rows) / max(n_omni, 1), "omni_segments": n_omni}
    for key in ("gt", "pred"):
        p = os.path.join(a.out, f"predictions_{key}.csv")
        write_csv(rows, key, p)
        checks[f"map_{key}"] = official_map(a, p)
        print(f"== article {key}: mAP {checks[f'map_{key}']['mAP']:.2f}", flush=True)
    if a.hf_csv:
        checks["vs_hf_csv"] = compare_csv(os.path.join(a.out, "predictions_gt.csv"), a.hf_csv)
    with open(os.path.join(a.out, "checks.json"), "w", encoding="utf-8") as f:
        json.dump(checks, f, indent=1)
    print(json.dumps({k: v for k, v in checks.items() if not k.startswith("map_")} | {
        k: {x: y for x, y in checks[k].items() if x != "stdout"} for k in ("map_gt", "map_pred")}, indent=1))


def conditioning(model, item, uh, how: str, rng) -> tuple[dict, int]:
    """``(predict output, article used)``: the model's top-1 (``pred``, the main mode), the GT article (``gt``, oracle), or a
    random article other than the top-1 (``rand``, plan bổ sung §5.4; one draw per video in sorted video order)."""
    p = uh.predict(model, item, "pred")
    if how == "pred":
        return p, p["article_top1"]
    if how == "gt":
        return uh.predict(model, item, "gt"), int(item["article"])
    r = int(rng.integers(len(model.article_steps) - 1))
    r += r >= p["article_top1"]
    return uh.predict(model, item, article=r), r


def cmd_student(a, model, ds, uh) -> None:
    os.makedirs(a.out, exist_ok=True)
    parts = {n: json.load(open(os.path.join(a.segments, f"{n}.json"), encoding="utf-8")) for n in a.names}
    row_of = {x["id"]: i for i, x in enumerate(ds.data_list)}
    acc = {n: [] for n in a.names}
    ids = {n: [] for n in a.names}
    used = []
    rng = np.random.default_rng(a.rand_seed)
    t0 = time.time()
    videos = sorted(next(iter(parts.values())))
    for i, v in enumerate(videos, 1):
        p, art = conditioning(model, ds[row_of[v]], uh, a.article, rng)
        used.append(art)
        for n in a.names:
            segs = parts[n][v]
            if not segs:      # benchmark: a video without any proposal above θ has no R6 / R3 segment
                continue
            acc[n].append(uh.student_vectors(model, p, segs))
            ids[n] += [(v, s[0], s[1]) for s in segs]
        if i % 50 == 0:
            print(f"[student] {i}/{len(videos)} {(time.time() - t0) / i:.2f} s/video", flush=True)
    for n in a.names:
        path = os.path.join(a.out, f"student_{n}.npz" if a.article == "pred" else f"student_{n}_{a.article}.npz")
        np.savez(path, emb=np.concatenate(acc[n]).astype(np.float16), video_id=np.array([x[0] for x in ids[n]]),
                 ts=np.array([x[1] for x in ids[n]], np.float32), te=np.array([x[2] for x in ids[n]], np.float32),
                 article_video=np.array(videos), article=np.array(used))
        print(f"{path}: {len(ids[n])} segments, sha256 {sha256(path)}")


def train_teacher_mean(folder) -> np.ndarray:
    """Mean of every L2-normalised train teacher vector in ``folder`` (``<id>.npz`` with ``emb``). HanLi's
    ``omni_teacher_mean`` averages the same rows minus the few segments its dataset drops (they start after the features
    end), which needs the train features; the difference is checked against the checkpoint's own OmniCosC."""
    total, n = 0.0, 0
    for f in sorted(os.listdir(folder)):
        if f.endswith(".npz"):
            with np.load(os.path.join(folder, f)) as z:
                e = z["emb"].astype(np.float64)
            total = total + (e / np.maximum(np.linalg.norm(e, axis=1, keepdims=True), 1e-6)).sum(0)
            n += len(e)
    return (total / max(n, 1)).astype(np.float32)


def cmd_omni(a, model, ds, uh) -> None:
    """Plan chốt 01 §6: span head vs teacher on the GT segments of every video, article GT and article predicted, with
    HanLi's ``omni_student_teacher_scores`` (OmniCos, OmniCosC, OmniR@1); and the check that changing the GT ``cand`` of a
    video leaves the main mode unchanged, on the first ``--cand-check`` videos."""
    import torch
    from libs.utils.train_utils import omni_student_teacher_scores

    row_of = {x["id"]: i for i, x in enumerate(ds.data_list)}
    videos = [v.strip() for v in open(a.videos, encoding="utf-8") if v.strip()]
    mean = torch.from_numpy(train_teacher_mean(a.train_teacher))
    acc = {m: ([], []) for m in uh.MODES}
    same, t0 = [], time.time()
    for i, v in enumerate(videos, 1):
        item = ds[row_of[v]]
        for m in uh.MODES:
            r = uh.predict(model, item, m)
            if len(r["omni_teacher"]):
                acc[m][0].append(r["omni_student"].float().cpu())
                acc[m][1].append(r["omni_teacher"].float().cpu())
            if m == "pred" and i <= a.cand_check:
                other = int((item["article"] + 1) % len(model.article_steps))
                q = uh.predict(model, dict(item, cand=torch.as_tensor(model.article_steps[other]), article=other), "pred")
                same.append(all(torch.equal(r[k], q[k]) for k in ("segments", "scores", "labels", "omni_student")))
        if i % 50 == 0:
            print(f"[omni] {i}/{len(videos)} {(time.time() - t0) / i:.2f} s/video", flush=True)
    out = {"train_teacher": a.train_teacher, "cand_check": {"videos": len(same), "unchanged": int(sum(same))}}
    for m, (st, te) in acc.items():
        cos, cos_c, r1 = omni_student_teacher_scores(st, te, mean)
        out[m] = {"segments": int(sum(len(x) for x in st)), "videos": len(st), "OmniCos": cos, "OmniCosC": cos_c, "OmniR@1": r1}
    with open(os.path.join(a.out, "omni_checks.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, indent=1)
    print(json.dumps(out, indent=1))
    if len(same) and not all(same):
        sys.exit("CHECK đổi cand GT không đổi chế độ chính: FAIL")


def cmd_feats(a, model, ds, uh) -> None:
    """``feats.npz``: the level-0 features the span head reads (``feat0`` after ``article_cond``, main mode), (V, T, D) fp16, with
    the centre time of every step (``t_sec``) and the top-1 article: Z of Context Fusion and the input of R5 (plan bổ sung §3, §5.2)."""
    os.makedirs(a.out, exist_ok=True)
    row_of = {x["id"]: i for i, x in enumerate(ds.data_list)}
    videos = [v.strip() for v in open(a.videos, encoding="utf-8") if v.strip()]
    feats, tsec, top1 = [], [], []
    t0 = time.time()
    for i, v in enumerate(videos, 1):
        p = uh.predict(model, ds[row_of[v]], "pred")
        f = p["feat0"].float().cpu().numpy().T                       # (T, D)
        feats.append(f.astype(np.float16))
        tsec.append(uh.steps_to_sec(np.arange(len(f)), p["feat_stride"], p["fps"]).astype(np.float32))
        top1.append(p["article_top1"])
        if i % 50 == 0:
            print(f"[feats] {i}/{len(videos)} {(time.time() - t0) / i:.2f} s/video", flush=True)
    path = os.path.join(a.out, "feats.npz")
    np.savez(path, video_id=np.array(videos), feat=np.stack(feats), t_sec=np.stack(tsec), article_top1=np.array(top1))
    print(f"{path}: {np.stack(feats).shape}, sha256 {sha256(path)}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    common = argparse.ArgumentParser(add_help=False)
    for flag in ("--uniav-repo", "--annotations", "--taxonomy", "--out"):
        common.add_argument(flag, required=True)
    model_args = argparse.ArgumentParser(add_help=False)
    for flag in ("--ckpt", "--step-emb", "--feat-dir", "--omni-dir", "--videos"):
        model_args.add_argument(flag, required=True)
    model_args.add_argument("--device", default="cuda")
    pr = sub.add_parser("proposals", parents=[common, model_args])
    pr.add_argument("--limit", type=int, default=0, help="only the first N videos left to do (timing)")
    st = sub.add_parser("student", parents=[common, model_args])
    st.add_argument("--segments", required=True)
    st.add_argument("--names", nargs="+", required=True)
    st.add_argument("--article", choices=("pred", "gt", "rand"), default="pred")
    st.add_argument("--rand-seed", type=int, default=0)
    sub.add_parser("feats", parents=[common, model_args])
    om = sub.add_parser("omni", parents=[common, model_args])
    om.add_argument("--train-teacher", required=True, help="<teacher set>/train: the mean removed by OmniCosC")
    om.add_argument("--cand-check", type=int, default=20)
    ch = sub.add_parser("check", parents=[common])
    ch.add_argument("--hf-csv", default="")
    a = p.parse_args()
    if a.cmd == "check":
        cmd_check(a)
        return
    from uemr_htstep import uniav_htstep as uh
    model, ds = uh.load(a.uniav_repo, a.ckpt, a.annotations, a.taxonomy, a.step_emb, a.feat_dir, a.omni_dir, a.device)
    {"proposals": cmd_proposals, "student": cmd_student, "feats": cmd_feats, "omni": cmd_omni}[a.cmd](a, model, ds, uh)


if __name__ == "__main__":
    main()
