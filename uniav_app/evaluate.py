"""Evaluation against YouCook2 GT, reported separately for val (main) and dev (seen in training).

a. Localization (label-free): AR@N (N = 1, 5, 10, K_pred, K_GT) at tIoU {0.3, 0.5, 0.7} on the raw
   proposals; P/R/F1 after one-to-one Hungarian matching on the final segments; mean best tIoU per GT;
   class-agnostic mAP@[0.3:0.1:0.7] (raw and final).
b. Counts and lengths: K_pred vs K_GT, % of videos split too little / too much, length histograms.
c. Semantics: judge cosine between the GT caption and the caption of the matched segment, sem-Recall@tIoU
   at tau {0.7, 0.8, 0.9}; the same with ORACLE captions (candidates = the video's GT captions).
d. Errors: by GT duration bin, missed GT, false positives, merges / splits, boundary offsets.
e. Baselines with the same number of segments: uniform K_pred and uniform K_GT.

Files: results.json, per_video.csv, per_gt_event.csv, analysis/{theta_sweep, ar_at_n, map, comparison,
counts, length_hist, semantic, duration_bins, errors, boundary}.csv.
"""

from __future__ import annotations

import time

import numpy as np

from .analysis import DURATION_BINS, analyze_video, count_stats, length_hist, offset_stats
from .baselines import uniform_segments
from .config import CAVEATS, JUDGE_TAUS, MAP_TIOUS, TIOUS, Config
from .io_utils import write_csv, write_json
from .metrics import mean_ap, mean_best_tiou, prf, recall_at_n
from .postprocess import postprocess_uemr

SPLITS = ("val", "dev")
SPLIT_NOTE = {"val": "val (main result)",
              "dev": "dev: đã thấy khi train (ranh giới + câu); reference only"}


# --------------------------------------------------------------------------- #
# Helpers                                                                     #
# --------------------------------------------------------------------------- #

def gt_segs(gt: dict, vids) -> dict[str, list[tuple]]:
    return {v: [(e["ts"], e["te"]) for e in gt[v]] for v in vids}


def raw_segs(raw: dict, vids) -> dict[str, list[tuple]]:
    return {v: [(p["ts"], p["te"], p["conf"]) for p in raw[v]["proposals"]] for v in vids}


def pred_segs(pred: dict, vids) -> dict[str, list[tuple]]:
    return {v: [(e["t_s"], e["t_e"], e["conf"]) for e in pred[v]["events"]] for v in vids}


def uniform_by_video(durations: dict[str, float], k: dict[str, int]) -> dict[str, list[tuple]]:
    return {v: [(s["ts"], s["te"], s["conf"]) for s in uniform_segments(durations[v], k[v])] for v in k}


def split_videos(vids, splits: dict[str, str]) -> dict[str, list[str]]:
    return {s: [v for v in vids if splits.get(v) == s] for s in SPLITS}


# --------------------------------------------------------------------------- #
# theta                                                                       #
# --------------------------------------------------------------------------- #

def theta_sweep(raw: dict, gt: dict, splits: dict[str, str], cfg: Config) -> list[dict]:
    """For every theta of the grid and split: mean K_pred / K_GT, AR@K_GT@0.5 and P/R/F1@0.5 (config 'uemr')."""
    rows = []
    vids = [v for v in raw if v in gt]
    for split, sv in split_videos(vids, splits).items():
        if not sv:
            continue
        gts = gt_segs(gt, sv)
        for th in cfg.theta_grid:
            segs = {v: postprocess_uemr(raw[v]["proposals"], raw[v]["duration"], theta=th, k_max=cfg.k_max,
                                        d_min=cfg.d_min, merge_tiou=cfg.merge_tiou) for v in sv}
            preds = {v: [(s["ts"], s["te"], s["conf"]) for s in segs[v]] for v in sv}
            p = prf(preds, gts, 0.5)
            rows.append({"split": split, "theta": th,
                         "mean_K_pred": float(np.mean([len(preds[v]) for v in sv])),
                         "mean_K_GT": float(np.mean([len(gts[v]) for v in sv])),
                         "AR@K_GT@0.5": recall_at_n(preds, gts, "K_GT", (0.5,))[0.5],
                         "P@0.5": p["precision"], "R@0.5": p["recall"], "F1@0.5": p["f1"]})
    return rows


def choose_theta(rows: list[dict], split: str = "dev") -> tuple[float, str]:
    """theta whose mean K_pred is closest to mean K_GT on ``split`` (UEMR_FINAL.md section 12); the
    smaller theta wins a tie. Falls back to val when the split is absent (SMOKE)."""
    cand = [r for r in rows if r["split"] == split]
    if not cand:
        split = "val"
        cand = [r for r in rows if r["split"] == split]
        print(f"[theta] no {split!r} rows -> choosing on val (SMOKE only; not the UEMR protocol)")
    best = min(cand, key=lambda r: (abs(r["mean_K_pred"] - r["mean_K_GT"]), r["theta"]))
    return float(best["theta"]), split


# --------------------------------------------------------------------------- #
# One split                                                                   #
# --------------------------------------------------------------------------- #

def _loc_block(preds, gts, thrs=TIOUS) -> dict:
    out = {"seg_per_video": float(np.mean([len(preds.get(v, [])) for v in gts])) if gts else 0.0,
           "mean_tIoU": mean_best_tiou(preds, gts)}
    rec = recall_at_n(preds, gts, None, thrs)
    for t in thrs:
        p = prf(preds, gts, t)
        out[f"Rcover@{t}"] = rec[t]
        out[f"P@{t}"], out[f"R@{t}"], out[f"F1@{t}"] = p["precision"], p["recall"], p["f1"]
    return out


def evaluate_split(vids: list[str], gt: dict, raw: dict, pred: dict, pred_api: dict, judge=None) -> dict:
    """Metrics + row tables for one split. ``judge``: ``semantic.Judge`` or None (skips part c)."""
    gts = gt_segs(gt, vids)
    rawp = raw_segs(raw, vids)
    uemr = pred_segs(pred, vids)
    api = pred_segs(pred_api, vids)
    dur = {v: float(raw[v]["duration"]) for v in vids}
    k_pred = {v: len(uemr[v]) for v in vids}
    k_api = {v: len(api[v]) for v in vids}
    k_gt = {v: len(gts[v]) for v in vids}
    res: dict = {"n_videos": len(vids), "n_gt": int(sum(k_gt.values()))}

    # a. localization
    ar_rows = []
    for n in (1, 5, 10, "K_pred", "K_GT"):
        r = recall_at_n(rawp, gts, k_pred if n == "K_pred" else n, TIOUS)
        ar_rows.append({"N": str(n), **{f"R@{t}": r[t] for t in TIOUS}})
    res["AR@N_raw"] = ar_rows
    res["mIoU_raw@K_GT"] = mean_best_tiou(rawp, gts, "K_GT")   # = Athena train.py mIoU (with its GT)
    res["mAP_raw"] = mean_ap(rawp, gts, MAP_TIOUS)
    res["mAP_uemr"] = mean_ap(uemr, gts, MAP_TIOUS)
    res["mAP_api"] = mean_ap(api, gts, MAP_TIOUS)

    # e. same-count baselines
    methods = {"uniav_uemr": uemr, "uniav_api": api,
               "uniform_K_pred": uniform_by_video(dur, k_pred), "uniform_K_GT": uniform_by_video(dur, k_gt)}
    res["comparison"] = {m: _loc_block(p, gts) for m, p in methods.items()}

    # b. counts / lengths
    res["counts_uemr"] = count_stats(k_pred, k_gt)
    res["counts_api"] = count_stats(k_api, k_gt)
    res["length_hist"] = length_hist([e[1] - e[0] for v in vids for e in uemr[v]],
                                     [e[1] - e[0] for v in vids for e in gts[v]])

    # d. per-video analysis (final segments, config uemr)
    gt_rows, seg_rows, video_rows = [], [], []
    uni = methods["uniform_K_pred"]
    for v in vids:
        evs = pred[v]["events"]
        g_rows, s_rows = analyze_video(gt[v], evs, TIOUS)
        for r in g_rows:
            pi = r["match@0.5"]
            bi = r["best_seg"]
            r.update(video_id=v, split=pred[v]["split"],
                     best_seg_ts=evs[bi]["t_s"] if bi >= 0 else None, best_seg_te=evs[bi]["t_e"] if bi >= 0 else None,
                     pred_caption=evs[pi]["caption"] if pi >= 0 else None,
                     pred_caption_oracle=evs[pi].get("caption_oracle") if pi >= 0 else None)
        for r in s_rows:
            r.update(video_id=v, split=pred[v]["split"])
        gt_rows += g_rows
        seg_rows += s_rows
        p05 = prf({v: uemr[v]}, {v: gts[v]}, 0.5)
        video_rows.append({"video_id": v, "split": pred[v]["split"], "duration": dur[v], "n": raw[v]["n"],
                           "K_GT": k_gt[v], "K_pred": k_pred[v], "K_api": k_api[v],
                           "P@0.5": p05["precision"], "R@0.5": p05["recall"], "F1@0.5": p05["f1"],
                           "mean_tIoU": mean_best_tiou({v: uemr[v]}, {v: gts[v]}),
                           "R@0.5_uniform_K_pred": prf({v: uni[v]}, {v: gts[v]}, 0.5)["recall"],
                           "n_missed": sum(r["missed"] for r in g_rows), "n_split_gt": sum(r["split"] for r in g_rows),
                           "n_fp_no_overlap": sum(r["fp_no_overlap"] for r in s_rows),
                           "n_merge_seg": sum(r["merge"] for r in s_rows)})

    # c. semantics
    if judge is not None and gt_rows:
        res["semantic"] = semantic_rows(gt_rows, vids, gt, pred, judge)

    res["errors"] = {
        "n_gt": len(gt_rows), "pct_gt_missed": 100 * float(np.mean([r["missed"] for r in gt_rows])) if gt_rows else 0.0,
        "pct_gt_split": 100 * float(np.mean([r["split"] for r in gt_rows])) if gt_rows else 0.0,
        "pct_gt_in_merged_seg": 100 * float(np.mean([r["in_merged_seg"] for r in gt_rows])) if gt_rows else 0.0,
        "n_seg": len(seg_rows),
        "pct_seg_fp_no_overlap": 100 * float(np.mean([r["fp_no_overlap"] for r in seg_rows])) if seg_rows else 0.0,
        "pct_seg_unmatched@0.5": 100 * float(np.mean([r["unmatched@0.5"] for r in seg_rows])) if seg_rows else 0.0,
        "pct_seg_merge": 100 * float(np.mean([r["merge"] for r in seg_rows])) if seg_rows else 0.0,
    }
    res["boundary"] = {"left": offset_stats(r["left_offset"] for r in gt_rows),
                       "right": offset_stats(r["right_offset"] for r in gt_rows)}
    res["duration_bins"] = duration_rows(gt_rows)
    return {"metrics": res, "gt_rows": gt_rows, "seg_rows": seg_rows, "video_rows": video_rows}


def semantic_rows(gt_rows: list[dict], vids, gt, pred, judge) -> list[dict]:
    """sem-Recall@tIoU,tau = share of GT matched at tIoU >= t whose caption agrees (judge cos >= tau)."""
    n_gt = len(gt_rows)
    out = []
    for t in TIOUS:
        for kind, key in (("retrieved (8218 train captions)", "caption"), ("ORACLE (video's GT captions)", "caption_oracle")):
            a, b = [], []
            for r in gt_rows:
                pi = r[f"match@{t}"]
                if pi < 0:
                    continue
                c = pred[r["video_id"]]["events"][pi].get(key)
                if c is None:
                    continue
                a.append(r["caption"]); b.append(c)
            if key == "caption_oracle" and not b:
                continue
            cos = judge.cos(a, b)
            if t == 0.5:   # keep the per-GT cosine of the 0.5 pairs for per_gt_event.csv
                it = iter(cos.tolist())
                for r in gt_rows:
                    pi = r["match@0.5"]
                    if pi >= 0 and pred[r["video_id"]]["events"][pi].get(key) is not None:
                        r["judge_cos" if key == "caption" else "judge_cos_ORACLE"] = next(it)
            row = {"tIoU": t, "captions": kind, "n_matched": len(a), "loc_recall": len(a) / max(n_gt, 1),
                   "mean_cos": float(cos.mean()) if len(cos) else None}
            for tau in JUDGE_TAUS:
                hit = int((cos >= tau).sum())
                row[f"sem_recall@tau{tau}"] = hit / max(n_gt, 1)
                row[f"acc_given_match@tau{tau}"] = hit / max(len(a), 1)
            out.append(row)
    return out


def duration_rows(gt_rows: list[dict]) -> list[dict]:
    rows = []
    for name, _, _ in DURATION_BINS:
        sub = [r for r in gt_rows if r["dur_bin"] == name]
        if not sub:
            rows.append({"bin": name, "n_gt": 0}); continue
        rows.append({"bin": name, "n_gt": len(sub), "pct_gt": 100 * len(sub) / len(gt_rows),
                     "R@0.5": float(np.mean([r["match@0.5"] >= 0 for r in sub])),
                     "R@0.3": float(np.mean([r["match@0.3"] >= 0 for r in sub])),
                     "mean_best_tIoU": float(np.mean([r["best_tiou"] for r in sub])),
                     "pct_missed": 100 * float(np.mean([r["missed"] for r in sub])),
                     "pct_split": 100 * float(np.mean([r["split"] for r in sub])),
                     "pct_in_merged_seg": 100 * float(np.mean([r["in_merged_seg"] for r in sub])),
                     "mean_abs_left": offset_stats(r["left_offset"] for r in sub)["mean_abs"],
                     "mean_abs_right": offset_stats(r["right_offset"] for r in sub)["mean_abs"]})
    return rows


# --------------------------------------------------------------------------- #
# Everything                                                                  #
# --------------------------------------------------------------------------- #

def evaluate_all(cfg: Config, gt: dict, raw: dict, pred: dict, pred_api: dict, judge=None,
                 theta_info: dict | None = None, sweep: list[dict] | None = None, extra: dict | None = None) -> dict:
    """Evaluate every split present and write all result files under ``cfg.out_dir``."""
    t0 = time.time()
    vids = [v for v in pred if v in gt and v in raw and v in pred_api]
    splits = {v: pred[v]["split"] for v in vids}
    results = {"caveats": CAVEATS, "tag": cfg.tag, "smoke": cfg.smoke, "feature_source": cfg.source,
               "checkpoint": cfg.checkpoint, "postprocess": {"uemr": cfg.postprocess_params(
                   theta_info["theta"] if theta_info else None), "api": cfg.api_params()},
               "theta": theta_info, "judge": getattr(judge, "name", None), "splits": {}, **(extra or {})}
    all_gt, all_seg, all_video = [], [], []
    tables = {k: [] for k in ("ar_at_n", "map", "comparison", "counts", "length_hist", "semantic",
                              "duration_bins", "errors", "boundary")}
    for split, sv in split_videos(vids, splits).items():
        if not sv:
            continue
        out = evaluate_split(sv, gt, raw, pred, pred_api, judge)
        m = out["metrics"]
        results["splits"][split] = dict(m, note=SPLIT_NOTE[split])
        all_gt += out["gt_rows"]; all_seg += out["seg_rows"]; all_video += out["video_rows"]
        tag = {"split": split, "note": SPLIT_NOTE[split]}
        tables["ar_at_n"] += [dict(tag, **r) for r in m["AR@N_raw"]]
        for name in ("mAP_raw", "mAP_uemr", "mAP_api"):
            tables["map"].append(dict(tag, set=name[4:], mAP=m[name]["mAP"],
                                      **{f"AP@{t}": a for t, a in m[name]["AP"].items()}))
        tables["comparison"] += [dict(tag, method=k, **v) for k, v in m["comparison"].items()]
        tables["counts"] += [dict(tag, config="uemr", **m["counts_uemr"]), dict(tag, config="api", **m["counts_api"])]
        tables["length_hist"] += [dict(tag, **r) for r in m["length_hist"]]
        tables["semantic"] += [dict(tag, **r) for r in m.get("semantic", [])]
        tables["duration_bins"] += [dict(tag, **r) for r in m["duration_bins"]]
        tables["errors"].append(dict(tag, **m["errors"]))
        tables["boundary"] += [dict(tag, side=side, **st) for side, st in m["boundary"].items()]
    if sweep:
        tables["theta_sweep"] = sweep
    out_dir, adir = cfg.out_dir, cfg.analysis_dir
    write_json(out_dir / "results.json", results)
    write_csv(out_dir / "per_video.csv", all_video)
    write_csv(out_dir / "per_gt_event.csv", all_gt)
    write_csv(adir / "segments.csv", all_seg)
    for name, rows in tables.items():
        if rows:
            write_csv(adir / f"{name}.csv", rows)
    print(f"[evaluate] {len(vids)} videos, {len(all_gt)} GT -> {out_dir} ({time.time() - t0:.1f} s)")
    return results


def print_summary(results: dict) -> None:
    """The headline numbers, val first."""
    print("\n".join(f"[!] {c}" for c in results["caveats"]))
    for split, m in results["splits"].items():
        print(f"\n=== {m['note']} | {m['n_videos']} videos, {m['n_gt']} GT ===")
        ar = {r["N"]: r for r in m["AR@N_raw"]}
        print("AR@N raw (R@0.3/0.5/0.7): " + " | ".join(
            f"N={n}: {100 * ar[n]['R@0.3']:.1f}/{100 * ar[n]['R@0.5']:.1f}/{100 * ar[n]['R@0.7']:.1f}" for n in ar))
        print(f"mIoU raw@K_GT {100 * m['mIoU_raw@K_GT']:.1f} | mAP raw {100 * m['mAP_raw']['mAP']:.1f} "
              f"| mAP uemr {100 * m['mAP_uemr']['mAP']:.1f} | mAP api {100 * m['mAP_api']['mAP']:.1f}")
        print(f"{'method':16s} {'seg/v':>6s} {'P@.5':>6s} {'R@.5':>6s} {'F1@.5':>6s} {'F1@.3':>6s} {'F1@.7':>6s} {'mtIoU':>6s}")
        for k, c in m["comparison"].items():
            print(f"{k:16s} {c['seg_per_video']:6.2f} {100 * c['P@0.5']:6.1f} {100 * c['R@0.5']:6.1f} "
                  f"{100 * c['F1@0.5']:6.1f} {100 * c['F1@0.3']:6.1f} {100 * c['F1@0.7']:6.1f} {100 * c['mean_tIoU']:6.1f}")
        cu = m["counts_uemr"]
        print(f"K_pred {cu['mean_K_pred']:.2f} vs K_GT {cu['mean_K_GT']:.2f} | fewer {cu['pct_fewer']:.0f}% "
              f"equal {cu['pct_equal']:.0f}% more {cu['pct_more']:.0f}%")
        for r in m.get("semantic", []):
            if r["tIoU"] == 0.5:
                print(f"sem@0.5 {r['captions']:34s} tau.7 {100 * r['sem_recall@tau0.7']:.1f} "
                      f"tau.8 {100 * r['sem_recall@tau0.8']:.1f} tau.9 {100 * r['sem_recall@tau0.9']:.1f} "
                      f"(loc {100 * r['loc_recall']:.1f})")
