"""UEMR chốt file 03 §4–9: the locked method (``configs/uemr_final.json``) on a zero-shot benchmark. Nothing is chosen
here: θ, ``L_min``, ``L_fill``, β* and the main method are read from the config; the two label-free rules (L of R2,
μ / σ of the z-scores) are recomputed on the benchmark as the config says. Same code paths as the dev run
(``final_dev``): ``partitions.r6``, ``fill_gaps``, ``pick_chunk``, ``row_indexes``, ``score.run``, ``score.boot``.

Called from ``notebooks/uemr/uemr_final_benchmark.ipynb``.
"""

from __future__ import annotations

import os
from collections import Counter

import numpy as np

from omni_retrieval.bo_sung import fill_gaps
from omni_retrieval.partitions import uniform
from omni_retrieval.partitions_uemr import _round, best_tiou, coverage, uni_chunk

from . import bo_sung as bs
from . import partitions as hp
from . import score as sc
from .final_dev import dump, pooled, row_indexes

NAMES = ("r0", "r2", "r3", "r6", "r8", "r7")
MAIN_COLUMN = {"UEMR-Full": "U", "UEMR-Omni": "b"}


def uniav_annotations(durations: dict, step_id: int) -> dict:
    """The benchmark videos in the layout of HT-Step ``annotations.json`` so that the unchanged ``HTStepDataset`` reads
    their features (split ``val_seen``). The one annotation is a placeholder the dataset needs to keep the video: in the
    main mode (``uniav_htstep.predict(mode='pred')``) the model replaces ``cand`` by the steps of its own top-1 article,
    so the placeholder step reaches no head (checked in the notebook by changing it)."""
    return {v: {"subset": "val_seen", "duration": float(d), "activity": "benchmark",
                "annotations": [{"segment": [0.0, 1.0], "id": int(step_id)}]} for v, d in durations.items()}


def build_rows(props: dict, dur: dict, gt: dict, cfg: dict) -> tuple[dict, dict]:
    """``({name: {video: segments}}, info)`` with the numbers of the config. A video without any proposal above θ has no
    R6 and no R3 segment (it cannot be retrieved by those rows); its R8 is the fill windows alone."""
    p, pp = cfg["params"], cfg["params"]["r6_postprocess"]
    vids = sorted(dur)
    r6 = {v: hp.r6(props[v], dur[v], p["theta"], pp["k_max"], pp["d_min"], pp["merge_tiou"]) for v in vids}
    r8 = {v: _round(fill_gaps(r6[v], dur[v], p["l_min"], p["l_fill"])[0], dur[v]) for v in vids}
    chunk, n_r2 = hp.pick_chunk(dur, sum(len(s) for s in r8.values()), pp["d_min"])
    rows = {"r0": {v: _round([[0.0, dur[v]]], dur[v]) for v in vids},
            "r2": {v: _round(uni_chunk(dur[v], chunk, pp["d_min"]), dur[v]) for v in vids},
            "r3": {v: _round(uniform(dur[v], len(r6[v])), dur[v]) for v in vids},
            "r6": r6, "r7": {v: gt[v] for v in vids}, "r8": r8}
    return rows, {"theta": p["theta"], "l_min": p["l_min"], "l_fill": p["l_fill"], "chunk_sec": chunk,
                  "clips": {n: sum(len(s) for s in part.values()) for n, part in rows.items()},
                  "videos_without_r6": [v for v in vids if not r6[v]]}


def quality(rows: dict, gt: dict, dur: dict, article_top1: dict, article_names=None) -> dict:
    """§4, descriptive only: how the R6 cuts sit on the GT steps, and which HT-Step articles the model guesses."""
    vids = sorted(dur)
    bt = best_tiou(rows["r6"], {v: gt[v] for v in vids})
    arts = Counter(article_top1[v] for v in vids)
    name = (lambda a: str(article_names[a])) if article_names is not None else str
    return {**{f"recall tIoU {t}": 100 * float((bt >= t).mean()) for t in (0.3, 0.5, 0.7)},
            "% bước bị bỏ sót (tIoU < 0.3)": 100 * float((bt < 0.3).mean()),
            "độ phủ R6 %": 100 * float(np.mean([coverage(rows["r6"][v], dur[v]) for v in vids])),
            "độ phủ R8 %": 100 * float(np.mean([coverage(rows["r8"][v], dur[v]) for v in vids])),
            "K_pred / video": float(np.mean([len(rows["r6"][v]) for v in vids])),
            "K_GT / video": float(np.mean([len(gt[v]) for v in vids])),
            "video không có đoạn R6": sum(not rows["r6"][v] for v in vids),
            "article đoán khác nhau": len(arts), "article đoán nhiều nhất": [[name(a), n] for a, n in arts.most_common(10)]}


def criteria(pairs: dict, method: str) -> dict:
    """§8 from the bootstrap pairs (``{name: boot dict}``), multi-positive. K1–K5 pass when the lower CI bound > 0."""
    ok = lambda k: bool(pairs[k]["ci_low"] > 0)
    out = {"K1": {"cặp": "UEMR R8 - R0 (b)", **pairs["UEMR R8 - R0 (b)"], "đạt": ok("UEMR R8 - R0 (b)")},
           "K2": {"cặp": "UEMR R8 - R3 cùng cột", **pairs["UEMR R8 - R3"], "đạt": ok("UEMR R8 - R3")},
           "K3": {"cặp": "VCMR R6 (b) - R3 (b)", **pairs["VCMR R6 - R3 (b)"], "đạt": ok("VCMR R6 - R3 (b)")},
           "K4": {"cặp": "UEMR-Lite R8 - R0 (b)", **pairs["(S) R8 - R0 (b)"], "đạt": ok("(S) R8 - R0 (b)")},
           "K5": {"cặp": "(U) - (b) R8", **pairs["(U) - (b) R8"], "đạt": ok("(U) - (b) R8")},
           "K6 (chỉ báo)": {k: pairs[k] for k in ("UEMR R8 - R2", "C1 R8 - R6 (b)", "R7 - UEMR R8", "pt - best R0 (b)", "pt - best R8 (b)")
                            if k in pairs}}
    k1, k2 = out["K1"]["đạt"], out["K2"]["đạt"]
    out["kết luận"] = ("UEMR có tác dụng zero-shot trên <bộ>, cả so với một vector lẫn so với cắt đều" if k1 and k2 else
                       "UEMR hơn một vector trên <bộ>, nhưng lợi thế so với cắt đều không phân biệt được" if k1 else
                       "Trên <bộ>, UEMR không hơn một vector Omni cho mỗi video; ghi là giới hạn đã đo, kèm chất lượng cắt")
    out["UEMR"] = method
    return out


def score(caps: list[dict], text: dict, parts: dict, omni: dict, student_dir: str, cfg: dict, out: str, pt: dict | None = None) -> dict:
    """§9, once: every row x column x mode, the pairs with CI, the criteria. ``omni``: ``{seg_key: unit vector}`` of Omni
    ``best``; ``pt``: ``{'text': load_text(...), 'omni': {...}}`` of the pretrained adapter (rows R0, R8). Writes
    ``metrics.json, ranks_*.json, pairs.json, criteria.json`` to ``out`` and returns the criteria."""
    if [str(x) for x in text["caption_id"]] != [c["caption_id"] for c in caps]:
        raise SystemExit("CHECK text.npz cùng thứ tự câu hỏi: FAIL")
    t, vids, ev = text["emb"], text["video_id"], cfg["eval"]["bootstrap"]
    beta, method = float(cfg["params"]["beta_star"]), cfg["main_method"]
    main = MAIN_COLUMN[method]
    boot = lambda x, y: sc.boot(x, y, vids, ev["n_boot"], ev["seed"])
    st = lambda n: os.path.join(student_dir, f"student_{n}.npz")
    IDX, ST, MISS = {}, {}, {}
    for n in NAMES:
        IDX[n], ST[n], MISS[n] = row_indexes(parts[n], omni, st(n), n)
    total = {n: sum(len(x) for x in parts[n].values()) for n in NAMES}
    fail_pct = 100 * sum(MISS.values()) / sum(total.values())
    print("clip không có vector Omni:", MISS, f"({fail_pct:.2f} %)")
    if fail_pct > 2:
        raise SystemExit(f"CHECK clip thiếu vector Omni <= 2 %: FAIL ({fail_pct:.2f} %)")

    R = {}
    for n in NAMES:
        sb, ss = t @ IDX[n].emb.T, t @ ST[n].T
        R[("b", n)] = sc.run(IDX[n], text, caps, sim=sb)
        R[("S", n)] = sc.run(IDX[n], text, caps, sim=ss)
        R[("U", n)] = sc.run(IDX[n], text, caps, sim=sb + beta * ss)
        if n == "r8":      # z-scores recomputed on this gallery, no label (rule of the config)
            r6_keys = {sc.hm.row_key(v, x, y) for v, x, y in zip(IDX["r6"].seg_video, IDX["r6"].ts, IDX["r6"].te)}
            fill = np.array([sc.hm.row_key(v, x, y) not in r6_keys for v, x, y in zip(IDX[n].seg_video, IDX[n].ts, IDX[n].te)])
            zb, zs = bs.zstats(sb), bs.zstats(ss)
            R[("C1", n)] = sc.run(IDX[n], text, caps, sim=bs.c1z(sb, ss, fill, zb, zs))
        del sb, ss
    p1b, p1s = IDX["r6"].mean_pooled("r1"), pooled(IDX["r6"], ST["r6"])
    R[("b", "r1")] = sc.run(p1b, text, caps)
    R[("S", "r1")] = sc.run(p1s, text, caps)
    R[("U", "r1")] = sc.run(p1b, text, caps, sim=t @ p1b.emb.T + beta * (t @ p1s.emb.T))
    index_of = {**{(c, n): IDX[n] for c in ("b", "S", "U", "C1") for n in NAMES}, **{(c, "r1"): p1b for c in ("b", "S", "U")}}
    if pt is not None:
        if [str(x) for x in pt["text"]["caption_id"]] != [c["caption_id"] for c in caps]:
            raise SystemExit("CHECK text.npz (pt) cùng thứ tự câu hỏi: FAIL")
        for n in ("r0", "r8"):
            idx, _, miss = row_indexes(parts[n], pt["omni"], st(n), n)
            MISS[f"pt {n}"] = miss
            R[("b_pt", n)] = sc.run(idx, pt["text"], caps)
            index_of[("b_pt", n)] = idx

    os.makedirs(out, exist_ok=True)
    everyone = np.arange(len(caps))
    table = []
    for (c, n), res in R.items():
        met = sc.row_metrics(res, everyone, index_of[(c, n)], pooled=n == "r1")
        table += [{"dòng": n.upper(), "cột": f"U{beta:g}" if c == "U" else c, "mode": m,
                   **{k: v for k, v in met[m].items() if k != "R@1 Full Match"}} for m in sc.MODES]
        dump(os.path.join(out, f"ranks_{c}_{n.upper()}.json"),
             {**{m: res[m]["ranks"].tolist() for m in sc.MODES}, **{f"vcmr_{m}": res[m]["vcmr"].astype(int).tolist() for m in sc.MODES}})
    rk = lambda c, n, m: R[(c, n)][m]["ranks"]
    pairs = {}
    for m in sc.MODES:
        P = {"UEMR R8 - R0 (b)": boot(rk(main, "r8", m), rk("b", "r0", m)), "UEMR R8 - R3": boot(rk(main, "r8", m), rk(main, "r3", m)),
             "VCMR R6 - R3 (b)": boot(sc.vcmr_rank(R[("b", "r6")][m]["vcmr"]), sc.vcmr_rank(R[("b", "r3")][m]["vcmr"])),
             "(S) R8 - R0 (b)": boot(rk("S", "r8", m), rk("b", "r0", m)), "(U) - (b) R8": boot(rk("U", "r8", m), rk("b", "r8", m)),
             "UEMR R8 - R2": boot(rk(main, "r8", m), rk(main, "r2", m)), "C1 R8 - R6 (b)": boot(rk("C1", "r8", m), rk("b", "r6", m)),
             "R7 - UEMR R8": boot(rk(main, "r7", m), rk(main, "r8", m))}
        for x, y in (("r8", "r0"), ("r6", "r0"), ("r8", "r3"), ("r6", "r3"), ("r8", "r2"), ("r6", "r1"), ("r7", "r8")):
            P[f"{x.upper()} - {y.upper()} (b)"] = boot(rk("b", x, m), rk("b", y, m))
        if pt is not None:
            for n in ("r0", "r8"):
                P[f"pt - best {n.upper()} (b)"] = boot(rk("b_pt", n, m), rk("b", n, m))
        pairs[m] = P
    crit = criteria(pairs["multi"], method)
    dump(os.path.join(out, "metrics.json"), {"table": table, "beta": beta, "main_method": method, "zscore": {"b": zb, "S": zs},
                                             "missing_omni": MISS, "clips": total, "queries": len(caps), "videos": len(parts["r0"])})
    dump(os.path.join(out, "pairs.json"), pairs)
    dump(os.path.join(out, "criteria.json"), crit)
    return crit
