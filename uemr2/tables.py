"""Tables and gates of UEMR 2 (docs/uemr2/UEMR2_AGENT_SO_BO.md §4.2-4.4, §5.3-5.4), as fixed in
``configs/uemr2_prereg.json``. ``dev_reference`` (fixed configuration, dev, leaked) and ``val_tables`` (2-fold CV
on val, out of fold) return pandas DataFrames; nothing here reads or writes files."""

from __future__ import annotations

import numpy as np
import pandas as pd

from omni_retrieval.evaluate import paired_bootstrap

from . import experiments as X
from .score import metrics

A_PARTS = ("global", "uni_chunk", "uni_M", "uniav_pred", "gt", "uni_M_gt", "uniav_shift")
B_PARTS = ("global", "uni_chunk", "uni_M", "uniav_pred", "gt")
NOISE_PARTS = ("gt_noise_03", "gt_noise_05", "gt_noise_07", "gt_noise_09", "gt", "uni_M_gt")
LABEL = {"global": "R0", "uni_chunk": "R2", "uni_M": "R3", "uniav_pred": "R6", "gt": "R7", "uni_M_gt": "uni_M_gt",
         "uniav_shift": "uniav_shift", "gt_noise_03": "noise 0.3", "gt_noise_05": "noise 0.5", "gt_noise_07": "noise 0.7",
         "gt_noise_09": "noise 0.9"}
LEN_BINS = (("<10s", 0, 10), ("10-40s", 10, 40), (">=40s", 40, np.inf))
DUP_BINS = (("0", 0, 1), ("1-2", 1, 3), ("3-9", 3, 10), ("10-29", 10, 30), (">=30", 30, np.inf))
GATE_GROUPS = (">=40s", "3-9", "10-29", ">=30")


def _ci(b):
    return f"[{b['ci_low']:+.2f}, {b['ci_high']:+.2f}]"


def _row(exp, part, gallery, rk, hit, base, groups, **extra):
    r = {"thí nghiệm": exp, "cách cắt": LABEL.get(part, part), "kho": gallery, **metrics(rk, hit), **extra}
    if base is not None:
        b = paired_bootstrap(rk, base, 1, 1000, 0, "video", groups)
        r.update({"Δ R@1 so với (b)": b["diff"], "CI 95%": _ci(b), "cận dưới": b["ci_low"]})
    return r


def dev_reference(world, z) -> pd.DataFrame:
    """Fixed configuration β = 1, W = 8, γ = 0 on dev (773 captions, gallery = 104 dev videos). Leaked: reference."""
    rows, g = [], world["text"]["dev"]["video_id"]
    base = {p: X.score_part(world, z, "dev", "dev", p) for p in A_PARTS}
    for p in A_PARTS:
        rows.append(_row("(b) Omni ft", p, "dev104", *base[p], None, g))
        rows.append(_row("A-d", p, "dev104", *X.score_part(world, z, "dev", "dev", p, "A", alone=True), base[p][0], g))
        rows.append(_row("A-e β=1", p, "dev104", *X.score_part(world, z, "dev", "dev", p, "A", 1.0), base[p][0], g))
        if p in B_PARTS:
            rows.append(_row("B-d", p, "dev104", *X.score_part(world, z, "dev", "dev", p, "B", alone=True), base[p][0], g))
            rows.append(_row("B-e β=1", p, "dev104", *X.score_part(world, z, "dev", "dev", p, "B", 1.0), base[p][0], g))
    for src in ("A", "B"):
        for kind in ("gap", "dense"):
            pre = X.e3_scores(world, z, "dev", "dev", src, 8, kind)
            rows.append(_row(f"{'E3' if src == 'A' else 'B-E3'} {kind}_8 γ=0", "uniav_pred", "dev104",
                             *X.e3_ranks(world, "dev", "dev", pre, 0.0), base["uniav_pred"][0], g,
                             **{"vector rẻ/video": len(pre[2]["video_id"]) / len(world["videos"]["dev"])}))
    for src in ("A", "B"):
        rows.append({"thí nghiệm": f"{src}-0 (text -> clip GT dev)", "cách cắt": "R7 clip", "kho": "773 clip",
                     **X.text_to_clip(world, "dev", src)})
    rows.append({"thí nghiệm": "Omni ft (text -> clip GT dev)", "cách cắt": "R7 clip", "kho": "773 clip",
                 **X.text_to_clip(world, "dev", "omni")})
    out = pd.DataFrame(rows)
    out.insert(0, "ghi chú", "dev: rò rỉ, tham khảo")
    return out


def _groups(name, values, bins, rk_b, rk_x, vid, part):
    rows = []
    for label, lo, hi in bins:
        m = (values >= lo) & (values < hi)
        if m.any():
            b = paired_bootstrap(rk_x[m], rk_b[m], 1, 1000, 0, "video", vid[m])
            rows.append({"cách cắt": LABEL[part], "theo": name, "nhóm": label, "n": int(m.sum()),
                         "R@1 (b)": 100 * float((rk_b[m] == 1).mean()), "R@1 mới": 100 * float((rk_x[m] == 1).mean()),
                         "Δ": b["diff"], "CI 95%": _ci(b), "cận dưới": b["ci_low"]})
    return rows


def val_tables(world, z, folds, gallery="val") -> dict:
    """Every val table of §4.2 / §5.3 on one gallery ('val' = 394, 'all' = 498), out of fold, plus the gates."""
    T = world["text"]["val"]
    vid = np.asarray(T["video_id"])
    fq = X.fold_of_queries(world, folds)
    kho = {"val": "val394", "all": "all498"}[gallery]
    parts = sorted(set(A_PARTS) | set(NOISE_PARTS), key=lambda p: (A_PARTS + NOISE_PARTS).index(p))
    base = {p: X.score_part(world, z, "val", gallery, p) for p in parts}
    main, picks = [], {}
    for p in A_PARTS:
        main.append(_row("(b) Omni ft", p, kho, *base[p], None, vid))
    for src, plist in (("A", A_PARTS), ("B", B_PARTS)):
        for p in plist:
            main.append(_row(f"{src}-d", p, kho, *X.score_part(world, z, "val", gallery, p, src, alone=True), base[p][0], vid))
        chosen, oofr = X.cv_beta(world, z, gallery, fq, src, parts if src == "A" else plist)
        picks[f"{src}-e β (chọn trên F1, F2)"] = chosen
        for p in plist:
            main.append(_row(f"{src}-e", p, kho, *oofr[p], base[p][0], vid, **{"β F1/F2": str(chosen)}))
        for p in plist:                                     # fixed configuration, reference only
            main.append(_row(f"{src}-e β=1 (cố định, tham khảo)", p, kho,
                             *X.score_part(world, z, "val", gallery, p, src, 1.0), base[p][0], vid))
        if src == "A":
            oof_a = oofr
        else:
            oof_b = oofr
    chosen_ab, oof_ab = X.cv_beta_ab(world, z, gallery, fq, ("uniav_pred",))
    picks["A+B (βA, βB) F1/F2"] = chosen_ab
    main.append(_row("A-e + B-e", "uniav_pred", kho, *oof_ab["uniav_pred"], base["uniav_pred"][0], vid,
                     **{"(βA, βB) F1/F2": str(chosen_ab)}))
    # E3
    e3rows, e3oof = [], {}
    miss = X.miss_labels(world)
    for src in ("A", "B"):
        for kind in ("gap", "dense"):
            ch, rk, hit, nwin = X.cv_e3(world, z, gallery, fq, src, kind)
            name = f"{'E3' if src == 'A' else 'B-E3'} R6 ∪ {kind}_W"
            picks[f"{name} (W, γ) F1/F2"] = ch
            e3oof[(src, kind)] = rk
            e3rows.append(_row(name, "uniav_pred", kho, rk, hit, base["uniav_pred"][0], vid,
                               **{"(W, γ) F1/F2": str(ch), "vector rẻ/video": {str(k): round(v, 1) for k, v in nwin.items()}}))
            pre = X.e3_scores(world, z, "val", gallery, src, 8, kind)
            e3rows.append(_row(f"{name} W=8 γ=0 (cố định, tham khảo)", "uniav_pred", kho,
                               *X.e3_ranks(world, "val", gallery, pre, 0.0), base["uniav_pred"][0], vid))
    # groups (A-g, B-g, E3-g) on the main tables' out-of-fold ranks
    glen = np.asarray(T["gt_te"]) - np.asarray(T["gt_ts"])
    dup = np.nan_to_num(np.asarray(T.get("dup", np.full(len(vid), np.nan)), np.float64), nan=-1)
    grows = []
    for src, oo in (("A", oof_a), ("B", oof_b)):
        for p in ("uniav_pred", "gt"):
            for name, val, bins in (("độ dài GT", glen, LEN_BINS), ("bước gần trùng", dup, DUP_BINS)):
                for r in _groups(name, val, bins, base[p][0], oo[p][0], vid, p):
                    grows.append({"thí nghiệm": f"{src}-g", **r})
    for (src, kind), rk in e3oof.items():
        for r in _groups("bỏ sót (R6 cover < 0.3)", miss.astype(float), (("không bỏ sót", 0, 0.5), ("bỏ sót", 0.5, 2)),
                         base["uniav_pred"][0], rk, vid, "uniav_pred"):
            grows.append({"thí nghiệm": f"{'E3' if src == 'A' else 'B-E3'}-g {kind}", **r})
    # A-r: R2 - R6 per column
    arows = []
    for col, r2, r6 in (("(b)", base["uni_chunk"][0], base["uniav_pred"][0]),
                        ("(d) A", X.score_part(world, z, "val", gallery, "uni_chunk", "A", alone=True)[0],
                         X.score_part(world, z, "val", gallery, "uniav_pred", "A", alone=True)[0]),
                        ("(e) A", oof_a["uni_chunk"][0], oof_a["uniav_pred"][0])):
        b = paired_bootstrap(r2, r6, 1, 1000, 0, "video", vid)
        arows.append({"cột": col, "kho": kho, "R2 - R6": b["diff"], "CI 95%": _ci(b)})
    # A-n: Exp 3 line (e)
    nrows = [{"cách cắt": LABEL[p], "kho": kho, "R@1 (b)": 100 * float((base[p][0] == 1).mean()),
              "R@1 (e)": 100 * float((oof_a[p][0] == 1).mean())} for p in NOISE_PARTS]
    out = {"main": pd.DataFrame(main), "e3": pd.DataFrame(e3rows), "groups": pd.DataFrame(grows),
           "r2_r6": pd.DataFrame(arows), "noise": pd.DataFrame(nrows), "picks": picks, "miss_count": int(miss.sum())}
    if gallery == "val":
        out["text_to_clip"] = pd.DataFrame([{"thí nghiệm": f"{s}-0", **X.text_to_clip(world, "val", s)}
                                            for s in ("omni", "A", "B")])
    out["gates"] = gates(out)
    return out


def gates(t) -> dict:
    """§4.4, §5.4 on the out-of-fold tables of one gallery."""
    m, e3t, g = t["main"], t["e3"], t["groups"]

    def d(exp, part="R6"):
        r = m[(m["thí nghiệm"] == exp) & (m["cách cắt"] == part)]
        return float(r["Δ R@1 so với (b)"].iloc[0])

    def grp(exp):
        r = g[(g["thí nghiệm"] == exp) & (g["nhóm"].isin(GATE_GROUPS))]
        return r[r["Δ"] >= 2.0][["cách cắt", "nhóm", "Δ"]].to_dict("records")

    def e3d(name):
        return float(e3t[e3t["thí nghiệm"] == name]["Δ R@1 so với (b)"].iloc[0])

    a_beta = t["picks"]["A-e β (chọn trên F1, F2)"]
    a_e, b_e = d("A-e"), d("B-e")
    e3_gap, be3_gap = e3d("E3 R6 ∪ gap_W"), e3d("B-E3 R6 ∪ gap_W")
    a2_closed = a_beta == [0.0, 0.0] or (a_e < 1.0 and not grp("A-g"))
    b_open = b_e >= 1.0 or bool(grp("B-g")) or be3_gap >= 2.0
    return {"A2": "đóng" if a2_closed else "mở", "A-e R6 Δ": a_e, "A-g nhóm >= +2": grp("A-g"), "β A": a_beta,
            "E3": "mở" if e3_gap >= 2.0 else "đóng", "E3 gap Δ": e3_gap,
            "B": "mở" if b_open else "đóng", "B-e R6 Δ": b_e, "B-g nhóm >= +2": grp("B-g"), "B-E3 gap Δ": be3_gap,
            "pt": "encode" if (b_e >= 1.0 or be3_gap >= 2.0) else "không chạy (ft không có tín hiệu)"}
