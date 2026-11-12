"""UEMR chốt file 02 §2–4 on HT-Step dev (local CPU): the rows of the new UniAV and the choice of ``L_min``, ``L_fill`` by the
cheap column (S) (no Omni 7B), before the clips of the new rows are encoded on Colab.

    python -m uemr_htstep.final_dev rows --uniav-raw <new uniav>/uniav_raw.jsonl --val-manifest val_omni_htstep.jsonl --out DIR/segments
    python -m uemr_htstep.run_uniav student ... --segments DIR/segments --names r0 r3 r6 r7 r8_3_8 ...
    python -m uemr_htstep.final_dev select --segments DIR/segments --student DIR/student --text <dev-1 db> --val-manifest ... \
        --tried results/uemr_final/dev/tried.jsonl
    python -m uemr_htstep.final_dev score --segments DIR/segments --student DIR/student --text <dev-1 db> --omni-db <dev-1 db> \
        --omni-store <store of the new clips> --val-manifest ... --uniav-checks results/uemr_final/uniav_best_checks.json \
        --out results/uemr_final/dev --config configs/uemr_final.json

Row names: ``r0, r3, r6, r7`` and ``r8_<L_min>_<L_fill>`` for the grid of §3; ``select`` adds ``r2`` (L by the clip-count rule) and
``r8`` (the chosen pair) and writes ``selected.json``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time

import numpy as np

from omni_retrieval.bo_sung import fill_gaps
from omni_retrieval.evaluate import load_text
from omni_retrieval.partitions import uniform
from omni_retrieval.partitions_uemr import _round, uni_chunk
from omni_retrieval.manifest import read_store, seg_key
from omni_retrieval.search import SegmentIndex

from . import bo_sung as bs
from . import data as hd
from . import metrics as hm
from . import partitions as hp
from . import score as sc

L_MIN, L_FILL = (3.0, 5.0), (8.0, 10.0, 15.0)    # file 02 §3
DEFAULT = (5.0, 10.0)                            # tie: the YC2 pair (or the closest one)


def r8_name(l_min: float, l_fill: float) -> str:
    return f"r8_{l_min:g}_{l_fill:g}"


def gallery(pr: dict, manifest: str):
    kho = pr["data"]["gallery"]
    caps = hd.queries(hd.read_jsonl(manifest), kho)
    return kho, {v: float(pr["data"]["durations"][v]) for v in kho}, caps


def build_rows(props: dict, dur: dict, gt: dict, pr: dict) -> tuple[dict, dict]:
    """``({name: {video: segments}}, info)``: θ by the rule of the dev runs on the predicted-article proposals."""
    p = pr["params"]["r6_postprocess"]
    a, b, s = pr["rules"]["theta"]["grid"]
    vids = sorted(dur)
    theta, k, _ = hp.pick_theta({v: props[v] for v in vids}, dur, pr["resolved"]["train_steps_per_video"], p["k_max"], p["d_min"],
                                p["merge_tiou"], np.round(np.arange(a, b, s), 6))
    r6 = {v: hp.r6(props[v], dur[v], theta, p["k_max"], p["d_min"], p["merge_tiou"]) for v in vids}
    empty = [v for v in vids if not r6[v]]
    if empty:
        raise ValueError(f"{len(empty)} videos have no R6 segment: {empty[:5]}")
    rows = {"r0": {v: _round([[0.0, dur[v]]], dur[v]) for v in vids},
            "r3": {v: _round(uniform(dur[v], len(r6[v])), dur[v]) for v in vids},
            "r6": r6, "r7": {v: gt[v] for v in vids}}
    for lm in L_MIN:
        for lf in L_FILL:
            rows[r8_name(lm, lf)] = {v: _round(fill_gaps(r6[v], dur[v], lm, lf)[0], dur[v]) for v in vids}
    return rows, {"theta": float(theta), "mean_k": float(k), "target_k": pr["resolved"]["train_steps_per_video"]}


def write_rows(rows: dict, out: str) -> dict:
    os.makedirs(out, exist_ok=True)
    sha = {}
    for name, part in rows.items():
        path = os.path.join(out, f"{name}.json")
        with open(path, "w", encoding="utf-8", newline="\n") as f:
            json.dump(part, f)
        sha[name] = hashlib.sha256(open(path, "rb").read()).hexdigest()
    return sha


def student_index(path: str, name: str) -> SegmentIndex:
    with np.load(path) as z:
        e = z["emb"].astype(np.float32)
        e /= np.maximum(np.linalg.norm(e, axis=1, keepdims=True), 1e-12)
        return SegmentIndex(e, z["video_id"], z["ts"], z["te"], name)


def r1_by_mode(index, text, caps) -> dict:
    return {m: 100 * float((r["ranks"] <= 1).mean()) for m, r in sc.run(index, text, caps).items()}


def pick(results: list[dict]) -> dict:
    """Highest R@1 multi of (S) on R8; tie -> the YC2 pair (5, 10), else the closest to it."""
    return min(results, key=lambda r: (-r["R@1 multi"], abs(r["l_min"] - DEFAULT[0]) + abs(r["l_fill"] - DEFAULT[1]), r["l_fill"]))


def cmd_rows(a, pr) -> None:
    kho, dur, caps = gallery(pr, a.val_manifest)
    raw = {json.loads(line)["video_id"]: json.loads(line) for line in open(a.uniav_raw, encoding="utf-8") if line.strip()}
    rows, info = build_rows({v: raw[v]["pred"] for v in kho}, dur, hd.gt_partition(caps), pr)
    sha = write_rows(rows, a.out)
    clips = {n: sum(len(s) for s in part.values()) for n, part in rows.items()}
    out = {**info, "uniav_raw_sha256": hashlib.sha256(open(a.uniav_raw, "rb").read()).hexdigest(), "files": sha, "clips": clips}
    json.dump(out, open(os.path.join(a.out, "rows.json"), "w", encoding="utf-8"), indent=1)
    print(json.dumps(out, indent=1))


def cmd_select(a, pr) -> None:
    kho, dur, caps = gallery(pr, a.val_manifest)
    text = load_text(a.text)
    if [str(x) for x in text["caption_id"]] != [c["caption_id"] for c in caps]:
        raise SystemExit("CHECK text.npz cùng thứ tự câu hỏi: FAIL")
    rows = json.load(open(os.path.join(a.segments, "rows.json"), encoding="utf-8"))
    os.makedirs(os.path.dirname(a.tried), exist_ok=True)
    results = []
    for lm in L_MIN:
        for lf in L_FILL:
            n = r8_name(lm, lf)
            r1 = r1_by_mode(student_index(os.path.join(a.student, f"student_{n}.npz"), n), text, caps)
            res = {"what": "R8 cột (S), chọn L_min / L_fill (file 02 §3)", "l_min": lm, "l_fill": lf, "R@1 multi": r1["multi"],
                   "R@1 strict": r1["strict"], "clips": rows["clips"][n], "theta": rows["theta"], "uniav_raw_sha256": rows["uniav_raw_sha256"],
                   "time": time.strftime("%Y-%m-%d %H:%M:%S")}
            results.append(res)
            with open(a.tried, "a", encoding="utf-8", newline="\n") as f:
                f.write(json.dumps(res, ensure_ascii=False) + "\n")
            print(json.dumps(res, ensure_ascii=False))
    best = pick(results)
    r8 = json.load(open(os.path.join(a.segments, f"{r8_name(best['l_min'], best['l_fill'])}.json"), encoding="utf-8"))
    d_min = pr["params"]["r6_postprocess"]["d_min"]
    chunk, n_r2 = hp.pick_chunk(dur, best["clips"], d_min)
    r2 = {v: _round(uni_chunk(dur[v], chunk, d_min), dur[v]) for v in sorted(dur)}
    sha = write_rows({"r8": r8, "r2": r2}, a.segments)
    sel = {"l_min": best["l_min"], "l_fill": best["l_fill"], "R@1 multi (S)": best["R@1 multi"], "clips_r8": best["clips"],
           "chunk_sec": chunk, "clips_r2": n_r2, "theta": rows["theta"], "files": sha, "grid": results,
           "rule": "R@1 multi cao nhất của (S) trên R8; hoà lấy (5, 10) hoặc cặp gần nó nhất; L của R2: tổng clip R2 gần nhất tổng clip R8"}
    json.dump(sel, open(os.path.join(a.segments, "selected.json"), "w", encoding="utf-8"), indent=1, ensure_ascii=False)
    with open(a.tried, "a", encoding="utf-8", newline="\n") as f:
        f.write(json.dumps({"what": "chọn (file 02 §3)", **{k: v for k, v in sel.items() if k != "grid"}, "time": time.strftime("%Y-%m-%d %H:%M:%S")},
                           ensure_ascii=False) + "\n")
    print(json.dumps({k: v for k, v in sel.items() if k != "grid"}, indent=1, ensure_ascii=False))


# --------------------------------------------------------------------------- #
# Scoring and the choices of file 02 §6-7                                      #
# --------------------------------------------------------------------------- #

BETAS = (0.0, 0.1, 0.2, 0.3, 0.5, 1.0)


def omni_vectors(dbs, stores) -> dict:
    """``{seg_key: fp32 unit vector}`` of the Omni ``best`` clips: dev-1 ``segments.npz`` folders and encoder stores."""
    out = {}
    for d in dbs:
        with np.load(os.path.join(d, "segments.npz")) as z:
            for k, e in zip(z["seg_key"].astype(str), z["emb"]):
                out.setdefault(k, e)
    for st in stores:
        out.update(read_store(st, "av"))
    return {k: np.asarray(v, np.float32) / max(float(np.linalg.norm(v)), 1e-12) for k, v in out.items()}


def row_indexes(part: dict, omni: dict, student_npz: str, name: str):
    """``(Omni index, student vectors in its row order, segments without an Omni vector)``. A segment without one is left
    out of every column of the row, so all columns rank the same segments."""
    have = [(v, s) for v in sorted(part) for s in part[v] if seg_key(v, s[0], s[1]) in omni]
    missing = sum(len(x) for x in part.values()) - len(have)
    idx = SegmentIndex(np.stack([omni[seg_key(v, s[0], s[1])] for v, s in have]), [v for v, _ in have],
                       [s[0] for _, s in have], [s[1] for _, s in have], name)
    with np.load(student_npz) as z:
        st = hm.student_rows(idx, z["video_id"], z["ts"], z["te"], z["emb"])
    return idx, st / np.maximum(np.linalg.norm(st, axis=1, keepdims=True), 1e-12), missing


def pooled(index, vecs) -> SegmentIndex:
    """R1: one L2-normalised mean per video of ``vecs`` (rows of ``index``)."""
    sums = np.add.reduceat(vecs, index.offsets, axis=0)
    return SegmentIndex(sums / np.maximum(np.linalg.norm(sums, axis=1, keepdims=True), 1e-12), index.videos,
                        np.minimum.reduceat(index.ts, index.offsets), np.maximum.reduceat(index.te, index.offsets), "r1")


def spearman(a, b) -> float:
    rank = lambda x: np.argsort(np.argsort(np.asarray(x), kind="stable"), kind="stable")
    return float(np.corrcoef(rank(a), rank(b))[0, 1])


def pick_beta(r1: dict) -> float:
    """§7.1: highest R@1 multi of (U)β on R8; tie -> the smaller β."""
    return min(r1, key=lambda b: (-r1[b], b))


def main_method(beta_star: float, pair_ub: dict) -> str:
    """§7.2: UEMR-Full if β* > 0 and the lower CI bound of (U)β* - (b) on R8 > 0, else UEMR-Omni."""
    return "UEMR-Full" if beta_star > 0 and pair_ub["ci_low"] > 0 else "UEMR-Omni"


def dump(path, obj) -> None:
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        json.dump(obj, f, indent=1, ensure_ascii=False)


def cmd_score(a, pr) -> None:
    kho, dur, caps = gallery(pr, a.val_manifest)
    text = load_text(a.text)
    if [str(x) for x in text["caption_id"]] != [c["caption_id"] for c in caps]:
        raise SystemExit("CHECK text.npz cùng thứ tự câu hỏi: FAIL")
    t, vids = text["emb"], text["video_id"]
    full = np.array([i for i, c in enumerate(caps) if c["partial"] == "Full Match"])
    sel = json.load(open(os.path.join(a.segments, "selected.json"), encoding="utf-8"))
    ev = pr["eval"]["bootstrap"]
    boot = lambda x, y, m=None: sc.boot(x if m is None else x[m], y if m is None else y[m], vids if m is None else vids[m], ev["n_boot"], ev["seed"])
    omni = omni_vectors(a.omni_db, a.omni_store)
    names = ("r0", "r2", "r3", "r6", "r8", "r7")
    parts = {n: json.load(open(os.path.join(a.segments, f"{n}.json"), encoding="utf-8")) for n in names}
    IDX, ST, MISS = {}, {}, {}
    for n in names:
        IDX[n], ST[n], MISS[n] = row_indexes(parts[n], omni, os.path.join(a.student, f"student_{n}.npz"), n)
    total = {n: sum(len(x) for x in parts[n].values()) for n in names}
    fail_pct = 100 * sum(MISS.values()) / sum(total.values())
    print("clip không có vector Omni:", MISS, f"({fail_pct:.2f} %)")
    if fail_pct > 2:
        raise SystemExit(f"CHECK clip thiếu vector Omni <= 2 %: FAIL ({fail_pct:.2f} %)")
    i8 = IDX["r8"]
    r6_keys = {hm.row_key(v, x, y) for v, x, y in zip(IDX["r6"].seg_video, IDX["r6"].ts, IDX["r6"].te)}
    fill = np.array([hm.row_key(v, x, y) not in r6_keys for v, x, y in zip(i8.seg_video, i8.ts, i8.te)])
    sb8, ss8 = t @ i8.emb.T, t @ ST["r8"].T
    zb, zs = bs.zstats(sb8), bs.zstats(ss8)
    print(f"điểm trên mọi cặp (câu, đoạn R8): (b) mean {zb[0]:.4f} std {zb[1]:.4f} | (S) mean {zs[0]:.4f} std {zs[1]:.4f}")

    R = {}
    for beta in BETAS:
        R[("U", beta, "r8")] = sc.run(i8, text, caps, sim=sb8 + beta * ss8)
    R[("b", None, "r8")] = sc.run(i8, text, caps, sim=sb8)
    if not all(np.array_equal(R[("U", 0.0, "r8")][m]["ranks"], R[("b", None, "r8")][m]["ranks"]) for m in sc.MODES):
        raise SystemExit("CHECK β = 0 cho đúng mảng hạng của (b): FAIL")
    print("CHECK β = 0 cho đúng mảng hạng của (b): OK")
    r1u = {beta: 100 * float((R[("U", beta, "r8")]["multi"]["ranks"] <= 1).mean()) for beta in BETAS}
    bstar = pick_beta(r1u)
    for n in names:
        sb, ss = t @ IDX[n].emb.T, t @ ST[n].T
        R.setdefault(("b", None, n), sc.run(IDX[n], text, caps, sim=sb))
        R[("S", None, n)] = sc.run(IDX[n], text, caps, sim=ss)
        R.setdefault(("U", bstar, n), sc.run(IDX[n], text, caps, sim=sb + bstar * ss))
    R[("C1", None, "r8")] = sc.run(i8, text, caps, sim=bs.c1z(sb8, ss8, fill, zb, zs))
    p1b, p1s = IDX["r6"].mean_pooled("r1"), pooled(IDX["r6"], ST["r6"])
    R[("b", None, "r1")] = sc.run(p1b, text, caps)
    R[("S", None, "r1")] = sc.run(p1s, text, caps)
    R[("U", bstar, "r1")] = sc.run(p1b, text, caps, sim=t @ p1b.emb.T + bstar * (t @ p1s.emb.T))
    rand = os.path.join(a.student, "student_r8_rand.npz")
    if os.path.exists(rand):
        _, srand, _ = row_indexes(parts["r8"], omni, rand, "r8")
        R[("S_rand", None, "r8")] = sc.run(i8, text, caps, sim=t @ srand.T)

    os.makedirs(a.out, exist_ok=True)
    col = lambda c, beta: f"U{beta:g}" if c == "U" else c
    table = []
    for (c, beta, n), res in R.items():
        met = sc.row_metrics(res, full, p1b if n == "r1" else IDX[n], pooled=n == "r1")
        table += [{"dòng": n.upper(), "cột": col(c, beta), "mode": m, **met[m]} for m in sc.MODES]
        dump(os.path.join(a.out, f"ranks_{col(c, beta)}_{n.upper()}.json"),
             {**{m: res[m]["ranks"].tolist() for m in sc.MODES}, **{f"vcmr_{m}": res[m]["vcmr"].astype(int).tolist() for m in sc.MODES}})
    get = lambda c, n, m="multi": R[(c, bstar if c == "U" else None, n)][m]
    pairs = []
    for m in sc.MODES:
        for x, y in (("r8", "r0"), ("r6", "r0"), ("r8", "r3"), ("r6", "r3"), ("r8", "r2"), ("r6", "r1"), ("r7", "r8")):
            pairs.append({"mode": m, "chỉ số": "R@1", "cột": "b", "cặp": f"{x.upper()} - {y.upper()}", **boot(get("b", x, m)["ranks"], get("b", y, m)["ranks"])})
        for name, (cx, nx), (cy, ny) in (("(U) - (b) R8", ("U", "r8"), ("b", "r8")), ("(S) R8 - R0 (b)", ("S", "r8"), ("b", "r0")),
                                          ("C1 - R6 (b)", ("C1", "r8"), ("b", "r6"))):
            pairs.append({"mode": m, "chỉ số": "R@1", "cột": f"{cx} / {cy}", "cặp": name, **boot(get(cx, nx, m)["ranks"], get(cy, ny, m)["ranks"])})
        pairs.append({"mode": m, "chỉ số": "VCMR R@1", "cột": "b", "cặp": "R6 - R3",
                      **boot(sc.vcmr_rank(get("b", "r6", m)["vcmr"]), sc.vcmr_rank(get("b", "r3", m)["vcmr"]))})
    ub = next(q for q in pairs if q["mode"] == "multi" and q["cặp"] == "(U) - (b) R8")
    method = main_method(bstar, ub)

    groups = {}
    miss = bs.missed_by(parts["r6"], caps, 0.3)
    for g, mask in (("bước bị R6 bỏ sót (tIoU < 0.3)", miss), ("bước R6 bắt được", ~miss)):
        groups[g] = {"queries": int(mask.sum()), **{f"{x.upper()} - {y.upper()} (b) multi": boot(get("b", x)["ranks"], get("b", y)["ranks"], mask)
                                                   for x, y in (("r6", "r3"), ("r8", "r6"))}}
    shared = bs.shared_text(caps)
    for g, mask in (("text trùng ở >= 2 video", shared), ("text duy nhất", ~shared)):
        groups[g] = {"queries": int(mask.sum()), **{f"R8 - R0 (b) {m}": boot(get("b", "r8", m)["ranks"], get("b", "r0", m)["ranks"], mask) for m in sc.MODES}}
    groups["tỉ lệ % đoạn đứng đầu là đoạn lấp (R8, multi)"] = {c: 100 * float(fill[get(c, "r8")["top_row"]].mean()) for c in ("b", "S", "U", "C1")}
    if ("S_rand", None, "r8") in R:
        groups["C3: (S) R8 article đoán - article ngẫu nhiên (multi)"] = boot(get("S", "r8")["ranks"], R[("S_rand", None, "r8")]["multi"]["ranks"])
    groups["Spearman hạng (b) với (S) trên R8 (multi)"] = spearman(get("b", "r8")["ranks"], get("S", "r8")["ranks"])
    groups["hit@1 (b) / (S) trên R8 (multi)"] = bs.hit_table(get("b", "r8")["ranks"], get("S", "r8")["ranks"])
    cost = {"clip Omni phía video trên kho": {"UEMR-Omni / Full (R8)": len(i8.emb), "UEMR-Hybrid (R6)": len(IDX["r6"].emb), "UEMR-Lite": 0},
            "clip Omni / video": {"R8": len(i8.emb) / len(kho), "R6": len(IDX["r6"].emb) / len(kho)}}
    dump(os.path.join(a.out, "metrics.json"), {"table": table, "beta_R@1_multi_R8": r1u, "beta_star": bstar, "zscore": {"b": zb, "S": zs},
                                               "missing_omni": MISS, "clips": total, "main_method": method})
    dump(os.path.join(a.out, "pairs.json"), pairs)
    dump(os.path.join(a.out, "groups.json"), groups)
    dump(os.path.join(a.out, "cost.json"), cost)
    now = time.strftime("%Y-%m-%d %H:%M:%S")
    with open(a.tried, "a", encoding="utf-8", newline="\n") as f:
        for beta in BETAS:
            f.write(json.dumps({"what": "(U)_β trên R8 (file 02 §6)", "beta": beta, "R@1 multi": r1u[beta], "time": now}, ensure_ascii=False) + "\n")
        f.write(json.dumps({"what": "chọn β*, phương pháp chính (file 02 §7)", "beta_star": bstar, "(U) - (b) R8 multi": ub, "main_method": method,
                            "time": now}, ensure_ascii=False) + "\n")

    uc = json.load(open(a.uniav_checks, encoding="utf-8"))
    cfg = {"version": 1, "date": time.strftime("%Y-%m-%d"), "doc": "docs/uemr_chot/UEMR_CHOT_02_HTSTEP_DEV.md §7",
           "note": "chọn trên dev HT-Step (510 video val_seen, 2.990 câu); CI của (U) - (b) lạc quan vì β* cũng chọn trên dev",
           "models": {"uniav": {"repo": "GraceFeel/uniav-htstep-best", "sha256": uc["new"]["ckpt_sha256"], "code": "HanLi-Retriever 77fc4fa",
                                "segments_from": uc["segment_source"], "h_from": "mới",
                                "article": "top-1 của article_head (uemr_htstep.uniav_htstep.predict mode='pred')"},
                      "omni_best": pr["models"]["omni_best"], "omni_pt": pr["models"]["omni_pt"]},
           "params": {"theta": sel["theta"], "r6_postprocess": pr["params"]["r6_postprocess"], "l_min": sel["l_min"], "l_fill": sel["l_fill"],
                      "beta_star": bstar, "agg": "max", "omni_vectors": "fp16"},
           "rules": {"r2_chunk": "L sao cho tổng clip R2 gần nhất tổng clip R8 trên từng bộ (uemr_htstep.partitions.pick_chunk, 0,01 s, hoà lấy L lớn hơn); không dùng nhãn",
                     "zscore": "μ, σ của cos trên mọi cặp (câu của kho, đoạn R8 của kho), cho (b) và (S); tính lại trên từng bộ; không dùng nhãn",
                     "text_normalize": "uemr_htstep.data.normalize: lower, bỏ dấu câu, gộp khoảng trắng",
                     "multi_positive": pr["eval"]["multi_positive"], "query": "câu mô tả của bước GT, nguyên văn"},
           "main_method": method,
           "columns": {"UEMR-Omni": "(b) trên R8", "UEMR-Lite": "(S) trên R8", "UEMR-Full": f"(U) β = {bstar:g} trên R8", "UEMR-Hybrid": "C1 z-score trên R8"},
           "dev_values": {"zscore": {"b": zb, "S": zs}, "chunk_sec": sel["chunk_sec"], "clips_r8": sel["clips_r8"], "(U) - (b) R8 multi": ub["CI"]},
           "eval": {"metrics": ["R@1", "R@5", "R@10", "MdR", "VCMR R@1"], "modes": ["multi", "strict"], "main_mode": "multi", "bootstrap": ev}}
    dump(a.config, cfg)
    print(json.dumps({"β → R@1 multi R8": r1u, "beta_star": bstar, "(U) - (b) R8 multi": ub["CI"], "main_method": method}, ensure_ascii=False))
    print("sha256", a.config, hashlib.sha256(open(a.config, "rb").read()).hexdigest())


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("rows")
    for flag in ("--uniav-raw", "--val-manifest", "--out"):
        r.add_argument(flag, required=True)
    s = sub.add_parser("select")
    for flag in ("--segments", "--student", "--text", "--val-manifest", "--tried"):
        s.add_argument(flag, required=True)
    c = sub.add_parser("score")
    for flag in ("--segments", "--student", "--text", "--val-manifest", "--uniav-checks", "--out", "--config"):
        c.add_argument(flag, required=True)
    c.add_argument("--omni-db", nargs="+", required=True)
    c.add_argument("--omni-store", nargs="*", default=[])
    c.add_argument("--tried", default="results/uemr_final/dev/tried.jsonl")
    for x in (r, s, c):
        x.add_argument("--protocol", default="configs/htstep_uemr_protocol.json")
    a = p.parse_args()
    pr = json.load(open(a.protocol, encoding="utf-8"))
    {"rows": cmd_rows, "select": cmd_select, "score": cmd_score}[a.cmd](a, pr)


if __name__ == "__main__":
    main()
