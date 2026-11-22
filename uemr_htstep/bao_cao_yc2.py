"""Phân tích SAU KHI KHOÁ trên benchmark YouCook2 (chỉ báo, không đổi K1–K6 hay tham số): số cho báo cáo KLTN.

1. CI của Omni + student: Sự kiện − Đều theo số đoạn (ranks_U_R6 − ranks_U_R3), từ hạng đã lưu, không cần vector.
2. Bốn ví dụ định tính cho mục 7.7 (chọn tự động, seed 0): UEMR đúng / Cả video sai; bước bộ định vị bỏ sót (tIoU < 0,3
   với mọi đoạn Sự kiện) nhưng UEMR đúng nhờ đoạn lấp; khoảnh khắc đúng (VCMR); thất bại (hạng > 100). Với mỗi câu: hạng
   theo từng cách, video top-1 và đoạn trả về, đoạn GT. Cần vector Omni + student của R8.
3. (tuỳ chọn ``--mp4``) 3 khung hình (25 / 50 / 75 %) của đoạn GT và đoạn trả về, bằng ffmpeg; không vẽ thêm gì.

Colab, sau ô 1–3 của ``notebooks/uemr/uemr_final_benchmark.ipynb`` (đã tải HF ``GraceFeel/uemr-bench-yc2`` về /content/bench):

    python -m uemr_htstep.bao_cao_yc2 --bench /content/bench/yc2 --cfg configs/uemr_final.json --out /content/bao_cao_yc2 \
        --mp4 /content/videos
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess

import numpy as np

from omni_retrieval.evaluate import load_text
from omni_retrieval.search import tiou_vec

from . import score as sc
from .final_dev import omni_vectors, row_indexes

COLS = [(c, r) for r in ("R0", "R3", "R6", "R2", "R8", "R7") for c in ("b", "S", "U")]


def rj(p):
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def miss_r6(caps, r6) -> np.ndarray:
    """Best tIoU of each query's GT step with the Sự kiện segments of its video (< 0.3 = bộ định vị bỏ sót)."""
    out = []
    for c in caps:
        segs = np.asarray(r6.get(c["video_id"], []), np.float64).reshape(-1, 2)
        out.append(float(tiou_vec(segs[:, 0], segs[:, 1], c["gt_ts"], c["gt_te"]).max()) if len(segs) else 0.0)
    return np.array(out)


def pick(rk, vcmr, best6, rng) -> dict:
    u8, u6, b0 = rk[("U", "R8")], rk[("U", "R6")], rk[("b", "R0")]
    cand = {"thắng Cả video": np.flatnonzero((u8 == 1) & (b0 >= 10)),
            "bỏ sót nhưng đoạn lấp cứu": np.flatnonzero((best6 < 0.3) & (u8 == 1) & (u6 > 1)),
            "đúng khoảnh khắc": np.flatnonzero(vcmr == 1),
            "thất bại": np.flatnonzero(u8 > 100)}
    return {k: (int(rng.choice(v)), len(v)) for k, v in cand.items() if len(v)}


def returned(q, t, idx, st, beta) -> dict:
    """Top-1 video of Omni + student on R8 and its argmax segment (= the moment UEMR returns)."""
    s = idx.emb @ t[q] + beta * (st @ t[q])
    vid = np.asarray(idx.seg_video)
    best = int(np.argmax(s))
    return {"top1_video": str(vid[best]), "top1_seg": [round(float(idx.ts[best]), 2), round(float(idx.te[best]), 2)],
            "top1_score": round(float(s[best]), 4)}


def frames(mp4, ts, te, out_prefix) -> list[str]:
    paths = []
    for p in (0.25, 0.5, 0.75):
        o = f"{out_prefix}_{int(p * 100)}.jpg"
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-ss", f"{ts + p * (te - ts):.2f}", "-i", mp4, "-frames:v", "1",
                        "-vf", "scale=320:-2", o], check=True)
        paths.append(o)
    return paths


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bench", required=True, help="thư mục yc2/ của GraceFeel/uemr-bench-yc2")
    ap.add_argument("--cfg", default="configs/uemr_final.json")
    ap.add_argument("--out", required=True)
    ap.add_argument("--mp4", default=None, help="thư mục <video_id>.mp4 (bỏ qua thì không trích khung hình)")
    a = ap.parse_args()
    B, cfg = a.bench, rj(a.cfg)
    beta, ev = float(cfg["params"]["beta_star"]), cfg["eval"]["bootstrap"]
    caps = rj(f"{B}/results/queries.json")
    vids = np.array([c["video_id"] for c in caps])
    R = {(c, r): rj(f"{B}/results/score/ranks_{c}_{r}.json") for c, r in COLS}
    os.makedirs(a.out, exist_ok=True)

    # 1. CI (chỉ báo)
    ci = {m: sc.boot(R[("U", "R6")][m], R[("U", "R3")][m], vids, ev["n_boot"], ev["seed"]) for m in ("multi", "strict")}
    ci["VCMR multi"] = sc.boot(sc.vcmr_rank(R[("U", "R6")]["vcmr_multi"]), sc.vcmr_rank(R[("U", "R3")]["vcmr_multi"]),
                               vids, ev["n_boot"], ev["seed"])
    print("Omni + student: Sự kiện − Đều theo số đoạn (sau khi khoá, chỉ báo):", json.dumps(ci, ensure_ascii=False))

    # 2. ví dụ định tính
    seg = {n: rj(f"{B}/segments/{n}.json") for n in ("r6", "r8")}
    best6 = miss_r6(caps, seg["r6"])
    rk = {k: np.asarray(v["multi"]) for k, v in R.items()}
    chosen = pick(rk, np.asarray(R[("U", "R8")]["vcmr_multi"]), best6, np.random.default_rng(0))
    text = load_text(f"{B}/omni/best/db")
    assert [str(x) for x in text["caption_id"]] == [c["caption_id"] for c in caps], "text.npz khác thứ tự câu hỏi"
    idx, st, miss = row_indexes(seg["r8"], omni_vectors([], [f"{B}/omni/best/clips"]), f"{B}/student/student_r8.npz", "r8")
    assert miss == 0, f"{miss} đoạn R8 không có vector Omni"
    r6keys = {(v, round(s[0], 2), round(s[1], 2)) for v, ss in seg["r6"].items() for s in ss}
    ex = []
    for kind, (q, n_cand) in chosen.items():
        c = caps[q]
        r = returned(q, text["emb"], idx, st, beta)
        r["top1_is_fill"] = (r["top1_video"], *r["top1_seg"]) not in r6keys
        gt = [c["gt_ts"], c["gt_te"]]
        e = {"loại": kind, "số câu thoả": n_cand, "caption_id": c["caption_id"], "video_id": c["video_id"], "câu": c["sentence"],
             "đoạn GT": gt, "tIoU GT với Sự kiện tốt nhất": round(float(best6[q]), 3),
             "hạng multi": {f"{col} {row}": int(rk[(col, row)][q]) for col, row in COLS},
             "VCMR U R8": int(R[("U", "R8")]["vcmr_multi"][q]), **r,
             "tIoU đoạn trả về với GT": round(float(tiou_vec(np.array([r["top1_seg"][0]]), np.array([r["top1_seg"][1]]), *gt)[0]), 3)
             if r["top1_video"] == c["video_id"] else None}
        if a.mp4:
            p = os.path.join(a.out, c["caption_id"])
            e["khung GT"] = frames(f"{a.mp4}/{c['video_id']}.mp4", *gt, p + "_gt")
            e["khung trả về"] = frames(f"{a.mp4}/{r['top1_video']}.mp4", *r["top1_seg"], p + "_ret")
        ex.append(e)
    json.dump({"ghi chú": "phân tích sau khi khoá, chỉ báo", "U R6 - U R3": ci, "ví dụ 7.7": ex},
              open(f"{a.out}/bao_cao_yc2.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(json.dumps(ex, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
