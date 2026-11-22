"""Giả thuyết recipe trên YouCook2 — phân tích SAU KHI KHOÁ, chỉ báo (báo cáo KLTN mục 7.5.3).

KẾ HOẠCH (ghi và commit trước khi có số; không đổi sau khi thấy kết quả)
-----------------------------------------------------------------------
Giả thuyết: vector student yếu trên YouCook2 vì bộ định vị đoán sai recipe HT-Step của video.

1. Chạy lại riêng bộ định vị (checkpoint 9447de87…, chế độ chính: recipe đoán) trên đặc trưng YouCook2 đã trích của lượt chính,
   lưu cho mỗi video recipe top-1 và xác suất softmax của nó (``run_uniav recipe`` -> ``recipe.jsonl``).
2. Kiểm trước: recipe top-1 chạy lại phải trùng ``uniav_raw.jsonl`` của lượt chính ở MỌI video. Lệch dù một video thì dừng, báo
   số video lệch, không phân tích.
3. Chia 1.500 video thành 3 nhóm theo xác suất top-1 bằng tam phân vị (thấp: p <= q1/3; giữa: q1/3 < p <= q2/3; cao: p > q2/3).
   Mỗi câu hỏi thuộc nhóm của video gốc của nó.
4. Với mỗi nhóm, trên cách cắt Sự kiện + lấp (R8), chế độ multi-positive, hạng lấy từ ``ranks_*_R8.json`` đã lưu:
   R@1 của Student (S), R@1 của Omni (b), và hiệu R@1 (Omni + student) − Omni (U − b) kèm CI 95 % paired bootstrap theo
   video (1.000 lần, seed 0, ``score.boot``).
5. Giả thuyết được ủng hộ nếu hiệu (U − b) tăng dần theo độ tự tin: thấp < giữa < cao (so ước lượng điểm). Ngược lại: không
   ủng hộ. Không thêm luật nào khác.
6. Ghi trong báo cáo: "sau khi khoá, chỉ báo"; độ tự tin của recipe head không phải độ đúng, vì YouCook2 không có recipe
   HT-Step đúng.

    python -m uemr_htstep.recipe_yc2 --recipe recipe.jsonl --raw yc2/uniav/uniav_raw.jsonl --bench yc2 --out recipe_yc2.json
"""

from __future__ import annotations

import argparse
import json

import numpy as np

from . import score as sc


def jl(path) -> dict:
    return {r["video_id"]: r for r in (json.loads(x) for x in open(path, encoding="utf-8") if x.strip())}


def same_top1(recipe: dict, raw: dict) -> list[str]:
    """Videos whose rerun top-1 differs from the main run (or is missing on one side)."""
    return sorted(v for v in set(recipe) | set(raw) if recipe.get(v, {}).get("article_top1") != raw.get(v, {}).get("article_top1"))


def terciles(p: dict) -> dict:
    """``{video: 'thấp' | 'giữa' | 'cao'}`` by the 1/3 and 2/3 quantiles of the top-1 probability."""
    q1, q2 = np.quantile(list(p.values()), [1 / 3, 2 / 3])
    return {v: "thấp" if x <= q1 else "giữa" if x <= q2 else "cao" for v, x in p.items()}


def supported(diffs: dict) -> bool:
    return diffs["thấp"] < diffs["giữa"] < diffs["cao"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--recipe", required=True)
    ap.add_argument("--raw", required=True)
    ap.add_argument("--bench", required=True, help="thư mục yc2/ của GraceFeel/uemr-bench-yc2")
    ap.add_argument("--cfg", default="configs/uemr_final.json")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    recipe, raw = jl(a.recipe), jl(a.raw)
    bad = same_top1(recipe, raw)
    if bad:
        raise SystemExit(f"DỪNG: recipe top-1 chạy lại lệch lượt chính ở {len(bad)} video, ví dụ {bad[:5]}")
    print(f"recipe top-1 trùng lượt chính: {len(raw)} / {len(raw)} video")
    ev = json.load(open(a.cfg, encoding="utf-8"))["eval"]["bootstrap"]
    caps = json.load(open(f"{a.bench}/results/queries.json", encoding="utf-8"))
    vids = np.array([c["video_id"] for c in caps])
    rk = {c: np.asarray(json.load(open(f"{a.bench}/results/score/ranks_{c}_R8.json", encoding="utf-8"))["multi"]) for c in "bSU"}
    grp = terciles({v: r["p_top1"] for v, r in recipe.items()})
    out, diffs = {}, {}
    for g in ("thấp", "giữa", "cao"):
        m = np.array([grp[v] == g for v in vids])
        p = [recipe[v]["p_top1"] for v in recipe if grp[v] == g]
        d = sc.boot(rk["U"][m], rk["b"][m], vids[m], ev["n_boot"], ev["seed"])
        diffs[g] = d["A - B"]
        out[g] = {"video": len(p), "câu": int(m.sum()), "p_top1 min–max": [round(min(p), 4), round(max(p), 4)],
                  "R@1 Student": 100 * float((rk["S"][m] == 1).mean()), "R@1 Omni": 100 * float((rk["b"][m] == 1).mean()),
                  "(Omni + student) − Omni": d}
    res = {"ghi chú": "sau khi khoá, chỉ báo; độ tự tin không phải độ đúng", "nhóm": out, "giả thuyết ủng hộ": supported(diffs)}
    json.dump(res, open(a.out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(json.dumps(res, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
