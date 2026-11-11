"""Scoring of the main run as functions (cell 11 of ``notebooks/uemr/uemr_htstep_AZ.ipynb``), for the add-on plan.

``reproduce`` recomputes every row x column of the main run from its vectors and compares with the saved results (plan
bổ sung §2.1): the rank arrays of column (b) must be identical, and every metric of the table and every bootstrap pair
must match. Local:

    python -m uemr_htstep.score --db <encode_omni>/best/db --student-text <encode_omni>/student_text/db \
        --uniav D:/Học/KL/output/htstep_uemr/uniav --val-manifest val_omni_htstep.jsonl --results results/uemr_htstep/a4ca43db
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from omni_retrieval.evaluate import load_text, paired_bootstrap, rank_metrics
from omni_retrieval.search import SegmentIndex

from . import data as hd
from . import metrics as hm
from .partitions import FILES, ROWS

MODES = hm.MODES
PAIRS = [("r8", "global"), ("r8", "uni_M"), ("uniav_pred", "uni_M"), ("r8", "uni_chunk"), ("uniav_pred", "global"),
         ("r8", "uniav_pred"), ("uniav_gtart", "uniav_pred"), ("gt", "r8")]
VCMR_PAIRS = [("r8", "uni_M"), ("uniav_pred", "uni_M"), ("r8", "uniav_pred")]


def boot(a, b, vids, n_boot: int = 1000, seed: int = 0, k: int = 1) -> dict:
    """``A - B`` of R@k, paired bootstrap by video. ``a`` and / or ``b`` may be ``[n_seeds, Q]`` (mean over seeds, R4)."""
    a, b = np.asarray(a), np.asarray(b)
    if b.ndim == 2 and a.ndim == 1:      # paired_bootstrap takes the seeds on its first argument: B - A, then negate
        r = paired_bootstrap(b, a, k, n_boot, seed, "video", vids)
        r = {**r, "diff": -r["diff"], "ci_low": -r["ci_high"], "ci_high": -r["ci_low"]}
    else:
        r = paired_bootstrap(a, b, k, n_boot, seed, "video", vids)
    return {"A - B": r["diff"], "CI": f"[{r['ci_low']:+.2f}, {r['ci_high']:+.2f}]", "ci_low": r["ci_low"], "ci_high": r["ci_high"]}


def vcmr_rank(hit) -> np.ndarray:
    """VCMR hits as ranks (1 = hit) so that ``boot(..., k=1)`` gives the VCMR R@1 difference."""
    return np.where(np.asarray(hit), 1, 2)


def run(index, text, captions, sim=None, bonus=None, agg="max", **agg_kw) -> dict:
    return {m: hm.text_to_video(index, text, captions, m, agg=agg, sim=sim, bonus=bonus, **agg_kw) for m in MODES}


def row_metrics(res: dict, full_idx, index=None, pooled: bool = False) -> dict:
    """``{mode: metrics}`` as in the main table."""
    out = {}
    for m in MODES:
        r = res[m]
        out[m] = {**rank_metrics(r["ranks"]), "VCMR R@1": float("nan") if pooled else 100 * float(r["vcmr"].mean()),
                  "R@1 Full Match": 100 * float((r["ranks"][full_idx] <= 1).mean())}
        if index is not None:
            out[m].update({"vec/video": float(np.mean(index.counts)), "video có vector": int(index.n_videos)})
    return out


class Main:
    """The vectors of the main run: Omni ``best`` (``db``), teacher text, the cached student vectors, the queries."""

    def __init__(self, db, student_text, uniav_dir, captions):
        self.db, self.uniav = str(db), Path(uniav_dir)
        self.text = load_text(db)
        self.text_s = load_text(student_text) if student_text else None
        self.caps = captions
        if [str(x) for x in self.text["caption_id"]] != [c["caption_id"] for c in captions]:
            raise ValueError("text.npz and the queries are not in the same order")
        self.full = np.array([i for i, c in enumerate(captions) if c["partial"] == "Full Match"])
        self.vids = self.text["video_id"]
        self._idx = {}

    def index(self, partition: str) -> SegmentIndex:
        if partition not in self._idx:
            self._idx[partition] = SegmentIndex.load(self.db, partition)
        return self._idx[partition]

    def student(self, index, name: str, suffix: str = "") -> np.ndarray:
        with np.load(self.uniav / f"student_{name}{suffix}.npz") as z:
            return hm.student_rows(index, z["video_id"], z["ts"], z["te"], z["emb"])

    def fill_mask(self, index) -> np.ndarray:
        r6 = self.index("uniav_pred")
        keys = {hm.row_key(v, a, b) for v, a, b in zip(r6.seg_video, r6.ts, r6.te)}
        return np.array([hm.row_key(v, a, b) not in keys for v, a, b in zip(index.seg_video, index.ts, index.te)])

    def columns(self, beta: float) -> dict:
        """``{(col, partition): run}`` of every row x column of the main run."""
        R = {("b", p): run(self.index(p), self.text, self.caps) for p in ROWS}
        if self.text_s is not None:
            for col in ("S", "U", "C1"):
                for p in ROWS:
                    if col == "C1" and p != "r8":
                        continue
                    idx = self.index(p)
                    sim = hm.column_sim(col, self.text["emb"], self.text_s["emb"], idx.emb, self.student(idx, FILES[p]), beta,
                                        self.fill_mask(idx))
                    R[(col, p)] = run(idx, self.text, self.caps, sim=sim)
        return R


def reproduce(main: Main, results_dir, beta: float, n_boot: int, seed: int) -> dict:
    """Plan bổ sung §2.1. ``{'ok': bool, 'checks': [...]}``; a check that fails is listed with its first difference."""
    res = Path(results_dir)
    metrics = json.loads((res / "metrics_best.json").read_text(encoding="utf-8"))
    R = main.columns(beta)
    checks = []

    def add(name, ok, detail=""):
        checks.append({"check": name, "ok": bool(ok), "detail": detail})

    for p in ROWS:
        saved = json.loads((res / f"ranks_best_b_{ROWS[p]}.json").read_text(encoding="utf-8"))
        for m in MODES:
            add(f"ranks b/{ROWS[p]}/{m}", np.array_equal(R[("b", p)][m]["ranks"], saved[m]))
            add(f"vcmr b/{ROWS[p]}/{m}", np.array_equal(R[("b", p)][m]["vcmr"].astype(int), saved[f"vcmr_{m}"]))
    table = {(t["mode"], t["cột"], t["dòng"]): t for t in metrics["table"]}
    for (col, p), r in R.items():
        got = row_metrics(r, main.full, main.index(p), pooled=p == "global")
        for m in MODES:
            want = table[(m, col, ROWS[p])]
            diff = {k: (got[m][k], want[k]) for k in got[m] if not (np.isnan(got[m][k]) and np.isnan(want[k])) and abs(got[m][k] - want[k]) > 1e-9}
            add(f"metrics {col}/{ROWS[p]}/{m}", not diff, str(diff)[:200])
    pairs = {(q["mode"], q["chỉ số"], q["cột"], q["cặp"]): q for q in metrics["pairs"]}
    mine = []
    for m in MODES:
        for a, b in PAIRS:
            mine.append(((m, "R@1", "b", f"{ROWS[a]} - {ROWS[b]}"), boot(R[("b", a)][m]["ranks"], R[("b", b)][m]["ranks"], main.vids, n_boot, seed)))
        for a, b in VCMR_PAIRS:
            mine.append(((m, "VCMR R@1", "b", f"{ROWS[a]} - {ROWS[b]}"),
                         boot(vcmr_rank(R[("b", a)][m]["vcmr"]), vcmr_rank(R[("b", b)][m]["vcmr"]), main.vids, n_boot, seed)))
        if main.text_s is not None:
            for key, a, b in (("U-b", ("U", "r8"), ("b", "r8")), ("C1-R6", ("C1", "r8"), ("b", "uniav_pred")), ("S-b", ("S", "r8"), ("b", "r8"))):
                cap = {"U-b": "(U) - (b) R8", "C1-R6": "C1 - R6", "S-b": "(S) - (b) R8"}[key]
                mine.append(((m, "R@1", key, cap), boot(R[a][m]["ranks"], R[b][m]["ranks"], main.vids, n_boot, seed)))
    for k, v in mine:
        w = pairs.get(k)
        add(f"pair {'/'.join(k)}", w is not None and all(abs(v[x] - w[x]) <= 1e-9 for x in ("A - B", "ci_low", "ci_high")),
            "" if w is None else f"{v['CI']} vs {w['CI']}")
    t2c_saved = json.loads((res / "ranks_best_b_t2c.json").read_text(encoding="utf-8"))
    ck, ce = hm.gt_clips(Path(main.db) / "segments.npz")
    for m in MODES:
        add(f"ranks t2c/{m}", np.array_equal(hm.text_to_clip(main.text, main.caps, ck, ce, m), t2c_saved[m]))
    return {"ok": all(c["ok"] for c in checks), "n": len(checks), "failed": [c for c in checks if not c["ok"]], "R": R}


def main_cli() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    for flag in ("--db", "--student-text", "--uniav", "--val-manifest", "--results"):
        p.add_argument(flag, required=True)
    p.add_argument("--protocol", default="configs/htstep_uemr_protocol.json")
    a = p.parse_args()
    pr = json.loads(Path(a.protocol).read_text(encoding="utf-8"))
    caps = hd.queries(hd.read_jsonl(a.val_manifest), pr["data"]["gallery"])
    out = reproduce(Main(a.db, a.student_text, a.uniav, caps), a.results, float(pr["params"]["beta"]),
                    pr["eval"]["bootstrap"]["n_boot"], pr["eval"]["bootstrap"]["seed"])
    print(json.dumps({k: v for k, v in out.items() if k != "R"}, indent=1, ensure_ascii=False))
    print(f"CHECK tái lập lượt chính ({out['n']} phép so): {'OK' if out['ok'] else 'FAIL'}")


if __name__ == "__main__":
    main_cli()
