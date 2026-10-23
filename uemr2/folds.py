"""Cross-fitting folds of Athena for UEMR 2 (spec UEMR_ALIGN_FINAL.md 4.4, CF2). Standard library + numpy.

The 1.002 trainsplit videos (``train_omni_video.trainsplit.jsonl``, the list Athena sạch was checked against) are sorted,
permuted with ``np.random.default_rng(seed)`` and cut into A (first 501) and B (the rest). Fold X's annotation is
``youcookii_annotations_yc2clean.json`` with every ``training`` video outside X relabelled ``holdout`` (this includes the
227 ``training`` videos without features); its config is ``youcook2_event_yc2clean.yaml`` with only ``json_file``
changed. Athena-A gives the q of the videos in B and the other way round.

    python -m uemr2.folds <athena repo> <trainsplit manifest> [--dev-val <ids file>]
"""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import numpy as np

ANNO = "data/youcookii/annotations/youcookii_annotations_yc2clean.json"
CONFIG = "configs/youcook2_event_yc2clean.yaml"


def manifest_videos(manifest) -> list[str]:
    """Video ids of an Omni manifest (one record per caption), e.g. train_omni_video.trainsplit.jsonl."""
    out = set()
    for line in Path(manifest).read_text(encoding="utf-8").splitlines():
        if line.strip():
            out.add(Path(json.loads(line)["video"]).stem)
    return sorted(out)


def halves(videos, seed: int = 0) -> tuple[list[str], list[str]]:
    ids = sorted(videos)
    perm = np.random.default_rng(seed).permutation(len(ids))
    cut = (len(ids) + 1) // 2
    return sorted(ids[i] for i in perm[:cut]), sorted(ids[i] for i in perm[cut:])


def fold_annotation(anno: dict, keep: set[str]) -> dict:
    out = copy.deepcopy(anno)
    for v, x in out["database"].items():
        if x["subset"] == "training" and v not in keep:
            x["subset"] = "holdout"
    return out


def write_folds(repo, train_videos, seed: int = 0, exclude=()) -> dict:
    """Write the two annotation files and the two configs into the Athena repo; return the split.

    ``exclude``: dev / val ids that must not be in either fold (checked, never removed silently)."""
    repo = Path(repo)
    train_videos = sorted(set(train_videos))
    a, b = halves(train_videos, seed)
    assert set(a) | set(b) == set(train_videos) and not set(a) & set(b), "folds must partition the trainsplit"
    assert not (set(a) | set(b)) & set(exclude), "a dev / val video is in a fold"
    anno = json.loads((repo / ANNO).read_text(encoding="utf-8"))
    db = anno["database"]
    assert set(train_videos) <= {v for v, x in db.items() if x["subset"] == "training"}, \
        "a trainsplit video is not 'training' in the yc2clean annotation"
    cfg_text = (repo / CONFIG).read_text(encoding="utf-8")
    assert cfg_text.count(Path(ANNO).name) == 1, "json_file of the yc2clean config not found exactly once"
    out = {"seed": seed, "n": len(train_videos), "A": a, "B": b, "files": {}}
    for name, keep in (("A", a), ("B", b)):
        fa = fold_annotation(anno, set(keep))
        got = sorted(v for v, x in fa["database"].items() if x["subset"] == "training")
        assert got == keep, f"fold {name}: training videos differ from the half"
        ann_path = repo / ANNO.replace(".json", f"_fold{name}.json")
        ann_path.write_text(json.dumps(fa, ensure_ascii=False), encoding="utf-8")
        cfg_path = repo / CONFIG.replace(".yaml", f"_fold{name}.yaml")
        cfg_path.write_text(f"# UEMR 2 fold {name} (Fushion uemr2/folds.py, seed {seed}): youcook2_event_yc2clean.yaml "
                            f"with only json_file changed\n" + cfg_text.replace(Path(ANNO).name, ann_path.name),
                            encoding="utf-8", newline="\n")
        out["files"][name] = {"annotation": str(ann_path.relative_to(repo)).replace("\\", "/"),
                              "config": str(cfg_path.relative_to(repo)).replace("\\", "/")}
    return out


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("repo"); p.add_argument("manifest")
    p.add_argument("--dev-val", nargs="*", default=[], help="files of dev / val video ids (one per line) or jsonl manifests")
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args(argv)
    exclude = set()
    for f in a.dev_val:
        exclude |= set(manifest_videos(f)) if f.endswith(".jsonl") else \
            {l.strip() for l in Path(f).read_text(encoding="utf-8").splitlines() if l.strip()}
    r = write_folds(a.repo, manifest_videos(a.manifest), a.seed, exclude)
    print(json.dumps({"n": r["n"], "A": len(r["A"]), "B": len(r["B"]), "excluded checked": len(exclude),
                      "files": r["files"]}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
