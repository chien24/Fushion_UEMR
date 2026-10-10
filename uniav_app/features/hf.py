"""FEATURE_SOURCE 'hf': the features Athena was trained and evaluated on.

HF dataset (public) ``nguyenminh04/uniav-youcook2-data``: ``iv2_feats/iv2_feats_00..07.tar`` +
``iv2_feats/manifest.json`` = ``{'n_files', 'shards': {tar_name: [<id>.npz, ...]}, 'format'}``
(written by Athena's tools/iv2_finalize_colab.py::pack). Only the shards holding wanted videos are
downloaded, and only those videos are extracted. No token is needed; one set in Colab Secrets
(``HF_TOKEN``) or the environment is used if present, never printed or written.
"""

from __future__ import annotations

import json
import os
import tarfile
from pathlib import Path

from ..hub import hf_token
from .base import FeatureSource


class HFSource(FeatureSource):
    name = "hf"

    def _download(self, filename: str, token: str) -> str:
        from huggingface_hub import hf_hub_download
        return hf_hub_download(repo_id=self.cfg.hf_repo, filename=filename, repo_type="dataset", token=token,
                               local_dir=self.cfg.hf_download_dir)

    def _shard_index(self, token: str) -> dict[str, str]:
        """video_id -> tar name, from manifest.json (or by listing the tars when it has no 'shards')."""
        sub = self.cfg.hf_subdir
        with open(self._download(f"{sub}/manifest.json", token), encoding="utf-8") as f:
            man = json.load(f)
        shards = man.get("shards") if isinstance(man, dict) else None
        if isinstance(shards, dict):
            return {Path(n).stem: tar for tar, names in shards.items() for n in names}
        print(f"[hf] manifest.json has no 'shards' -> scanning the tars")
        from huggingface_hub import HfApi
        files = HfApi().list_repo_files(self.cfg.hf_repo, repo_type="dataset", token=token)
        out = {}
        for f in sorted(x for x in files if x.startswith(sub + "/") and x.endswith(".tar")):
            with tarfile.open(self._download(f, token)) as t:
                for m in t.getmembers():
                    if m.isfile() and m.name.endswith(".npz"):
                        out.setdefault(Path(m.name).stem, Path(f).name)
        return out

    def _prepare(self, todo, durations):
        token = hf_token()   # None is fine: the dataset is public
        where = self._shard_index(token)
        for v in [v for v in todo if v not in where]:
            self._fail(v, f"not in {self.cfg.hf_repo}/{self.cfg.hf_subdir}/manifest.json")
        by_tar: dict[str, list[str]] = {}
        for v in todo:
            if v in where:
                by_tar.setdefault(where[v], []).append(v)
        for i, (tar, vids) in enumerate(sorted(by_tar.items()), 1):
            print(f"[hf] shard {i}/{len(by_tar)} {tar}: {len(vids)} videos", flush=True)
            path = self._download(f"{self.cfg.hf_subdir}/{tar}", token)
            with tarfile.open(path) as t:
                members = {Path(m.name).stem: m for m in t.getmembers() if m.isfile() and m.name.endswith(".npz")}
                for v in vids:
                    m = members.get(v)
                    if m is None:
                        self._fail(v, f"not found in {tar}")
                        continue
                    try:
                        dst = self.npz_path(v)
                        tmp = dst.with_name(dst.name + ".part")
                        src = t.extractfile(m)   # read by name: nothing is written outside self.dir
                        with open(tmp, "wb") as out:
                            out.write(src.read())
                        os.replace(tmp, dst)
                        self._add(v, dst, durations.get(v), shard=tar)
                    except Exception as e:   # noqa: BLE001 - one bad member must not stop the others
                        self._fail(v, repr(e))
            if not self.cfg.hf_keep_tars:
                os.remove(path)
