"""Downloads from the Hugging Face Hub (both repos are public; a token is used only if one is set).

* model repo ``nguyenminh04/athena``: ``athena.pth`` (run ov_E1), ``internvideo2/InternVideo2-stage2_1b-224p-f4.pt``,
  ``internvideo2/audio_6b.pth`` (the encoders of the video path)
* dataset repo ``nguyenminh04/uniav-youcook2-data``: ``iv2_feats/*.tar`` + ``manifest.json`` (features/hf.py)
"""

from __future__ import annotations

import os

ATHENA_REPO = "nguyenminh04/athena"
FEATURE_REPO = "nguyenminh04/uniav-youcook2-data"

# sizes of the copies in UniAV-fixed/ckpt (a mismatch is reported, not fatal)
SIZES = {
    "athena.pth": 278180035,
    "internvideo2/InternVideo2-stage2_1b-224p-f4.pt": 2820610931,
    "internvideo2/audio_6b.pth": 361325972,
}
ENCODERS = ("internvideo2/InternVideo2-stage2_1b-224p-f4.pt", "internvideo2/audio_6b.pth")


def hf_token() -> str | None:
    """Optional token (Colab Secrets ``HF_TOKEN`` or the environment); never printed or written."""
    tok = None
    try:
        from google.colab import userdata
        tok = userdata.get("HF_TOKEN")
    except Exception:   # not on Colab, secret missing or notebook access off: public repos need none
        tok = None
    return tok or os.environ.get("HF_TOKEN") or None


def download(repo_id: str, filename: str, local_dir: str, repo_type: str = "model") -> str:
    """``hf_hub_download`` into ``local_dir/filename`` (skipped when already complete); prints the size check."""
    from huggingface_hub import hf_hub_download
    path = hf_hub_download(repo_id=repo_id, filename=filename, repo_type=repo_type, local_dir=local_dir,
                           token=hf_token())
    size = os.path.getsize(path)
    want = SIZES.get(filename)
    note = "" if want is None else ("OK" if size == want else f"[!] khác {want} byte của bản local")
    print(f"{path}: {size} byte {note}")
    return path


def download_checkpoint(local_dir: str = "/content/ckpt", filename: str = "athena.pth") -> str:
    return download(ATHENA_REPO, filename, local_dir)


def download_encoders(local_dir: str = "/content/ckpt") -> list[str]:
    """The two InternVideo2 encoders -> ``local_dir/internvideo2/`` (about 3.2 GB)."""
    return [download(ATHENA_REPO, f, local_dir) for f in ENCODERS]
