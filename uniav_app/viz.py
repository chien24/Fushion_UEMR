"""Look at one video: GT vs predicted segments as a table and a matplotlib timeline, and ffmpeg clips."""

from __future__ import annotations

import os
import shutil
import subprocess


def _fmt(t: float) -> str:
    return f"{int(t // 60)}:{t % 60:04.1f}"


def print_timeline(gt: list[dict] | None, events: list[dict], width: int = 70) -> None:
    """GT steps and predicted events side by side, in time order."""
    if gt:
        print("GT:")
        for e in gt:
            print(f"  [{_fmt(e['ts'])} - {_fmt(e['te'])}] {e['caption'][:width]}")
    print("Dự đoán:")
    for e in events:
        extra = f" | ORACLE: {e['caption_oracle'][:40]}" if e.get("caption_oracle") else ""
        print(f"  [{_fmt(e['t_s'])} - {_fmt(e['t_e'])}] conf {e['conf']:.2f}  {e['caption'][:width]}{extra}")


def plot_timeline(gt: list[dict] | None, events: list[dict], duration: float, title: str = "",
                  api_events: list[dict] | None = None, ax=None):
    """Bars per row: GT (if any), the final segments (alpha = conf), optionally the API selection."""
    import matplotlib.pyplot as plt
    rows = ([("GT", gt, "tab:green")] if gt else []) + [("UniAV (uemr)", events, "tab:blue")]
    if api_events is not None:
        rows.append(("UniAV (api)", api_events, "tab:orange"))
    if ax is None:
        _, ax = plt.subplots(figsize=(14, 0.8 + 0.7 * len(rows)))
    for y, (name, evs, color) in enumerate(reversed(rows)):
        for i, e in enumerate(evs):
            ts = e.get("ts", e.get("t_s")); te = e.get("te", e.get("t_e"))
            alpha = 0.35 + 0.65 * float(e.get("conf", 1.0)) if name != "GT" else 0.8
            ax.barh(y, te - ts, left=ts, height=0.6, color=color, alpha=min(alpha, 1.0), edgecolor="black", lw=0.5)
            ax.text(ts + 0.5 * (te - ts), y, str(i), ha="center", va="center", fontsize=7)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([r[0] for r in reversed(rows)])
    ax.set_xlim(0, duration)
    ax.set_xlabel("giây")
    ax.set_title(title)
    plt.tight_layout()
    return ax


def find_ffmpeg() -> str:
    exe = shutil.which("ffmpeg")
    if exe:
        return exe
    from .athena_api import import_athena
    import_athena()
    from athena.encoders.media import find_ffmpeg as _find
    return _find()


def cut_clip(video: str, ts: float, te: float, out: str, height: int = 360) -> str:
    """Re-encode [ts, te] of ``video`` to a small mp4 (frame-accurate, with audio)."""
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    cmd = [find_ffmpeg(), "-v", "error", "-y", "-ss", f"{ts:.3f}", "-i", video, "-t", f"{max(te - ts, 0.1):.3f}",
           "-vf", f"scale=-2:{height}", "-c:v", "libx264", "-preset", "veryfast", "-crf", "28", "-c:a", "aac", out]
    subprocess.run(cmd, check=True)
    return out


def show_clips(video: str, events: list[dict], out_dir: str, n: int = 3) -> list[str]:
    """Cut the first ``n`` events and display them inline (IPython)."""
    from IPython.display import HTML, Video, display
    paths = []
    for i, e in enumerate(events[:n]):
        ts = e.get("ts", e.get("t_s")); te = e.get("te", e.get("t_e"))
        p = cut_clip(video, ts, te, os.path.join(out_dir, f"{os.path.splitext(os.path.basename(video))[0]}_{i}.mp4"))
        display(HTML(f"<b>#{i} [{_fmt(ts)} - {_fmt(te)}]</b> {e.get('caption', '')}"))
        display(Video(p, embed=True, width=480))
        paths.append(p)
    return paths
