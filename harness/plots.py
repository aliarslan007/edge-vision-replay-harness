"""Required plots: queue length over time, and time-to-post per entry (report both from a 1x run)."""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _title(res, what):
    s = res["run"]["speed"]
    note = "" if s == 1 else f"  (replayed at {s}x: not the required 1x figure)"
    return f"{what} — {res['run']['sample_fps']} fps sampling, {s}x{note}"


def plot_queue(res: dict, path: str):
    series = res["_detail"]["queue_series"]
    fig, ax = plt.subplots(figsize=(9, 3.2))
    if series:
        xs, ys = zip(*series)
        ax.step(xs, ys, where="post", color="#2b55b8")
    for v in res["_detail"]["visits"]:
        ax.axvspan(v["start_s"], v["end_s"], color="#999", alpha=0.15, lw=0)
    ax.set_xlabel("stream time (s)   [grey bands = ground-truth visits]")
    ax.set_ylabel("frames waiting for the VLM")
    ax.set_title(_title(res, "Queue length"), fontsize=10)
    ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


def plot_ttp(res: dict, path: str):
    d = res["_detail"]["ttp"]
    missed = set(res["entries"]["missed_visits"])
    fig, ax = plt.subplots(figsize=(9, 3.6))
    if d:
        ax.plot([x["start_s"] for x in d], [x["ttp_media_s"] for x in d], "o-", color="#2b55b8", ms=4,
                label="time-to-post (entry -> record posted)")
        ax.plot([x["start_s"] for x in d], [x["capture_delay_s"] for x in d], ".", color="#999",
                label="of which: entry -> sampled frame")
    mv = [v for v in res["_detail"]["visits"] if v["visit_id"] in missed]
    if mv:
        ax.plot([v["start_s"] for v in mv], [0] * len(mv), "x", color="#b3402a", ms=8, label=f"missed entries ({len(mv)})")
    ax.set_xlabel("entry time in the stream (s)")
    ax.set_ylabel("stream seconds")
    ax.set_title(_title(res, "Time-to-post"), fontsize=10)
    ax.grid(alpha=0.3); ax.legend(fontsize=8, loc="upper left")
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)
