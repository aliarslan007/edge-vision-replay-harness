"""Synthetic test stream with known ground truth (stand-in until P1's composer is ready).

Each 'person' is a coloured block (upper colour over lower colour) that crosses the frame
during its entry window. Includes a consecutive repeat by the same subject and a lighting
hard negative. Writes <out>.mp4, <out>.gt.json, <out>.labels.json and a frame-list version.
"""
import json
import sys
from pathlib import Path

import cv2
import numpy as np

BGR = {"red": (40, 40, 220), "blue": (200, 80, 30), "green": (60, 170, 60), "black": (20, 20, 20),
       "white": (235, 235, 235), "grey": (128, 128, 128), "yellow": (40, 220, 230)}

ENTRIES = [  # subject, portal, start, end
    ("0001", "P1", 5.0, 8.0), ("0002", "P1", 14.0, 17.0), ("0002", "P1", 19.0, 22.0),   # repeat pass
    ("0003", "P1", 28.0, 31.0), ("0001", "P1", 37.0, 40.0),                             # returns after someone else
    ("0004", "P1", 46.0, 49.0), ("0005", "P1", 53.0, 56.0)]
LABELS = {
    "0001_P1": {"upper_color": "red", "lower_color": "black", "outerwear": "no", "bag": "yes", "glasses": "no"},
    "0002_P1": {"upper_color": "blue", "lower_color": "grey", "outerwear": "yes", "bag": "no", "glasses": "no"},
    "0003_P1": {"upper_color": "green", "lower_color": "black", "outerwear": "no", "bag": "no", "glasses": "yes"},
    "0004_P1": {"upper_color": "white", "lower_color": "blue", "outerwear": "no", "bag": "no", "glasses": "no"},
    "0005_P1": {"upper_color": "yellow", "lower_color": "black", "outerwear": "yes", "bag": "yes", "glasses": "no"},
}
HARD_NEG = [{"start_s": 32.0, "end_s": 34.0, "type": "lighting"}]


def make(out_stem: str, duration=60.0, fps=30, w=320, h=240, write_frames=True):
    out = Path(out_stem); out.parent.mkdir(parents=True, exist_ok=True)
    vw = cv2.VideoWriter(str(out.with_suffix(".mp4")), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    fdir = out.parent / (out.name + "_frames")
    if write_frames:
        fdir.mkdir(exist_ok=True)
    lst = []
    n = int(duration * fps)
    for i in range(n):
        t = i / fps
        img = np.full((h, w, 3), 90, np.uint8)
        for hn in HARD_NEG:
            if hn["start_s"] <= t <= hn["end_s"]:
                img[:] = 170                                   # sudden lighting change, nobody there
        for s, p, a, b in ENTRIES:
            if a <= t <= b:
                lab = LABELS[f"{s}_{p}"]
                x = int((t - a) / (b - a) * (w - 60))
                cv2.rectangle(img, (x, 40), (x + 50, 120), BGR[lab["upper_color"]], -1)
                cv2.rectangle(img, (x, 120), (x + 50, 200), BGR[lab["lower_color"]], -1)
        cv2.putText(img, f"{t:6.2f}s", (5, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
        vw.write(img)
        if write_frames:
            fp = fdir / f"{i:06d}.jpg"
            cv2.imwrite(str(fp), img)
            lst.append(f"{fdir.name}/{fp.name}")
    vw.release()
    if write_frames:
        out.with_suffix(".frames.txt").write_text("\n".join(lst) + "\n")
    gt = {"stream_id": out.name, "duration_s": duration, "native_fps": fps,
          "entries": [{"entry_id": k, "subject_id": s, "portal": p, "start_s": a, "end_s": b, "second_person": False}
                      for k, (s, p, a, b) in enumerate(ENTRIES)],
          "hard_negatives": HARD_NEG, "crowded": False,
          "composition": {"note": "synthetic test stream", "repeat_passes": 1}}
    out.with_suffix(".gt.json").write_text(json.dumps(gt, indent=1))
    out.with_suffix(".labels.json").write_text(json.dumps(LABELS, indent=1))
    return out


if __name__ == "__main__":
    make(sys.argv[1] if len(sys.argv) > 1 else "streams/synthetic")
