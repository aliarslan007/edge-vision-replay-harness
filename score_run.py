#!/usr/bin/env python3
"""Score an existing run: python score_run.py --log runs/x/events.jsonl --gt stream.gt.json [--labels labels.json]"""
import argparse, json
from pathlib import Path
from harness import read_log, score, summary_line
from harness.plots import plot_queue, plot_ttp

ap = argparse.ArgumentParser()
ap.add_argument("--log", required=True); ap.add_argument("--gt", required=True); ap.add_argument("--labels")
ap.add_argument("--pre-s", type=float, default=1.0, help="match window before an entry starts")
ap.add_argument("--post-s", type=float, default=1.0, help="match window after an entry ends")
ap.add_argument("--attrs-on", choices=["first", "all"], default="first")
ap.add_argument("--out", default=None, help="folder for metrics.json and plots (default: next to the log)")
a = ap.parse_args()
res = score(read_log(a.log), json.loads(Path(a.gt).read_text()),
            json.loads(Path(a.labels).read_text()) if a.labels else None, a.pre_s, a.post_s, a.attrs_on)
out = Path(a.out or Path(a.log).parent); out.mkdir(parents=True, exist_ok=True)
(out / "metrics.json").write_text(json.dumps({k: v for k, v in res.items() if k != "_detail"}, indent=2))
(out / "details.json").write_text(json.dumps(res["_detail"], indent=1))
plot_queue(res, out / "queue_length.png"); plot_ttp(res, out / "time_to_post.png")
print(summary_line(res))
