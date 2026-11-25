#!/usr/bin/env python3
"""Replay a stream (or the live camera) through a pipeline and log everything.

Examples
  # stub VLM (no model) on a composed test stream, 4x speed, 2 fps sampling, no queue
  python run_replay.py --source streams/test_01.mp4 --gt streams/test_01.gt.json --labels labels.json \
      --pipeline stub --service-s 3 --capacity 0 --sample-fps 2 --speed 4 --out runs/stub_cap0
  # baseline: llama-server VLM, one full frame every 30 s, unbounded queue, no gating, real time
  python run_replay.py --source streams/test_01.mp4 --gt streams/test_01.gt.json --labels labels.json \
      --pipeline llama --llama-url http://127.0.0.1:8080 --interval-s 30 --capacity none --speed 1 --out runs/baseline
  # live camera (USB index 0, or an RTSP / MJPEG URL)
  python run_replay.py --source camera:0 --pipeline llama --sample-fps 1 --max-duration 60 --out runs/live
"""
import argparse, json, sys
from pathlib import Path

from harness import EventLog, open_source, run_replay, StubVLM, LlamaServerVLM, read_log, score, summary_line, build_visits
from harness.schema import ATTRS
from harness.plots import plot_queue, plot_ttp


def oracle_from_gt(gt, labels):
    """Stub helper: the persons truly visible at media_ts, with their labelled attributes."""
    def visible(ts):
        out = []
        for e in gt["entries"]:
            if e["start_s"] <= ts <= e["end_s"]:
                lab = (labels or {}).get(f"{e['subject_id']}_{e.get('portal')}", {})
                out.append({a: lab.get(a, "unknown") for a in ATTRS})
        return out
    return visible


def oracle_gate(gt):
    """Stub helper: a perfect detector + tracker; passes only the first sampled frame of each visit."""
    visits, seen = build_visits(gt["entries"]), set()
    def gate(frame):
        for v in visits:
            if v["start_s"] <= frame.media_ts <= v["end_s"]:
                if v["visit_id"] in seen:
                    return False
                seen.add(v["visit_id"]); return True
        return False
    return gate


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", required=True, help="video file, image folder, frame-list .txt, camera:N, rtsp://, http://")
    ap.add_argument("--native-fps", type=float, default=None, help="override the source frame rate (image folders default to 30)")
    rate = ap.add_mutually_exclusive_group(required=True)
    rate.add_argument("--sample-fps", type=float, help="frames per STREAM second handed to the pipeline (e.g. 2)")
    rate.add_argument("--interval-s", type=float, help="one frame every N stream seconds (baseline: 30)")
    ap.add_argument("--speed", type=float, default=1.0, help="replay speed-up (1, 2, 4); camera must be 1")
    ap.add_argument("--max-duration", type=float, default=None, help="stop after this many stream seconds")
    ap.add_argument("--drain-timeout", default="inf",
                    help="after the stream ends, wait this many WALL seconds for queued frames to be described "
                         "('inf' = wait until the queue is empty; frames still queued after the timeout are "
                         "logged as dropped 'unfinished_at_end' and their entries count as missed)")
    ap.add_argument("--pipeline", choices=["stub", "llama"], default="stub")
    ap.add_argument("--capacity", default="0", help="queue capacity: 'none' (unbounded), 0 (no queue) or N")
    ap.add_argument("--drop-policy", choices=["reject_new", "drop_oldest"], default="reject_new")
    ap.add_argument("--service-s", type=float, default=2.0, help="stub: wall seconds per VLM call")
    ap.add_argument("--oracle-gate", action="store_true", help="stub: perfect gating (one VLM call per visit)")
    ap.add_argument("--llama-url", default="http://127.0.0.1:8080")
    ap.add_argument("--resize", type=int, default=None, help="llama: resize long side to N px before sending")
    ap.add_argument("--max-tokens", type=int, default=512)
    ap.add_argument("--gt", default=None, help="stream ground truth (needed for scoring and for the stub oracle)")
    ap.add_argument("--labels", default=None, help="attribute labels from P4")
    ap.add_argument("--out", required=True, help="run folder (events.jsonl, metrics.json, plots)")
    ap.add_argument("--tag", default="", help="free text stored with the run, e.g. 'baseline' or lineage step name")
    a = ap.parse_args(argv)

    cap = None if a.capacity.lower() == "none" else int(a.capacity)
    sample_fps = a.sample_fps if a.sample_fps else 1.0 / a.interval_s
    drain = float(a.drain_timeout)
    gt = json.loads(Path(a.gt).read_text()) if a.gt else None
    labels = json.loads(Path(a.labels).read_text()) if a.labels else None
    if a.pipeline == "stub":
        pipe = StubVLM(service_s=a.service_s, oracle=oracle_from_gt(gt, labels) if gt else None,
                       capacity=cap, drop_policy=a.drop_policy, gate=oracle_gate(gt) if (a.oracle_gate and gt) else None)
    else:
        pipe = LlamaServerVLM(url=a.llama_url, resize_long_side=a.resize, max_tokens=a.max_tokens,
                              capacity=cap, drop_policy=a.drop_policy)

    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    src = open_source(a.source, a.native_fps)
    log = EventLog(out / "events.jsonl")
    try:
        info = run_replay(src, pipe, log, sample_fps=sample_fps, speed=a.speed, max_duration_s=a.max_duration,
                          drain_timeout_s=drain,
                          meta={"tag": a.tag, "argv": sys.argv[1:], "gt": a.gt, "labels": a.labels,
                                "interval_s": a.interval_s, "drain_timeout_s": a.drain_timeout})
    finally:
        log.close(); src.close()
    print("replay finished:", info)
    if gt:
        res = score(read_log(out / "events.jsonl"), gt, labels)
        (out / "metrics.json").write_text(json.dumps({k: v for k, v in res.items() if k != "_detail"}, indent=2))
        plot_queue(res, out / "queue_length.png"); plot_ttp(res, out / "time_to_post.png")
        print(summary_line(res))
    return 0


if __name__ == "__main__":
    sys.exit(main())
