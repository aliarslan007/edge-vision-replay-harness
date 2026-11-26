"""Scores one replay run (event log) against the stream's ground truth and attribute labels.

Definitions (agreed in INTERFACES.md):
  visit          consecutive ground-truth entries by the SAME subject merged into one visit; a repeat
                 pass must not create a second record, and a person logged again after someone else is fine
  match          a valid record belongs to the visit whose window [start - pre_s, end + post_s]
                 contains the record's frame_media_ts (the moment the described frame was taken)
  entry recall   visits with >= 1 valid record / all visits
  duplicate      every valid record of a visit after its first
  false record   a valid record that matches no visit (e.g. a hard negative, or nobody there)
  invalid output VLM output that did not parse; counted separately (never a hit)
  time-to-post   post time of the visit's first record - visit start, in STREAM seconds
                 (= wall seconds x speed). Also split into capture delay (start -> frame) and
                 pipeline delay (frame -> post).
  attribute acc. on each visit's first record: correct / labelled attributes; "unknown" = abstain (wrong)
  hallucination  presence attributes (outerwear, bag, glasses) labelled "no" but asserted "yes",
                 / presence attributes labelled "no"
"""
from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path
from statistics import median
from typing import Optional

import numpy as np

from .schema import ATTRS, PRESENCE_ATTRS, UNKNOWN


def load_json(path):
    return json.loads(Path(path).read_text())


def build_visits(entries: list[dict]) -> list[dict]:
    visits = []
    for e in sorted(entries, key=lambda e: e["start_s"]):
        if visits and visits[-1]["subject_id"] == e["subject_id"]:
            v = visits[-1]
            v["end_s"] = max(v["end_s"], e["end_s"])
            v["entry_ids"].append(e.get("entry_id"))
            v["n_passes"] += 1
        else:
            visits.append({"visit_id": len(visits), "subject_id": e["subject_id"], "portal": e.get("portal"),
                           "start_s": e["start_s"], "end_s": e["end_s"], "entry_ids": [e.get("entry_id")],
                           "n_passes": 1})
    return visits


def _pct(v, q):
    return float(np.percentile(v, q)) if v else None


def score(events: list[dict], gt: dict, labels: Optional[dict] = None,
          pre_s: float = 1.0, post_s: float = 1.0, attrs_on: str = "first") -> dict:
    by = defaultdict(list)
    for e in events:
        by[e["ev"]].append(e)
    rs = by["run_start"][0]
    t0, speed, fps = rs["wall_ts"], rs["speed"], rs["sample_fps"]
    se = by["stream_end"][0] if by["stream_end"] else {}
    run_end = by["run_end"][0]["wall_ts"] if by["run_end"] else max(e["wall_ts"] for e in events)

    # only score the part of the stream that was actually served
    media_end = se.get("media_end") or gt.get("duration_s")
    if gt.get("duration_s"):
        media_end = min(media_end, gt["duration_s"])
    visits = [v for v in build_visits(gt["entries"]) if v["start_s"] < media_end]
    hours = media_end / 3600.0

    # ---------------- records -> visits ----------------
    records = sorted(by["record"], key=lambda r: r["post_wall_ts"])
    valid = [r for r in records if r["parse_ok"] and r["person"] is not None]
    n_invalid_outputs = sum(1 for o in by["vlm_output"] if not o["parse_ok"])
    per_visit, false_recs = defaultdict(list), []
    for r in valid:
        ts = r["frame_media_ts"]
        cands = [v for v in visits if v["start_s"] - pre_s <= ts <= v["end_s"] + post_s]
        if not cands:
            false_recs.append(r)
            continue
        v = min(cands, key=lambda v: 0 if v["start_s"] <= ts <= v["end_s"] else
                min(abs(ts - v["start_s"]), abs(ts - v["end_s"])))
        per_visit[v["visit_id"]].append(r)

    detected = [v for v in visits if per_visit.get(v["visit_id"])]
    missed = [v for v in visits if not per_visit.get(v["visit_id"])]
    dups = [(vid, r) for vid, rs_ in per_visit.items() for r in rs_[1:]]

    # false records inside hard-negative windows, by type
    hn = Counter()
    for r in false_recs:
        for h in gt.get("hard_negatives", []):
            if h["start_s"] <= r["frame_media_ts"] <= h["end_s"]:
                hn[h.get("type", "hard_negative")] += 1
                break

    # ---------------- time-to-post ----------------
    ttp = []
    for v in detected:
        first = per_visit[v["visit_id"]][0]
        ttp.append({"visit_id": v["visit_id"], "subject_id": v["subject_id"], "start_s": v["start_s"],
                    "ttp_media_s": first["post_media_ts"] - v["start_s"],
                    "capture_delay_s": first["frame_media_ts"] - v["start_s"],
                    "pipeline_delay_media_s": first["post_media_ts"] - first["frame_media_ts"],
                    "pipeline_delay_wall_s": first["post_wall_ts"] - first["frame_wall_ts"]})
    tt = [x["ttp_media_s"] for x in ttp]
    growth = None
    if len(ttp) >= 4:
        xs = np.array([x["start_s"] for x in ttp]); ys = np.array(tt)
        slope = float(np.polyfit(xs, ys, 1)[0])
        third = max(1, len(ys) // 3)
        growth = {"slope_s_per_s": slope,
                  "first_third_median_s": float(np.median(ys[:third])),
                  "last_third_median_s": float(np.median(ys[-third:])),
                  "looks_unbounded": bool(slope > 0.05 and np.median(ys[-third:]) > 2 * np.median(ys[:third]))}

    # ---------------- attributes ----------------
    attr = None
    if labels is not None:
        n_lab = n_ok = n_abst = 0
        per_attr = {a: [0, 0] for a in ATTRS}
        h_num = h_den = 0
        unlabelled = []
        for v in detected:
            key = f"{v['subject_id']}_{v['portal']}" if v.get("portal") else str(v["subject_id"])
            lab = labels.get(key)
            if lab is None:
                unlabelled.append(key)
                continue
            recs = per_visit[v["visit_id"]] if attrs_on == "all" else per_visit[v["visit_id"]][:1]
            for r in recs:
                p = r["person"]
                for a in ATTRS:
                    truth = lab.get(a)
                    if truth in (None, UNKNOWN):
                        continue
                    pred = p.get(a, UNKNOWN)
                    n_lab += 1; per_attr[a][1] += 1
                    if pred == truth:
                        n_ok += 1; per_attr[a][0] += 1
                    elif pred == UNKNOWN:
                        n_abst += 1
                    if a in PRESENCE_ATTRS and truth == "no":
                        h_den += 1
                        h_num += pred == "yes"
        attr = {"attribute_accuracy": n_ok / n_lab if n_lab else None, "n_labelled_attrs": n_lab,
                "abstain_rate": n_abst / n_lab if n_lab else None,
                "per_attribute_accuracy": {a: (c / t if t else None) for a, (c, t) in per_attr.items()},
                "hallucination_rate": h_num / h_den if h_den else None,
                "hallucinations": h_num, "absent_cases": h_den, "visits_without_labels": unlabelled}

    # ---------------- frames, drops, VLM, queue ----------------
    drops = Counter(d.get("reason", "busy") for d in by["drop"])
    n_frames = len(by["frame"])
    starts = {e["call_id"]: e["wall_ts"] for e in by["vlm_start"]}
    calls = [(starts[e["call_id"]], e["wall_ts"], e) for e in by["vlm_end"] if e["call_id"] in starts]
    busy = sum(b - a for a, b, _ in calls)
    run_wall = run_end - t0
    numeric = defaultdict(list)
    for _, _, e in calls:
        for k, val in e.items():
            if k not in ("ev", "wall_ts", "call_id") and isinstance(val, (int, float)) and not isinstance(val, bool):
                numeric[k].append(val)
    q = sorted(by["queue"], key=lambda e: e["wall_ts"])
    q_series = [((e["wall_ts"] - t0) * speed, e["length"]) for e in q]
    q_avg = None
    if len(q) >= 2:
        area = sum(q[i]["length"] * (q[i + 1]["wall_ts"] - q[i]["wall_ts"]) for i in range(len(q) - 1))
        q_avg = area / (q[-1]["wall_ts"] - q[0]["wall_ts"]) if q[-1]["wall_ts"] > q[0]["wall_ts"] else None

    res = {
        "run": {"speed": speed, "sample_fps": fps, "scored_stream_s": media_end, "run_wall_s": run_wall,
                "pipeline": rs.get("pipeline"), "source": rs.get("source"), "meta": rs.get("meta"),
                "warning": None if speed == 1 else f"replayed at {speed}x: report time-to-post and drops from a 1x run"},
        "entries": {"n_entries": len(gt["entries"]), "n_visits": len(visits),
                    "n_repeat_passes_merged": sum(v["n_passes"] - 1 for v in visits),
                    "entry_recall": len(detected) / len(visits) if visits else None,
                    "missed_visits": [v["visit_id"] for v in missed]},
        "records": {"n_records": len(records), "n_valid": len(valid),
                    "n_invalid_outputs": n_invalid_outputs,
                    "invalid_output_rate": n_invalid_outputs / len(by["vlm_output"]) if by["vlm_output"] else None,
                    "duplicates": len(dups), "false_records": len(false_recs),
                    "duplicates_per_hour": len(dups) / hours if hours else None,
                    "false_per_hour": len(false_recs) / hours if hours else None,
                    "false_in_hard_negatives": dict(hn),
                    "meets_target_2_per_hour": (len(dups) / hours <= 2 and len(false_recs) / hours <= 2) if hours else None},
        "end_of_stream": {"drain_timeout_s": se.get("drain_timeout_s"),
                          "records_posted_after_stream_end": sum(1 for r in valid if r["post_wall_ts"] > se.get("wall_ts", float("inf"))),
                          "frames_unfinished_at_end": sum(1 for d in by["drop"] if d.get("reason") == "unfinished_at_end"),
                          "drain_wall_s": (run_end - se["wall_ts"]) if se else None},
        "time_to_post": {"n": len(tt), "median_s": median(tt) if tt else None, "p90_s": _pct(tt, 90),
                         "max_s": max(tt) if tt else None, "mean_s": float(np.mean(tt)) if tt else None,
                         "median_wall_s": median(tt) / speed if tt else None,
                         "median_capture_delay_s": median([x["capture_delay_s"] for x in ttp]) if ttp else None,
                         "median_pipeline_delay_media_s": median([x["pipeline_delay_media_s"] for x in ttp]) if ttp else None,
                         "growth": growth},
        "attributes": attr,
        "frames": {"offered": n_frames, "dropped": sum(drops.values()), "drops_by_reason": dict(drops),
                   "drop_rate": sum(drops.values()) / n_frames if n_frames else None,
                   "late_frames": se.get("n_late")},
        "vlm": {"calls": len(calls), "duty_cycle": busy / run_wall if run_wall > 0 else None,
                "truncated_calls": sum(1 for _, _, e in calls if e.get("truncated")),
                "failed_calls": sum(1 for _, _, e in calls if e.get("error")),
                "mean_call_wall_s": busy / len(calls) if calls else None,
                "per_call_means": {k: float(np.mean(v)) for k, v in numeric.items()}},
        "queue": {"max_length": max((l for _, l in q_series), default=0), "time_avg_length": q_avg},
        "_detail": {"ttp": ttp, "queue_series": q_series,
                    "duplicates": [{"visit_id": vid, "record_id": r["record_id"], "frame_media_ts": r["frame_media_ts"]} for vid, r in dups],
                    "false_records": [{"record_id": r["record_id"], "frame_media_ts": r["frame_media_ts"]} for r in false_recs],
                    "visits": visits},
    }
    return res


def summary_line(res: dict) -> str:
    e, r, t, a, f, v = res["entries"], res["records"], res["time_to_post"], res["attributes"] or {}, res["frames"], res["vlm"]
    fmt = lambda x, p=2: "n/a" if x is None else (f"{x:.{p}f}" if isinstance(x, float) else str(x))
    return (f"[{res['run']['speed']}x, {res['run']['sample_fps']} fps] recall {fmt(e['entry_recall'])} | "
            f"dup/h {fmt(r['duplicates_per_hour'],1)} false/h {fmt(r['false_per_hour'],1)} invalid {r['n_invalid_outputs']} | "
            f"TTP median {fmt(t['median_s'],1)} s p90 {fmt(t['p90_s'],1)} s | attr acc {fmt(a.get('attribute_accuracy'))} "
            f"halluc {fmt(a.get('hallucination_rate'))} | dropped {f['dropped']}/{f['offered']} | duty {fmt(v['duty_cycle'])} | "
            f"max queue {res['queue']['max_length']}")
