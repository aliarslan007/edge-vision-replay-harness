"""End-to-end tests with known answers. Run:  python tests/test_harness.py   (about 1.5 minutes)"""
import json
import statistics
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "tests"))

from harness import (EventLog, StubVLM, build_visits, open_source, parse_vlm_output, read_log, run_replay, score,
                     summary_line)
from make_synthetic import make
import run_replay as cli

TMP = Path(tempfile.mkdtemp(prefix="a2h_"))
STREAM = make(str(TMP / "synthetic"))
GT = json.loads(STREAM.with_suffix(".gt.json").read_text())
LABELS = json.loads(STREAM.with_suffix(".labels.json").read_text())
MP4, GTP, LABP = str(STREAM.with_suffix(".mp4")), str(STREAM.with_suffix(".gt.json")), str(STREAM.with_suffix(".labels.json"))


def metrics(run):
    return json.loads((TMP / run / "metrics.json").read_text())


def test_parser():
    p, ok, _ = parse_vlm_output('<think>let me look</think>```json\n{"persons":[{"upper_color":"Navy","bag":true}]}\n```')
    assert ok and p[0]["upper_color"] == "blue" and p[0]["bag"] == "yes" and p[0]["glasses"] == "unknown"
    p, ok, _ = parse_vlm_output("[]")
    assert ok and p == []
    p, ok, err = parse_vlm_output('{"persons": [{"upper_color": "red"')         # truncated output
    assert not ok, err
    p, ok, _ = parse_vlm_output("<think>still thinking when max_tokens ran out")
    assert not ok
    # a sentence (with stray brackets) before the JSON list: the first VALID [...] block is used
    p, ok, _ = parse_vlm_output('I can see one person [image 1]. Here: [{"upper_color": "red", "glasses": "no"}]')
    assert ok and len(p) == 1 and p[0]["upper_color"] == "red" and p[0]["glasses"] == "no"


def test_visits():
    v = build_visits(GT["entries"])
    assert len(GT["entries"]) == 7 and len(v) == 6                             # 0002's repeat pass merged
    assert v[1]["subject_id"] == "0002" and v[1]["n_passes"] == 2 and (v[1]["start_s"], v[1]["end_s"]) == (14.0, 22.0)
    assert v[0]["subject_id"] == v[3]["subject_id"] == "0001"                  # a return after someone else stays separate


def test_perfect_gated_pipeline():
    """One VLM call per visit with a perfect describer: every metric must be exact."""
    cli.main(["--source", MP4, "--gt", GTP, "--labels", LABP, "--pipeline", "stub", "--service-s", "0.5",
              "--oracle-gate", "--capacity", "none", "--sample-fps", "4", "--speed", "4", "--out", str(TMP / "perfect")])
    m = metrics("perfect")
    assert m["entries"]["entry_recall"] == 1.0, m["entries"]
    assert m["records"]["duplicates"] == 0 and m["records"]["false_records"] == 0 and m["records"]["n_invalid_outputs"] == 0
    assert m["attributes"]["attribute_accuracy"] == 1.0 and m["attributes"]["hallucination_rate"] == 0.0
    assert m["vlm"]["calls"] == 6 and m["frames"]["dropped"] == 0
    # 0.5 wall s per call at 4x = 2.0 stream s; entries start exactly on a sampled frame
    assert abs(m["time_to_post"]["median_s"] - 2.0) < 0.25, m["time_to_post"]


def test_ungated_no_queue_gives_duplicates_and_drops():
    cli.main(["--source", MP4, "--gt", GTP, "--labels", LABP, "--pipeline", "stub", "--service-s", "0.5",
              "--capacity", "0", "--sample-fps", "4", "--speed", "4", "--out", str(TMP / "ungated")])
    m = metrics("ungated")
    assert m["entries"]["entry_recall"] == 1.0
    assert m["records"]["duplicates"] > 0 and m["frames"]["dropped"] > 0
    assert m["records"]["false_records"] == 0                                 # the stub never invents people
    assert m["frames"]["drops_by_reason"] == {"busy": m["frames"]["dropped"]}


def test_false_records_counted():
    """A describer that always reports a person creates false records in empty stretches and hard negatives."""
    src, log = open_source(MP4), EventLog(TMP / "always" / "events.jsonl")
    pipe = StubVLM(service_s=0.25, oracle=lambda ts: [{"upper_color": "red"}], capacity=0)
    run_replay(src, pipe, log, sample_fps=1, speed=4); log.close(); src.close()
    m = score(read_log(TMP / "always" / "events.jsonl"), GT, LABELS)
    assert m["records"]["false_records"] > 0 and m["records"]["false_in_hard_negatives"].get("lighting", 0) > 0


def test_unbounded_queue_detected():
    """lambda > mu with an unbounded queue: time-to-post must grow and be flagged."""
    from run_replay import oracle_from_gt
    src, log = open_source(MP4), EventLog(TMP / "unbounded" / "events.jsonl")
    pipe = StubVLM(service_s=0.4, oracle=oracle_from_gt(GT, LABELS), capacity=None)
    run_replay(src, pipe, log, sample_fps=2, speed=4, drain_timeout_s=200); log.close(); src.close()
    m = score(read_log(TMP / "unbounded" / "events.jsonl"), GT, LABELS)
    assert m["queue"]["max_length"] > 50, m["queue"]
    assert m["time_to_post"]["growth"]["looks_unbounded"], m["time_to_post"]


def test_invalid_json_counted():
    class Garbage(StubVLM):
        def describe_frame(self, frame):
            return "Sure! The person is wearing a red shirt."
    src, log = open_source(MP4), EventLog(TMP / "garbage" / "events.jsonl")
    run_replay(src, Garbage(service_s=0.1, capacity=0), log, sample_fps=1, speed=4, max_duration_s=20)
    log.close(); src.close()
    m = score(read_log(TMP / "garbage" / "events.jsonl"), GT, LABELS)
    assert m["records"]["n_invalid_outputs"] == m["vlm"]["calls"] > 0 and m["entries"]["entry_recall"] == 0.0


def test_baseline_config_interval():
    """Baseline config from the CLI: one full frame every N s, unbounded queue, no gate, all frames described."""
    cli.main(["--source", MP4, "--gt", GTP, "--labels", LABP, "--pipeline", "stub", "--service-s", "0.05",
              "--interval-s", "5", "--capacity", "none", "--speed", "4", "--out", str(TMP / "baseline_cfg")])
    ev = read_log(TMP / "baseline_cfg" / "events.jsonl")
    frames = [e for e in ev if e["ev"] == "frame"]
    assert [round(f["media_ts"], 3) for f in frames] == [5.0 * i for i in range(12)]
    assert all(f["accepted"] for f in frames) and not [e for e in ev if e["ev"] == "drop"]
    assert next(e for e in ev if e["ev"] == "run_start")["pipeline"]["capacity"] is None


def test_end_of_stream_drain_vs_timeout():
    """Backlog at the end: drain='inf' describes everything (posted after the stream ends);
    drain=0 logs leftovers as dropped 'unfinished_at_end' and their entries count as missed."""
    from run_replay import oracle_from_gt
    res = {}
    for name, drain in (("drain_inf", float("inf")), ("drain_0", 0.0)):
        src, log = open_source(MP4), EventLog(TMP / name / "events.jsonl")
        pipe = StubVLM(service_s=0.15, oracle=oracle_from_gt(GT, LABELS), capacity=None)
        run_replay(src, pipe, log, sample_fps=2, speed=4, max_duration_s=30, drain_timeout_s=drain)
        log.close(); src.close()
        res[name] = score(read_log(TMP / name / "events.jsonl"), GT, LABELS)
    a, b = res["drain_inf"], res["drain_0"]
    assert a["end_of_stream"]["frames_unfinished_at_end"] == 0 and a["end_of_stream"]["records_posted_after_stream_end"] > 0
    assert b["end_of_stream"]["frames_unfinished_at_end"] > 0
    assert b["frames"]["drops_by_reason"].get("unfinished_at_end", 0) > 0
    assert a["entries"]["entry_recall"] == 1.0 and b["entries"]["entry_recall"] < 1.0


def test_llama_pipeline_sends_extra_and_flags_truncation():
    """Fake llama-server: checks the request body (max_tokens, --extra fields, image) and that a
    finish_reason of 'length' is logged as a truncated call."""
    import http.server, threading
    seen = []

    class Fake(http.server.BaseHTTPRequestHandler):
        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            seen.append(body)
            truncated = len(seen) % 2 == 0          # every second call "runs out of tokens"
            msg = ({"content": "", "reasoning_content": "thinking " * 50} if truncated else
                   {"content": 'Here: [{"upper_color": "red"}]', "reasoning_content": "short thought"})
            out = json.dumps({"choices": [{"message": msg, "finish_reason": "length" if truncated else "stop"}],
                              "timings": {"prompt_n": 900, "prompt_ms": 4000.0, "predicted_n": 40, "predicted_ms": 3000.0},
                              "usage": {"completion_tokens": 40}}).encode()
            self.send_response(200); self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(out))); self.end_headers(); self.wfile.write(out)

        def log_message(self, *a):
            pass

    srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Fake)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    try:
        cli.main(["--source", MP4, "--gt", GTP, "--labels", LABP, "--pipeline", "llama",
                  "--llama-url", f"http://127.0.0.1:{srv.server_address[1]}", "--extra", '{"cache_prompt": false}',
                  "--interval-s", "10", "--capacity", "none", "--speed", "4", "--out", str(TMP / "fake_llama")])
    finally:
        srv.shutdown()
    b = seen[0]
    assert b["max_tokens"] == 2048 and b["cache_prompt"] is False and b["temperature"] == 0
    assert b["messages"][0]["content"][0]["image_url"]["url"].startswith("data:image/jpeg;base64,")
    m = metrics("fake_llama")
    assert m["vlm"]["calls"] == len(seen) == 6 and m["vlm"]["truncated_calls"] == 3, m["vlm"]
    assert m["records"]["n_invalid_outputs"] == 3                     # truncated -> no JSON -> invalid record
    assert m["vlm"]["per_call_means"]["prompt_n"] == 900
    ev = read_log(TMP / "fake_llama" / "events.jsonl")
    assert next(e for e in ev if e["ev"] == "run_start")["meta"]["extra"] == {"cache_prompt": False}
    out = next(e for e in ev if e["ev"] == "vlm_output" and e["parse_ok"])
    assert out["raw"].startswith("<think>short thought</think>")


def test_image_list_source_and_pacing():
    """Frame-list source gives the same frames; pacing is accurate; 1x and 4x serve the same frames."""
    counts = {}
    for speed in (1, 4):
        src = open_source(str(STREAM.with_suffix(".frames.txt")))
        log = EventLog(TMP / f"pace{speed}" / "events.jsonl")
        run_replay(src, StubVLM(service_s=0.05, capacity=0), log, sample_fps=5, speed=speed, max_duration_s=8)
        log.close()
        ev = read_log(TMP / f"pace{speed}" / "events.jsonl")
        frames = [e for e in ev if e["ev"] == "frame"]
        counts[speed] = [round(f["media_ts"], 3) for f in frames]
        late = [abs(f["late_s"]) for f in frames]
        assert statistics.median(late) < 0.02, f"pacing off at {speed}x: median late {statistics.median(late)}"
        t0 = next(e for e in ev if e["ev"] == "run_start")["wall_ts"]
        span = frames[-1]["wall_ts"] - t0
        assert abs(span - (8 - 0.2) / speed) < 0.15, (speed, span)          # 40 frames, last due at 7.8 s / speed
    assert counts[1] == counts[4] and len(counts[1]) == 40


if __name__ == "__main__":
    tests = [v for k, v in dict(globals()).items() if k.startswith("test_")]
    failed = 0
    for t in tests:
        try:
            t(); print(f"PASS {t.__name__}")
        except AssertionError as e:
            failed += 1; print(f"FAIL {t.__name__}: {e}")
    for run in ("perfect", "ungated"):
        print(f"{run:>9}: {summary_line(json.loads(json.dumps(score(read_log(TMP / run / 'events.jsonl'), GT, LABELS))))}")
    print(f"\n{len(tests) - failed}/{len(tests)} passed   (artifacts in {TMP})")
    sys.exit(1 if failed else 0)
