# A2 replay harness (P3)

Serves recorded streams through the same interface as the live camera, paces them by wall clock at the sampled
rate without waiting for the pipeline, drops and logs frames that arrive while it is busy, timestamps every frame and
record, and scores a run against ground truth and attribute labels. Interfaces: [INTERFACES.md](INTERFACES.md).

## Layout
```
harness/
  schema.py     attribute vocabulary, Frame, VLM-output parser
  sources.py    VideoFileSource, ImageSequenceSource (folder / frame list), CameraSource (USB, RTSP, MJPEG)
  replay.py     wall-clock pacer, speed-up, PipelineContext (what pipelines report through)
  pipelines.py  QueuedVLMPipeline (queue + drop policy + gate), StubVLM (no model), LlamaServerVLM
  events.py     JSONL event log
  scoring.py    all per-change metrics
  plots.py      queue length and time-to-post plots
run_replay.py   run a stream or camera through a pipeline (scores it if --gt is given)
score_run.py    re-score an existing run
tests/          synthetic stream generator + end-to-end tests with known answers
```

## Setup (Pi 5 or laptop)
```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt      # opencv-python-headless, numpy, matplotlib (Python 3.9+)
python tests/test_harness.py         # 10 tests, about 2 min; must print 10/10 passed
```

## Typical runs
```bash
# 1. Harness only, no model: synthetic or composed stream, stub VLM taking 3 s per call
python run_replay.py --source streams/test_01.frames.txt --gt streams/test_01.gt.json --labels labels.json \
    --pipeline stub --service-s 3 --capacity 0 --sample-fps 2 --speed 1 --out runs/stub_1x

# 2. Baseline (P2): llama-server running Qwen3.5-2B (BF16 decoder, F16 mmproj), one full frame every 30 s,
#    unbounded queue, no gating, real time, wait for the queue to empty at the end
python run_replay.py --source streams/test_01.frames.txt --gt streams/test_01.gt.json --labels labels.json \
    --pipeline llama --llama-url http://127.0.0.1:8080 --interval-s 30 --capacity none --speed 1 \
    --drain-timeout inf --tag baseline --out runs/baseline_1x

# 3. Live camera through the same interface (P4)
python run_replay.py --source camera:0 --pipeline llama --sample-fps 1 --capacity 0 --max-duration 120 --out runs/live

# Re-score a run (e.g. after a scoring fix) without re-running it
python score_run.py --log runs/baseline_1x/events.jsonl --gt streams/test_01.gt.json --labels labels.json
```
Each run folder gets `events.jsonl` (everything that happened), `metrics.json`, `queue_length.png`, `time_to_post.png`.

## Settings that matter

| Flag | Meaning |
|---|---|
| `--interval-s 30` or `--sample-fps 2` | how often a frame is handed to the pipeline (stream time) |
| `--capacity none / 0 / N` | queue: unbounded (baseline) / no queue (drop while busy) / bounded |
| `--drop-policy reject_new / drop_oldest` | what a full bounded queue does |
| `--speed 1 / 2 / 4` | replay speed-up; time-to-post and drops must come from 1x |
| `--drain-timeout inf / N` | end-of-stream behaviour (below) |

## How it calls the VLM (`--pipeline llama`)

`harness/pipelines.py: LlamaServerVLM` sends each frame to a running llama.cpp `llama-server`
(started with the model GGUF and `--mmproj`) through its OpenAI-compatible endpoint:

```
POST http://127.0.0.1:8080/v1/chat/completions
{"messages": [{"role": "user", "content": [
     {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,<frame as JPEG>"}},
     {"type": "text", "text": "<record prompt>"}]}],
 "max_tokens": 512, "temperature": 0}
```

- The frame is sent at native resolution unless `--resize N` is given (long side in pixels).
- It returns the model's text (any `reasoning_content`, i.e. thinking, is kept inside `<think>` tags for logging)
  plus llama-server's timings (`prompt_n`, `prompt_ms`, `predicted_n`, `predicted_ms`) and the HTTP round trip,
  which are logged with every call.
- Calls run one at a time on a worker thread behind the queue. Chat-template options (e.g. turning thinking off)
  go in the `extra` request fields, or are set when llama-server is started.
- A different VLM runner plugs in by subclassing `QueuedVLMPipeline` and implementing
  `describe_frame(frame) -> raw_text` (INTERFACES.md §6).

**Parsing** (`harness/schema.py: parse_vlm_output`): drops `<think>…</think>`, then finds the first `[...]` block
that is valid JSON and looks like a list of persons (so a sentence before the JSON, or stray brackets such as
"[image 1]", are skipped), falling back to a `{...}` object. Output with no usable JSON is logged as one
*invalid record* and counted, never silently dropped.

## End of the stream

When the last frame has been served, the harness stops offering frames and waits for the pipeline to finish:

- `--drain-timeout inf` (default): waits until the queue is empty and the last call has finished. Every queued frame is
  described; records posted after the stream ended are counted (`end_of_stream.records_posted_after_stream_end`) and
  keep their full time-to-post. Use this for the baseline, so a slow pipeline shows up as long time-to-post rather
  than as missed entries.
- `--drain-timeout N`: waits at most N wall seconds. Frames still queued are logged as dropped
  (`unfinished_at_end`), and any entry that never got a record counts as **missed** in entry recall.

Either way, an entry with no posted record by the end of the run is a missed entry; `metrics.json` reports how
many frames were left unfinished and how long the drain took.

## Notes for the Pi
- The harness decodes video and competes for CPU with the VLM. Pin it away from the VLM's cores, e.g.
  `taskset -c 0 python run_replay.py ...` with llama-server on cores 1–3, and record the pinning with the run (`--tag`).
- `late_frames` in metrics.json counts frames the harness itself served late; it should be 0. If not, the harness is
  starved of CPU and the run's drop and time-to-post numbers are not trustworthy.
- Report time-to-post and dropped frames from **1×** runs (INTERFACES.md §9).
