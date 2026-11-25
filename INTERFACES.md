# A2 interfaces (draft for the Day-1 group agreement)

Owner: P3 (replay harness). Everything below is implemented in `harness/` and covered by `tests/`.
Items marked **DECIDE** need a group decision on Day 1; the current default is shown.

---

## 1. Clocks and timestamps

| Name | Meaning | Unit |
|---|---|---|
| `media_ts` | position in the **stream** (0 = first frame of the composed stream) | seconds |
| `wall_ts` | `time.monotonic()` on the machine running the harness (never jumps) | seconds |
| `speed` | replay speed-up s; frame with stream time t is due at wall time `t0 + t / s` | × |

- A posted record gets `post_media_ts = (post_wall_ts − t0) × speed`, so time-to-post is always in **stream seconds**,
  directly comparable to the ground-truth entry times. Wall seconds = stream seconds / speed.
- The live camera always runs at speed 1; its `media_ts` is capture time since the camera was opened.

## 2. Frames (harness → pipeline)

```python
Frame(frame_id: int,      # index of SAMPLED frames in this run: 0, 1, 2, ...
      media_ts: float,    # stream time of this frame
      wall_ts: float,     # when the harness handed it over
      image: np.ndarray,  # H x W x 3, uint8, BGR (OpenCV order), native resolution, no resizing
      source_id: str)
```

Every source implements `read(media_ts) -> (image, actual_media_ts) | None`:

| Source | `--source` value | Who uses it |
|---|---|---|
| Video file (constant frame rate) | `streams/test_01.mp4` | P1's composed streams |
| Frame list (one image path per line) or a folder of images | `streams/test_01.frames.txt` | P1, if streams are spliced from the original ChokePoint JPEGs without re-encoding |
| USB camera | `camera:0` | P4 |
| Network camera | `rtsp://...` or `http://.../video` (MJPEG) | P4 (phone / IP camera) |

**DECIDE (P1):** video file or frame list. A frame list avoids re-encoding artefacts and is the default suggestion;
either works. ChokePoint native rate is **30 fps**; stream files must have a constant frame rate.

## 3. Ground truth per stream (P1 → harness): `<stream>.gt.json`

```json
{
  "stream_id": "test_01",
  "duration_s": 600.0,
  "native_fps": 30,
  "crowded": false,
  "entries": [
    {"entry_id": 0, "subject_id": "0012", "portal": "P1", "start_s": 12.00, "end_s": 15.30,
     "sequence": "P1E_S1_C1", "second_person": false}
  ],
  "hard_negatives": [{"start_s": 40.0, "end_s": 43.0, "type": "lighting"}],
  "composition": {"entry_rate_per_min": 2.0, "repeat_rate": 0.2, "seed": 0}
}
```

- `start_s` / `end_s`: stream time of the **first and last frame of the passage** (the person's visible interval).
  **DECIDE:** time-to-post is measured from `start_s`.
- `subject_id` + `portal` select the attribute label (clothing changes between portals).
- Consecutive entries with the same `subject_id` are a **repeat pass**: the scorer merges them into one *visit*
  that must produce exactly one record.
- `hard_negatives.type`: `lighting`, `shadow`, `motion`, or anything else; false records are counted per type.

## 4. Attribute labels (P4 → harness): `labels.json`

Key = `"<subject_id>_<portal>"`. One entry per subject and portal.

```json
{"0012_P1": {"upper_color": "blue", "lower_color": "black", "outerwear": "no", "bag": "yes", "glasses": "no"}}
```

| Attribute | Values |
|---|---|
| `upper_color`, `lower_color` | black, white, grey, red, orange, yellow, green, blue, purple, pink, brown |
| `outerwear`, `bag`, `glasses` | `yes` / `no` (**`no` = absent**, which is what hallucination is scored against) |

- Use `null` when the annotator cannot tell from the footage; such attributes are not scored.
- **DECIDE (P4):** this is a 5-attribute set (within the 4–6 required). Add or rename attributes here first, in
  `harness/schema.py`, so prompt, labels and scoring stay identical.

## 5. VLM output and records (P2 → harness)

The VLM is prompted for exactly this JSON:

```json
{"persons": [{"upper_color": "blue", "lower_color": "black", "outerwear": "no", "bag": "yes", "glasses": "unknown"}]}
```

- `"unknown"` is allowed for every attribute (abstaining is never a hallucination). `{"persons": []}` = nobody seen.
- The shared parser (`harness.schema.parse_vlm_output`) strips `<think>…</think>`, code fences and chatter,
  normalises values (gray → grey, navy → blue, true → yes, ...).
- **Each described person becomes one record.** Output that does not parse becomes one *invalid record*
  (`person: null`, `parse_ok: false`), so broken JSON is counted, never silently lost.

## 6. Pipeline contract (P2's code plugs in here)

```python
class MyPipeline:
    def describe(self) -> dict                    # settings logged with the run (model, quant, prompt, ...)
    def start(self, ctx) -> None                  # ctx = harness.replay.PipelineContext
    def offer(self, frame) -> bool                # MUST NOT BLOCK. False = refused -> harness drops + logs it
    def drain(self, timeout_s) -> bool            # finish queued work after the stream ends
    def stop(self) -> None
```

The pipeline reports through `ctx`:

| Call | Logs | Used for |
|---|---|---|
| `ctx.vlm_start(frame)` / `ctx.vlm_end(call_id, **timings)` | VLM busy intervals + per-call timings | duty cycle, cost per record |
| `ctx.post_output(frame, raw_text)` | one record per person | every accuracy metric, time-to-post |
| `ctx.queue_len(n)` | queue length after every change | queue-length plot |
| `ctx.drop(frame, reason)` | frames the pipeline itself discards (e.g. evicted from a bounded queue) | dropped frames |
| `ctx.event(name, **fields)` | anything else (gate decisions, detector scores, ...) | gating operating curve |

**DECIDE (P2):** names of per-call timing fields. Suggested: `image_tokens`, `encode_ms`, `prefill_ms`,
`decode_ms`, `output_tokens`, `thinking_tokens`. `LlamaServerVLM` already logs llama-server's
`prompt_n`, `prompt_ms`, `predicted_n`, `predicted_ms`.

Ready-made: `QueuedVLMPipeline(capacity=None|0|N, drop_policy="reject_new"|"drop_oldest", gate=fn)` handles the queue,
drop policy and logging; a subclass only implements `describe_frame(frame) -> raw_text`.
The baseline is `capacity=None` (unbounded, no gate) with a sample rate of one frame per 30 s.

## 7. Event log: `runs/<name>/events.jsonl`

One JSON object per line; every line has `ev` and `wall_ts`.

| `ev` | Key fields |
|---|---|
| `run_start` | `sample_fps`, `speed`, `source`, `pipeline` (its `describe()`), `meta` (tag, command line) |
| `frame` | `frame_id`, `media_ts`, `due_wall_ts`, `late_s`, `accepted` |
| `drop` | `frame_id`, `media_ts`, `reason` (`busy`, `evicted_oldest`, `unfinished_at_end`) |
| `queue` | `length`, `media_ts` |
| `vlm_start` / `vlm_end` | `call_id`, `frame_id`, timings |
| `vlm_output` | `n_persons`, `parse_ok`, `error`, `raw` (first 4,000 chars) |
| `record` | `record_id`, `frame_id`, `frame_media_ts`, `post_media_ts`, `post_wall_ts`, `person`, `parse_ok` |
| `stream_end` / `run_end` | `n_frames`, `n_dropped`, `n_late`, `drained` |

P4's power/thermal logger writes its **own** file with `time.monotonic()` timestamps on the same machine, so it
lines up with this log. **DECIDE (P4):** power samples as `{"wall_ts", "power_w", "temp_c", "throttled"}` per line.

## 8. Metric definitions

| Metric | Definition |
|---|---|
| Visit | consecutive entries by the same subject, merged |
| Match | a valid record belongs to the visit whose window `[start − 1 s, end + 1 s]` contains the record's `frame_media_ts` (**DECIDE** the 1 s tolerance) |
| Entry recall | visits with ≥ 1 valid record / all visits |
| Duplicate | every valid record of a visit after the first; reported per hour of stream |
| False record | valid record matching no visit; per hour, and per hard-negative type |
| Invalid output | VLM output that did not parse; reported separately |
| Time-to-post | first record's post time − visit start, in stream seconds; split into capture delay (start → described frame) and pipeline delay (frame → post) |
| Attribute accuracy | on each visit's first record: correct / labelled attributes; `unknown` counts as wrong and is reported as abstain rate (**DECIDE**) |
| Hallucination rate | presence attributes labelled `no` but asserted `yes` / presence attributes labelled `no` |
| Dropped frames | `drop` events / frames offered, by reason |
| VLM duty cycle | total VLM busy time / run wall time |
| Queue length | time series + max + time-averaged |
| Bounded? | slope of time-to-post against entry time; flagged if it keeps growing |

Targets from the project description: time-to-post bounded; ≤ 2 duplicates and ≤ 2 false records per hour.

## 9. Speed-up rule

Replaying at s× multiplies the arrival rate by s, while the VLM is not faster: a pipeline that keeps up at 1× can
overflow at 4×. **Every result states its speed** (it is in the log and on every plot), and **time-to-post and
dropped frames are reported from 1× runs.** Accuracy metrics can come from faster runs only if the pipeline was not
dropping frames there.
