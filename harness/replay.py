"""Wall-clock replay: serves frames at the sampled rate WITHOUT waiting for the pipeline.

Timeline (speed s, sampled rate f):
    frame k has stream time      media_t = k / f
    and is due at wall time      t0 + media_t / s
The harness decodes frame k early, sleeps until it is due, then offers it to the pipeline
with a non-blocking call. If the pipeline refuses (busy / queue full), the frame is dropped
and logged. Records posted by the pipeline are stamped on arrival in wall and stream time:
    post_media_ts = (post_wall_ts - t0) * s
so time-to-post is always comparable with the ground-truth entry times (stream seconds).
"""
from __future__ import annotations

import itertools
import platform
import threading
import time
from typing import Optional

from .events import EventLog, now
from .schema import Frame, parse_vlm_output
from .sources import FrameSource


class PipelineContext:
    """What the harness gives the pipeline: one place to post results and report state."""

    def __init__(self, log: EventLog, t0: float, speed: float):
        self.log, self.t0, self.speed = log, t0, speed
        self._rec_ids = itertools.count()
        self._call_ids = itertools.count()
        self._lock = threading.Lock()

    def media_now(self) -> float:
        return (now() - self.t0) * self.speed

    # ---- VLM activity (for duty cycle and per-stage timing) ----
    def vlm_start(self, frame: Frame) -> int:
        with self._lock:
            cid = next(self._call_ids)
        self.log.write("vlm_start", call_id=cid, frame_id=frame.frame_id, media_ts=frame.media_ts)
        return cid

    def vlm_end(self, call_id: int, **timings):
        """timings: anything measured inside the call, e.g. image_tokens, encode_ms, prefill_ms, decode_ms."""
        self.log.write("vlm_end", call_id=call_id, **timings)

    # ---- results ----
    def post_output(self, frame: Frame, raw_text: Optional[str], persons: Optional[list] = None,
                    parse_ok: Optional[bool] = None, error: str = "", call_id: Optional[int] = None) -> list[dict]:
        """Post one VLM output. Each described person becomes one record (the log sink).
        If persons is None the raw text is parsed here with the shared schema parser.
        Unparseable output becomes ONE invalid record (person = null), so it is counted."""
        if persons is None:
            persons, parse_ok, error = parse_vlm_output(raw_text or "")
        t = now()
        common = dict(frame_id=frame.frame_id, frame_media_ts=frame.media_ts, frame_wall_ts=frame.wall_ts,
                      post_wall_ts=t, post_media_ts=(t - self.t0) * self.speed, call_id=call_id)
        self.log.write("vlm_output", wall_ts=t, n_persons=len(persons), parse_ok=bool(parse_ok), error=error,
                       raw=(raw_text or "")[:4000], **common)
        out = []
        if not parse_ok:
            persons = [None]
        for i, p in enumerate(persons):
            with self._lock:
                rid = next(self._rec_ids)
            out.append(self.log.write("record", wall_ts=t, record_id=rid, person_index=i, person=p,
                                      parse_ok=bool(parse_ok), **common))
        return out

    # ---- queue / drops / anything else ----
    def queue_len(self, n: int):
        self.log.write("queue", length=n, media_ts=self.media_now())

    def drop(self, frame: Frame, reason: str):
        self.log.write("drop", frame_id=frame.frame_id, media_ts=frame.media_ts, reason=reason)

    def event(self, name: str, **fields):
        self.log.write(name, **fields)


def run_replay(source: FrameSource, pipeline, log: EventLog, sample_fps: float, speed: float = 1.0,
               max_duration_s: Optional[float] = None, drain_timeout_s: float = 120.0,
               meta: Optional[dict] = None) -> dict:
    """Serve `source` to `pipeline` at `sample_fps` (stream time) and `speed`x wall clock."""
    if source.is_live and speed != 1.0:
        raise ValueError("a live camera can only run at speed 1.0")
    if sample_fps <= 0 or speed <= 0:
        raise ValueError("sample_fps and speed must be positive")

    t0 = now()
    ctx = PipelineContext(log, t0, speed)
    log.write("run_start", wall_ts=t0, sample_fps=sample_fps, speed=speed, source=source.describe(),
              pipeline=pipeline.describe(), host=platform.node(), python=platform.python_version(),
              max_duration_s=max_duration_s, meta=meta or {})
    pipeline.start(ctx)

    period = 1.0 / sample_fps
    k = n_dropped = n_late = 0
    try:
        while True:
            media_t = k * period
            if max_duration_s is not None and media_t >= max_duration_s:
                break
            due = t0 + media_t / speed
            if source.is_live:
                _sleep_until(due)
                got = source.read(media_t)
            else:
                got = source.read(media_t)          # decode ahead of time ...
                _sleep_until(due)                   # ... then wait until the frame is due
            if got is None:
                break
            img, mts = got
            emit = now()
            late = emit - due
            if late > period / speed:               # the harness itself could not keep up
                n_late += 1
            frame = Frame(frame_id=k, media_ts=mts, wall_ts=emit, image=img, source_id=str(source.describe().get("path", "")))
            accepted = bool(pipeline.offer(frame))   # must not block
            log.write("frame", wall_ts=emit, frame_id=k, media_ts=mts, due_wall_ts=due, late_s=round(late, 4),
                      accepted=accepted)
            if not accepted:
                n_dropped += 1
                log.write("drop", frame_id=k, media_ts=mts, reason="busy")
            k += 1
    finally:
        log.write("stream_end", n_frames=k, n_dropped=n_dropped, n_late=n_late, media_end=k * period,
                  stream_end_media_ts=(now() - t0) * speed, drain_timeout_s=drain_timeout_s)
        drained = pipeline.drain(drain_timeout_s)
        pipeline.stop()
        log.write("run_end", drained=drained)
    return {"frames": k, "dropped": n_dropped, "late": n_late, "drained": drained}


def _sleep_until(t: float):
    while True:
        dt = t - now()
        if dt <= 0:
            return
        time.sleep(min(dt, 0.05))
