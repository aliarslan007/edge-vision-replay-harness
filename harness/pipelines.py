"""Pipeline contract and reference implementations.

A pipeline is anything with:
    describe() -> dict            settings to log with the run
    start(ctx)                    ctx is a PipelineContext (post records, report queue / VLM state)
    offer(frame) -> bool          NON-BLOCKING. False = refused, the harness drops and logs the frame
    drain(timeout_s) -> bool      finish queued work after the stream ends; True if everything finished
    stop()

QueuedVLMPipeline puts a queue between capture and description (Task 4) with an explicit
drop policy, and an optional gate (Task 5) that decides which frames reach the VLM.
Subclasses only implement describe_frame(frame) -> raw VLM text (+ optional timings).
"""
from __future__ import annotations

import base64
import collections
import json
import threading
import time
import urllib.request
from typing import Callable, Optional

import cv2

from .events import now
from .schema import ATTRS, Frame


class QueuedVLMPipeline:
    """capacity=None : unbounded queue (the baseline: every frame is accepted and waits)
       capacity=0    : no queue: accept a frame only while the VLM is idle
       capacity=k    : bounded queue; when full, drop_policy decides:
                       'reject_new'  -> refuse the new frame (harness logs it as dropped)
                       'drop_oldest' -> evict the oldest queued frame (logged here), accept the new one
       gate(frame) -> bool : frames failing the gate are consumed without calling the VLM."""

    name = "queued_vlm"

    def __init__(self, capacity: Optional[int] = None, drop_policy: str = "reject_new",
                 gate: Optional[Callable[[Frame], bool]] = None):
        assert drop_policy in ("reject_new", "drop_oldest")
        self.capacity, self.drop_policy, self.gate = capacity, drop_policy, gate
        self._q: collections.deque[Frame] = collections.deque()
        self._cv = threading.Condition()
        self._busy = False
        self._stop = False

    def describe(self) -> dict:
        return {"name": self.name, "capacity": self.capacity, "drop_policy": self.drop_policy,
                "gated": self.gate is not None}

    def start(self, ctx):
        self.ctx = ctx
        self._thread = threading.Thread(target=self._worker, daemon=True)
        self._thread.start()
        ctx.queue_len(0)

    def offer(self, frame: Frame) -> bool:
        if self.gate is not None and not self.gate(frame):
            self.ctx.event("gate", frame_id=frame.frame_id, media_ts=frame.media_ts, passed=False)
            return True                              # consumed by the gate: not a drop
        with self._cv:
            if self.capacity == 0:
                if self._busy or self._q:
                    return False
            elif self.capacity is not None and len(self._q) >= self.capacity:
                if self.drop_policy == "reject_new":
                    return False
                evicted = self._q.popleft()
                self.ctx.drop(evicted, "evicted_oldest")
            self._q.append(frame)
            n = len(self._q)
            self._cv.notify()
        self.ctx.queue_len(n)
        return True

    def _worker(self):
        while True:
            with self._cv:
                while not self._q and not self._stop:
                    self._cv.wait(0.1)
                if self._stop and not self._q:
                    return
                frame = self._q.popleft()
                self._busy = True
                n = len(self._q)
            self.ctx.queue_len(n)
            cid = self.ctx.vlm_start(frame)
            try:
                out = self.describe_frame(frame)
                raw, timings = out if isinstance(out, tuple) else (out, {})
                err = ""
            except Exception as e:                   # a crashed call is logged, never silently lost
                raw, timings, err = "", {}, f"{type(e).__name__}: {e}"
            self.ctx.vlm_end(cid, error=err, **timings)
            self.ctx.post_output(frame, raw, call_id=cid)
            with self._cv:
                self._busy = False
                self._cv.notify_all()

    def drain(self, timeout_s: float) -> bool:
        deadline = now() + timeout_s
        with self._cv:
            while (self._q or self._busy) and now() < deadline:
                self._cv.wait(0.1)
            left = list(self._q)
            self._q.clear()
        for f in left:
            self.ctx.drop(f, "unfinished_at_end")
        return not left

    def stop(self):
        with self._cv:
            self._stop = True
            self._cv.notify_all()
        self._thread.join(timeout=5)

    def describe_frame(self, frame: Frame):
        raise NotImplementedError


class StubVLM(QueuedVLMPipeline):
    """Fake VLM for testing the harness without a model.
    service_s : seconds per call (simulates encoder + prefill + decode)
    oracle    : callable(media_ts) -> list of person dicts actually visible (from ground truth),
                so a perfect describer can be simulated; None -> always reports nobody."""
    name = "stub_vlm"

    def __init__(self, service_s: float = 1.0, oracle=None, **kw):
        super().__init__(**kw)
        self.service_s, self.oracle = service_s, oracle

    def describe(self):
        return {**super().describe(), "service_s": self.service_s}

    def describe_frame(self, frame):
        time.sleep(self.service_s)
        persons = self.oracle(frame.media_ts) if self.oracle else []
        return json.dumps({"persons": persons}), {"stub": True}


class LlamaServerVLM(QueuedVLMPipeline):
    """Calls a running llama.cpp `llama-server` (started with the model and --mmproj) through its
    OpenAI-compatible /v1/chat/completions endpoint. Logs llama-server's own timings per call."""
    name = "llama_server"

    DEFAULT_PROMPT = ("Describe every person visible in this image. Answer with JSON only, in the form "
                      '{"persons": [{' + ", ".join(f'"{a}": ...' for a in ATTRS) + "}]}. "
                      "Colours: black, white, grey, red, orange, yellow, green, blue, purple, pink, brown. "
                      'outerwear, bag, glasses: "yes" or "no". Use "unknown" if you cannot tell. '
                      'If nobody is visible answer {"persons": []}.')

    def __init__(self, url: str = "http://127.0.0.1:8080", prompt: Optional[str] = None,
                 max_tokens: int = 2048, resize_long_side: Optional[int] = None, jpeg_quality: int = 90,
                 extra: Optional[dict] = None, **kw):
        super().__init__(**kw)
        self.url, self.prompt = url.rstrip("/"), prompt or self.DEFAULT_PROMPT
        self.max_tokens, self.resize, self.q = max_tokens, resize_long_side, jpeg_quality
        self.extra = extra or {}

    def describe(self):
        return {**super().describe(), "url": self.url, "max_tokens": self.max_tokens,
                "resize_long_side": self.resize, "prompt": self.prompt, "extra": self.extra}

    def describe_frame(self, frame):
        img = frame.image
        if self.resize:
            h, w = img.shape[:2]
            s = self.resize / max(h, w)
            if s < 1:
                img = cv2.resize(img, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)
        ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, self.q])
        b64 = base64.b64encode(buf.tobytes()).decode()
        body = {"messages": [{"role": "user", "content": [
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
                    {"type": "text", "text": self.prompt}]}],
                "max_tokens": self.max_tokens, "temperature": 0, **self.extra}
        req = urllib.request.Request(f"{self.url}/v1/chat/completions", data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"})
        t = time.monotonic()
        with urllib.request.urlopen(req, timeout=600) as r:
            resp = json.loads(r.read())
        choice = resp["choices"][0]
        msg = choice["message"]
        text = (msg.get("reasoning_content") and f"<think>{msg['reasoning_content']}</think>" or "") + (msg.get("content") or "")
        tm = resp.get("timings", {})
        return text, {"http_s": round(time.monotonic() - t, 3), "image_hw": list(img.shape[:2]),
                      "prompt_n": tm.get("prompt_n"), "prompt_ms": tm.get("prompt_ms"),
                      "predicted_n": tm.get("predicted_n"), "predicted_ms": tm.get("predicted_ms"),
                      "usage": resp.get("usage"),
                      # "length" = the output hit max_tokens (e.g. thinking ran out of budget before the JSON)
                      "finish_reason": choice.get("finish_reason"),
                      "truncated": choice.get("finish_reason") == "length"}
