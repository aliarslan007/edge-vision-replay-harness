"""Append-only JSONL event log. One JSON object per line, always with "ev" and "wall_ts".

All times are time.monotonic() seconds (wall clock that never jumps). The scorer converts
them to run-relative and stream (media) time using the run_start event.
"""
from __future__ import annotations

import json
import threading
import time
from pathlib import Path

now = time.monotonic


class EventLog:
    def __init__(self, path: str | Path, flush_every: int = 50):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._f = open(self.path, "w")
        self._lock = threading.Lock()
        self._n = 0
        self._flush_every = flush_every

    def write(self, ev: str, **fields) -> dict:
        fields = {"ev": ev, "wall_ts": fields.pop("wall_ts", None) or now(), **fields}
        line = json.dumps(fields, default=_default)
        with self._lock:
            self._f.write(line + "\n")
            self._n += 1
            if self._n % self._flush_every == 0:
                self._f.flush()
        return fields

    def close(self):
        with self._lock:
            self._f.flush()
            self._f.close()


def _default(o):
    try:
        import numpy as np
        if isinstance(o, np.generic):
            return o.item()
    except ImportError:
        pass
    return str(o)


def read_log(path: str | Path) -> list[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]
