"""Shared data formats for the A2 pipeline (frames, person records, attribute vocabulary).

Everything the group exchanges is defined here so P1 (streams), P2 (VLM), P3 (harness)
and P4 (labels) use the same names and values. See INTERFACES.md for the prose version.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np

# --------------------------------------------------------------------------------------
# Attribute vocabulary (follows the PETA / Market-1501 style of HydraPlus-Net and APR)
# --------------------------------------------------------------------------------------
COLORS = ["black", "white", "grey", "red", "orange", "yellow", "green", "blue", "purple", "pink", "brown"]
PRESENCE_ATTRS = ["outerwear", "bag", "glasses"]          # yes / no: "no" means the item is ABSENT
COLOR_ATTRS = ["upper_color", "lower_color"]
ATTRS = COLOR_ATTRS + PRESENCE_ATTRS
UNKNOWN = "unknown"                                       # the model may abstain; never a hallucination

ALLOWED = {a: COLORS + [UNKNOWN] for a in COLOR_ATTRS}
ALLOWED.update({a: ["yes", "no", UNKNOWN] for a in PRESENCE_ATTRS})

# light normalisation of free-text values the VLM may produce
_SYNONYMS = {
    "gray": "grey", "navy": "blue", "dark blue": "blue", "light blue": "blue", "beige": "brown",
    "khaki": "brown", "tan": "brown", "maroon": "red", "violet": "purple", "silver": "grey",
    "true": "yes", "false": "no", "none": "no", "not visible": UNKNOWN, "n/a": UNKNOWN, "null": UNKNOWN,
    "unsure": UNKNOWN, "unclear": UNKNOWN,
}


@dataclass
class Frame:
    """One sampled frame handed from the harness to the pipeline.

    frame_id   : running index of SAMPLED frames in this run (0, 1, 2, ...)
    media_ts   : position in the stream in seconds (stream time; 0 = start of the stream)
    wall_ts    : time.monotonic() when the harness handed the frame over
    image      : H x W x 3 uint8, BGR (OpenCV order)
    """
    frame_id: int
    media_ts: float
    wall_ts: float
    image: np.ndarray = field(repr=False)
    source_id: str = ""


def normalize_value(attr: str, value: Any) -> str:
    if value is None:
        return UNKNOWN
    if isinstance(value, bool):
        value = "yes" if value else "no"
    v = str(value).strip().lower()
    v = _SYNONYMS.get(v, v)
    if attr in COLOR_ATTRS:
        # "dark blue jacket" -> first known colour word
        for c in COLORS:
            if re.search(rf"\b{c}\b", v):
                return c
    return v


def validate_person(p: Any) -> tuple[Optional[dict], list[str]]:
    """Return (clean person dict with every attribute, list of problems)."""
    if not isinstance(p, dict):
        return None, ["person is not a JSON object"]
    clean, errs = {}, []
    for a in ATTRS:
        v = normalize_value(a, p.get(a, UNKNOWN))
        if v not in ALLOWED[a]:
            errs.append(f"{a}={v!r} not allowed")
            v = UNKNOWN
        clean[a] = v
    extra = [k for k in p if k not in ATTRS]
    if extra:
        errs.append(f"extra keys ignored: {extra}")
    return clean, errs


def _strip_thinking(text: str) -> str:
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.S | re.I)
    return re.sub(r"<think>.*", "", text, flags=re.S | re.I)    # unterminated thinking block


def _balanced(text: str, start: int) -> Optional[str]:
    """The balanced [...] or {...} block that starts at text[start], or None if it never closes."""
    open_c = text[start]
    close_c = "}" if open_c == "{" else "]"
    depth, in_str, esc = 0, False, False
    for i in range(start, len(text)):
        ch = text[i]
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch == open_c:
            depth += 1
        elif ch == close_c:
            depth -= 1
            if depth == 0:
                return text[start:i + 1]
    return None


def _looks_like_persons(obj) -> bool:
    """A list of person objects (or empty), {"persons": ...}, or one person object."""
    if isinstance(obj, list):
        return all(isinstance(p, dict) for p in obj)
    if isinstance(obj, dict):
        return "persons" in obj or any(k in obj for k in ATTRS)
    return False


def _find_json(text: str):
    """Scan the text for the first JSON block that parses and looks like a person list.
    [...] blocks are tried first (so chatter before the JSON, or {"persons": [...]}, both work),
    then {...} blocks. Stray brackets in prose such as "[image]" are skipped.
    Returns (obj, error); obj is None if nothing usable was found."""
    last_err = "no JSON found"
    for opener in "[{":
        i = text.find(opener)
        while i >= 0:
            block = _balanced(text, i)
            if block is None:
                last_err = "JSON block never closes (output truncated?)"
            else:
                try:
                    obj = json.loads(block)
                    if _looks_like_persons(obj):
                        return obj, ""
                except json.JSONDecodeError as e:
                    last_err = f"invalid JSON: {e.msg}"
            i = text.find(opener, i + 1)
    return None, last_err


def parse_vlm_output(text: str) -> tuple[list[dict], bool, str]:
    """Parse raw VLM text into a list of person dicts.

    Accepts a bare list [...], {"persons": [...]}, or a single person object {...}, anywhere in the
    text: thinking blocks, code fences and sentences before or after the JSON are ignored.
    Returns (persons, parse_ok, error). parse_ok is False when no valid JSON was found;
    an empty list with parse_ok True means "the model saw nobody".
    """
    obj, err = _find_json(_strip_thinking(text or ""))
    if obj is None:
        return [], False, err
    if isinstance(obj, dict) and "persons" in obj:
        obj = obj["persons"]
    if isinstance(obj, dict):
        obj = [obj]
    if not isinstance(obj, list):
        return [], False, "JSON is not an object or list"
    persons, errs = [], []
    for p in obj:
        clean, e = validate_person(p)
        if clean is not None:
            persons.append(clean)
        errs.extend(e)
    return persons, True, "; ".join(errs)
