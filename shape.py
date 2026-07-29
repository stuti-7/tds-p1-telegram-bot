"""Work out the exact JSON shape a question asks for, and force our reply into it.

Why this module exists
----------------------
The public grading harness (grade.py) compares our final reply to the expected
answer with whole-object equality:

    answer = json.loads(replies[-1].strip())
    ok = answer == expected

So a reply that carries the right value but one extra key is scored *wrong*.
That is exactly what the original bot did: it appended "log_url" to every reply,
including questions whose stated shape never mentioned log_url, which failed
every single question in evals/questions.json.

Every question spells its shape out inline:

    ... Reply with ONLY a JSON object like {"sum": 200}
    ... Reply with ONLY this JSON object: {"answer": {"state": "<state>"}, "log_url": "<url>"}

This module extracts that template's top-level keys and uses them to
  * drop keys the model invented,
  * rename a near-miss key the model chose instead of the requested one,
  * add "log_url" only when the question actually asked for one.

The templates are not always valid JSON ("<state name>" placeholders, trailing
commentary), so the scanning here is deliberately lenient and never uses
json.loads on the question text.
"""

from __future__ import annotations

import json
import re

# Telegram clients happily send typographic quotes; normalise them before scanning.
_SMART_QUOTES = str.maketrans({"“": '"', "”": '"', "‘": "'", "’": "'"})

LOG_URL_KEY = "log_url"


def _normalise(text: str) -> str:
    return (text or "").translate(_SMART_QUOTES)


def _outer_brace_spans(text: str) -> list[tuple[int, int]]:
    """Return (start, end) for every top-level balanced {...} region.

    String-aware, so braces inside quoted values do not unbalance the scan.
    Nested objects are not returned separately - only the outermost span.
    """
    spans: list[tuple[int, int]] = []
    depth = 0
    start = -1
    i = 0
    n = len(text)
    while i < n:
        ch = text[i]
        if ch in "\"'":
            i = _skip_string(text, i)
            continue
        if ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}":
            if depth > 0:
                depth -= 1
                if depth == 0 and start >= 0:
                    spans.append((start, i + 1))
                    start = -1
        i += 1
    return spans


def _skip_string(text: str, i: int) -> int:
    """Given text[i] is a quote, return the index just past the closing quote."""
    quote = text[i]
    j = i + 1
    n = len(text)
    while j < n:
        if text[j] == "\\":
            j += 2
            continue
        if text[j] == quote:
            return j + 1
        j += 1
    return n


def _read_string(text: str, i: int) -> tuple[str, int]:
    """Given text[i] is a quote, return (unescaped contents, index past close)."""
    quote = text[i]
    out: list[str] = []
    j = i + 1
    n = len(text)
    while j < n:
        if text[j] == "\\" and j + 1 < n:
            out.append(text[j + 1])
            j += 2
            continue
        if text[j] == quote:
            return "".join(out), j + 1
        out.append(text[j])
        j += 1
    return "".join(out), n


def _top_level_keys(blob: str) -> list[str]:
    """Keys at depth 1 of a (possibly non-strict) JSON object literal, in order."""
    keys: list[str] = []
    depth = 0
    i = 0
    n = len(blob)
    while i < n:
        ch = blob[i]
        if ch in "\"'":
            value, nxt = _read_string(blob, i)
            k = nxt
            while k < n and blob[k].isspace():
                k += 1
            if depth == 1 and k < n and blob[k] == ":" and value not in keys:
                keys.append(value)
            i = nxt
            continue
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
        i += 1
    return keys


def requested_keys(text: str) -> list[str] | None:
    """Top-level keys of the JSON shape the message asks for, or None if absent.

    Uses the *last* balanced {...} in the message that actually contains keys -
    questions put the template at the end ("Reply with ONLY ... {shape}"), and a
    question may quote an example object earlier on.
    """
    normalised = _normalise(text)
    for start, end in reversed(_outer_brace_spans(normalised)):
        keys = _top_level_keys(normalised[start:end])
        if keys:
            return keys
    return None


def shape_from_conversation(user_messages: list[str]) -> tuple[list[str] | None, bool]:
    """Resolve the shape for a (possibly multi-turn) exchange.

    The last message is authoritative - the task says to answer the LAST message -
    but a lead-in turn ("Remember these numbers: ...") may carry no template, so
    fall back through earlier turns.

    Returns (keys_or_None, wants_log_url).
    """
    for text in reversed(user_messages):
        keys = requested_keys(text)
        if keys:
            return keys, LOG_URL_KEY in keys
    # No template anywhere: honour a bare textual mention of log_url.
    wants_log = any(LOG_URL_KEY in _normalise(t) for t in user_messages)
    return None, wants_log


_FENCE_RE = re.compile(r"^\s*```(?:json)?\s*|\s*```\s*$", re.IGNORECASE)


def parse_model_json(text: str):
    """Best-effort parse of the model's reply into a Python object.

    Handles the usual failure modes: markdown fences, a sentence before the
    object, trailing commentary. Returns None if nothing parses - callers must
    treat that as "model failed", never as a crash.
    """
    if not text:
        return None
    candidate = _FENCE_RE.sub("", text.strip())

    try:
        return json.loads(candidate)
    except (json.JSONDecodeError, ValueError):
        pass

    # Try each balanced {...} region, widest last-wins, then any of them.
    normalised = _normalise(candidate)
    spans = _outer_brace_spans(normalised)
    for start, end in reversed(spans):
        try:
            return json.loads(normalised[start:end])
        except (json.JSONDecodeError, ValueError):
            continue
    return None


def _match_key(obj: dict, key: str) -> str | None:
    """Find the key in obj corresponding to `key`, tolerating case/underscore drift."""
    if key in obj:
        return key

    def canon(s: str) -> str:
        return re.sub(r"[^a-z0-9]", "", s.lower())

    target = canon(key)
    for candidate in obj:
        if canon(candidate) == target:
            return candidate
    return None


def conform(obj, keys: list[str] | None, log_url: str, wants_log_url: bool):
    """Coerce a parsed model reply into exactly the requested shape.

    - keys is None  -> shape unknown; pass the object through, only ensuring
      log_url if the question mentioned it.
    - keys is known -> emit precisely those keys, in the order asked for.
    """
    if keys is None:
        if isinstance(obj, dict) and wants_log_url:
            obj = dict(obj)
            obj[LOG_URL_KEY] = log_url
        return obj

    # A single-key shape lets us rescue a bare scalar, e.g. shape {"sum": ...}
    # and the model replied `200`.
    if not isinstance(obj, dict):
        content_keys = [k for k in keys if k != LOG_URL_KEY]
        if obj is not None and len(content_keys) == 1:
            obj = {content_keys[0]: obj}
        else:
            obj = {}

    out: dict = {}
    consumed: set[str] = set()
    missing: list[str] = []

    for key in keys:
        if key == LOG_URL_KEY:
            out[key] = log_url
            source = _match_key(obj, key)
            if source:
                consumed.add(source)
            continue
        source = _match_key(obj, key)
        if source is not None:
            out[key] = obj[source]
            consumed.add(source)
        else:
            missing.append(key)

    # The model answered under a different name than the template used. If that
    # is unambiguous (exactly one requested key unfilled, exactly one value left
    # over), take it rather than dropping a correct answer.
    leftovers = [k for k in obj if k not in consumed]
    if len(missing) == 1 and len(leftovers) == 1:
        out[missing[0]] = obj[leftovers[0]]
        missing = []

    # Anything still missing is emitted as null so the reply keeps the promised
    # shape - a wrong value scores the same as a malformed one, but a malformed
    # reply also poisons multi-turn and log review.
    for key in missing:
        out[key] = None

    return {k: out[k] for k in keys}


def finalise(model_text: str, user_messages: list[str], log_url: str) -> tuple[str, dict]:
    """Turn raw model output into the exact string to send back.

    Returns (reply_text, debug) where debug is logged, never sent.
    """
    keys, wants_log_url = shape_from_conversation(user_messages)
    parsed = parse_model_json(model_text)
    shaped = conform(parsed, keys, log_url, wants_log_url)

    debug = {
        "requested_keys": keys,
        "wants_log_url": wants_log_url,
        "parsed_ok": parsed is not None,
        "raw_model_text": model_text,
    }
    # separators: no stray whitespace; ensure_ascii=False so Indian state names
    # and similar stay human-readable in the log and in the reply.
    return json.dumps(shaped, ensure_ascii=False, separators=(", ", ": ")), debug
