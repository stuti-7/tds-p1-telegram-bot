"""Append-only JSONL run log, plus optional auto-publish so log_url stays live.

The grader downloads log_url with wget and reads one JSON object per line, so
this module guarantees two things the original code did not:

  * every line is valid JSON, even when an event holds something unserialisable
    (a broken line makes the whole log unreadable);
  * concurrent handlers cannot interleave a half-written line.

Publishing is best-effort by design: if the git push fails the bot must keep
answering questions, since answers are worth far more than the log.
"""

from __future__ import annotations

import asyncio
import json
import os
import subprocess
import time
from pathlib import Path

LOG_FILE = Path(os.getenv("LOG_FILE", "run.jsonl"))

# Push every N appended events. 0 disables auto-publish (push manually instead).
PUBLISH_EVERY = int(os.getenv("PUBLISH_EVERY", "0"))
PUBLISH_MIN_INTERVAL = float(os.getenv("PUBLISH_MIN_INTERVAL_SECONDS", "60"))

_lock = asyncio.Lock()
_since_publish = 0
_last_publish = 0.0


def _safe(value):
    """Make an event JSON-serialisable rather than lose the line."""
    try:
        json.dumps(value)
        return value
    except (TypeError, ValueError):
        return repr(value)


async def log_event(event: dict) -> None:
    """Append one event. Never raises - logging must not break answering."""
    global _since_publish
    record = {"timestamp": time.time(), **{k: _safe(v) for k, v in event.items()}}
    line = json.dumps(record, ensure_ascii=False)
    try:
        async with _lock:
            LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
            with open(LOG_FILE, "a", encoding="utf-8") as fh:
                fh.write(line + "\n")
                fh.flush()
                os.fsync(fh.fileno())  # survive an abrupt worker restart
            _since_publish += 1
            due = PUBLISH_EVERY and _since_publish >= PUBLISH_EVERY
        if due:
            await publish()
    except Exception as exc:  # pragma: no cover - defensive
        print(f"[runlog] failed to write event: {type(exc).__name__}: {exc}", flush=True)


def _git(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(["git", *args], capture_output=True, text=True, timeout=60)


def _publish_blocking() -> tuple[bool, str]:
    if not (Path(".git").exists() or _git("rev-parse", "--git-dir").returncode == 0):
        return False, "not a git repository"
    if _git("add", str(LOG_FILE)).returncode != 0:
        return False, "git add failed"
    commit = _git("commit", "-m", f"run log {time.strftime('%Y-%m-%d %H:%M:%S')}")
    if commit.returncode != 0 and "nothing to commit" not in (commit.stdout + commit.stderr):
        return False, (commit.stderr or commit.stdout).strip()[:300]
    push = _git("push")
    if push.returncode != 0:
        return False, (push.stderr or push.stdout).strip()[:300]
    return True, "pushed"


async def publish() -> None:
    """Commit and push run.jsonl so log_url serves fresh content. Best effort."""
    global _since_publish, _last_publish
    now = time.monotonic()
    if now - _last_publish < PUBLISH_MIN_INTERVAL:
        return
    _last_publish = now
    _since_publish = 0
    try:
        ok, detail = await asyncio.to_thread(_publish_blocking)
        print(f"[runlog] publish: {'ok' if ok else 'skipped'} ({detail})", flush=True)
    except Exception as exc:
        print(f"[runlog] publish failed: {type(exc).__name__}: {exc}", flush=True)


def verify() -> list[str]:
    """Return a list of problems with the existing log file (empty == healthy)."""
    problems = []
    if not LOG_FILE.exists():
        return [f"{LOG_FILE} does not exist yet"]
    for n, line in enumerate(LOG_FILE.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            if not isinstance(json.loads(line), dict):
                problems.append(f"line {n}: not a JSON object")
        except json.JSONDecodeError as exc:
            problems.append(f"line {n}: invalid JSON ({exc.msg})")
    return problems


if __name__ == "__main__":
    issues = verify()
    print(f"{LOG_FILE}: " + ("healthy - every line is a JSON object" if not issues
                             else f"{len(issues)} problem(s)"))
    for issue in issues:
        print(" ", issue)
