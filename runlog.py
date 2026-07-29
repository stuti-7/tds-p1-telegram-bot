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


_git_auth_ready = False


def _configure_git_auth() -> None:
    """Let a headless deploy (Render etc.) push over HTTPS with no stored login.

    Idempotent, called once. Embeds GITHUB_TOKEN into the origin URL itself
    rather than relying on a credential helper, which a fresh container has
    none of. The token never appears in stdout/logs - only in local git config
    on that container's own disk, which is never committed or exposed.
    """
    global _git_auth_ready
    if _git_auth_ready:
        return
    _git_auth_ready = True

    token = os.getenv("GITHUB_TOKEN", "").strip()
    if token:
        current = _git("remote", "get-url", "origin")
        url = current.stdout.strip()
        if url.startswith("https://") and "@" not in url.split("//", 1)[1].split("/", 1)[0]:
            _git("remote", "set-url", "origin",
                 url.replace("https://", f"https://x-access-token:{token}@", 1))

    # A fresh container has no git identity configured; commit fails without one.
    _git("config", "user.email", os.getenv("GIT_AUTHOR_EMAIL", "bot@tds-p1-telegram-bot.local"))
    _git("config", "user.name", os.getenv("GIT_AUTHOR_NAME", "tds-p1-telegram-bot"))


def _publish_blocking() -> tuple[bool, str]:
    if not (Path(".git").exists() or _git("rev-parse", "--git-dir").returncode == 0):
        return False, "not a git repository"
    _configure_git_auth()
    if _git("add", str(LOG_FILE)).returncode != 0:
        return False, "git add failed"
    commit = _git("commit", "-m", f"run log {time.strftime('%Y-%m-%d %H:%M:%S')}")
    if commit.returncode != 0 and "nothing to commit" not in (commit.stdout + commit.stderr):
        return False, (commit.stderr or commit.stdout).strip()[:300]

    # HEAD:main survives a shallow/detached checkout, which some hosts use.
    push = _git("push", "origin", "HEAD:main")
    if push.returncode != 0:
        # Someone else (a manual push, a previous instance) moved main forward -
        # reconcile once and retry, rather than silently falling behind forever.
        _git("fetch", "origin", "main")
        if _git("rebase", "origin/main").returncode != 0:
            _git("rebase", "--abort")
            return False, "push rejected, rebase failed - needs a manual look"
        push = _git("push", "origin", "HEAD:main")
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
