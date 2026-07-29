"""Telegram entrypoint for the data-analyst bot.

Responsibilities kept deliberately narrow: receive a message, keep per-chat
context for multi-turn tasks, ask the agent, force the reply into the exact
requested JSON shape, log, send.

The two rules that drive every design choice here:
  1. ALWAYS reply. A missing reply is recorded as `timeout`, a terminal failure,
     and the collector waits for a reply after EVERY message it sends - including
     a context-setting turn like "Remember these numbers: 12, 25, 48."
  2. Reply with exactly one JSON object and nothing else (see shape.py).

Run: python bot.py
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
import time
from collections import defaultdict, deque

from dotenv import load_dotenv

import runlog
import shape
from agent import Agent

load_dotenv()

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
AIPIPE_TOKEN = os.getenv("AIPIPE_TOKEN", "").strip()
LOG_URL = os.getenv("LOG_URL", "").strip()

# How many past turns of a conversation to carry. The collector's multi-turn
# questions are short; this is bounded so a long-lived worker cannot grow
# without limit.
HISTORY_TURNS = int(os.getenv("HISTORY_TURNS", "8"))

# The grader sends every question to the same Telegram chat, so without a reset
# an earlier question's turns leak into the next one - a lead-in turn carrying no
# shape of its own ("Remember these numbers: 12, 25, 48.") would otherwise pick up
# the *previous* question's template. Turns within one question arrive seconds
# apart (the collector waits for a reply before sending the next), while separate
# questions are whole waves apart, so an idle gap cleanly separates tasks.
SESSION_GAP_SECONDS = float(os.getenv("SESSION_GAP_SECONDS", "120"))

# Never answer a message older than the question budget. The grader reads the
# FIRST reply that arrives after each message it sends, so one late duplicate
# reply shifts every following question's answer by one and fails the whole run.
# Past this age the grader has already timed out that question, so a reply cannot
# help and can only desynchronise what follows.
MAX_MESSAGE_AGE_SECONDS = float(os.getenv("MAX_MESSAGE_AGE_SECONDS", "300"))

# Per-chat user messages, and a per-chat lock so a burst of messages in one chat
# is handled in order (across chats we stay concurrent).
_history: dict[int, deque] = defaultdict(lambda: deque(maxlen=HISTORY_TURNS))
_chat_locks: dict[int, asyncio.Lock] = defaultdict(asyncio.Lock)
_last_seen: dict[int, float] = {}

_agent: Agent | None = None


def preflight() -> None:
    """Fail fast and loudly on misconfiguration, without leaking secrets."""
    problems = []
    if not TELEGRAM_BOT_TOKEN:
        problems.append("TELEGRAM_BOT_TOKEN is not set")
    if not AIPIPE_TOKEN:
        problems.append("AIPIPE_TOKEN is not set")

    warnings = []
    if not LOG_URL:
        warnings.append("LOG_URL is not set - replies that ask for log_url will send an empty string")
    elif not LOG_URL.startswith("http"):
        warnings.append(f"LOG_URL does not look like a URL: {LOG_URL!r}")
    elif "example.com" in LOG_URL:
        warnings.append(f"LOG_URL is still a placeholder ({LOG_URL}) - the grader cannot fetch it")

    if problems:
        print("Configuration error:", file=sys.stderr)
        for p in problems:
            print(f"  - {p}", file=sys.stderr)
        print("\nCopy .env.example to .env and fill it in, or set the variables in your "
              "host's dashboard.", file=sys.stderr)
        sys.exit(1)

    for w in warnings:
        print(f"WARNING: {w}", flush=True)

    # Identify the credentials without printing any part of them.
    print(f"config ok: telegram token ...{TELEGRAM_BOT_TOKEN[-4:]}, "
          f"aipipe token set ({len(AIPIPE_TOKEN)} chars), log_url={LOG_URL or '(unset)'}",
          flush=True)


def _message_age(message) -> float | None:
    """Seconds since the message was sent, or None if Telegram gave no date."""
    date = getattr(message, "date", None)
    if date is None:
        return None
    try:
        import datetime
        now = datetime.datetime.now(datetime.timezone.utc)
        return (now - date).total_seconds()
    except Exception:
        return None


async def handle_message(update, context) -> None:
    """Answer one message. Every failure path still sends a JSON reply."""
    message = getattr(update, "effective_message", None)
    if message is None or not (message.text or "").strip():
        return

    chat_id = update.effective_chat.id
    user_text = message.text.strip()

    age = _message_age(message)
    if age is not None and age > MAX_MESSAGE_AGE_SECONDS:
        await runlog.log_event({
            "type": "skipped_stale", "chat_id": chat_id,
            "text": user_text, "age_seconds": round(age, 1),
        })
        return

    async with _chat_locks[chat_id]:
        now = time.monotonic()
        previous = _last_seen.get(chat_id)
        new_session = previous is not None and (now - previous) > SESSION_GAP_SECONDS
        _last_seen[chat_id] = now

        history = _history[chat_id]
        if new_session:
            history.clear()
        history.append(user_text)
        turns = list(history)

        await runlog.log_event({
            "type": "incoming", "chat_id": chat_id, "text": user_text,
            "new_session": new_session,
        })

        events: list[dict] = []
        try:
            model_text = await _agent.answer(turns)
        except Exception as exc:  # agent.answer already guards, this is belt-and-braces
            model_text = ""
            events.append({"type": "agent_exception", "error": f"{type(exc).__name__}: {exc}"})

        reply_text, debug = shape.finalise(model_text, turns, LOG_URL)

        await runlog.log_event({
            "type": "outgoing",
            "chat_id": chat_id,
            "text": reply_text,
            "turns": len(turns),
            **debug,
            "events": events,
        })

        try:
            await message.reply_text(reply_text)
        except Exception as exc:
            await runlog.log_event({
                "type": "send_failed", "chat_id": chat_id,
                "error": f"{type(exc).__name__}: {exc}",
            })
            return

        # Keep the assistant turn out of history: the agent re-derives the answer
        # from the user's turns, and feeding our own JSON back encourages the
        # model to echo a previous question's shape.


async def on_error(update, context) -> None:
    """Last line of defence - never let an exception end without a reply."""
    err = context.error
    await runlog.log_event({"type": "handler_error", "error": f"{type(err).__name__}: {err}"})
    print(f"[error] {type(err).__name__}: {err}", flush=True)

    message = getattr(update, "effective_message", None)
    if message is None:
        return
    try:
        turns = list(_history.get(update.effective_chat.id, []))
        keys, wants_log = shape.shape_from_conversation(turns)
        fallback = shape.conform({}, keys, LOG_URL, wants_log)
        await message.reply_text(json.dumps(fallback, ensure_ascii=False,
                                            separators=(", ", ": ")))
    except Exception:
        pass


def main() -> None:
    global _agent

    preflight()

    issues = runlog.verify()
    if issues and "does not exist" not in issues[0]:
        print(f"WARNING: {runlog.LOG_FILE} has {len(issues)} malformed line(s); "
              f"run `python runlog.py` for details", flush=True)

    _agent = Agent(AIPIPE_TOKEN, on_event=lambda e: asyncio.create_task(runlog.log_event(e)))

    from telegram.ext import (ApplicationBuilder, ContextTypes, MessageHandler,
                              filters)

    app = (ApplicationBuilder()
           .token(TELEGRAM_BOT_TOKEN)
           .concurrent_updates(True)   # different chats proceed in parallel
           .build())
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
    app.add_handler(MessageHandler(filters.COMMAND, handle_message))  # /start etc. still answer
    app.add_error_handler(on_error)

    print("Bot is running... (Ctrl+C to stop)", flush=True)
    # drop_pending_updates=True: a killed worker never confirms its update offset,
    # so on restart Telegram redelivers messages we already answered. Answering
    # them again puts a stale reply in the chat, and the grader - which takes the
    # first reply after each send - then reads every answer one question late.
    # Losing one queued question beats corrupting all the questions after it.
    app.run_polling(drop_pending_updates=True, allowed_updates=["message"])


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nstopped", flush=True)
    except Exception as exc:
        name = type(exc).__name__
        if name == "Conflict":
            print("Startup failed: another instance is already polling this bot token. "
                  "Stop it (including any local run) and retry.", file=sys.stderr)
        elif name in ("InvalidToken", "Unauthorized"):
            print("Startup failed: TELEGRAM_BOT_TOKEN is rejected by Telegram. "
                  "Re-copy it from @BotFather.", file=sys.stderr)
        else:
            print(f"Startup failed: {name}: {exc}", file=sys.stderr)
        sys.exit(1)
