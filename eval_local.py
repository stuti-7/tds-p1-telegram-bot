"""Run the eval questions against the agent locally - no Telegram, no session.

This reproduces what the public harness does end to end:

    collect.py  sends each message in turn and waits for a reply after EVERY one,
                keeping the replies in order
    grade.py    parses replies[-1] with json.loads and tests `answer == expected`

so a pass here is the same comparison the grader makes. It just skips Telegram,
which makes iterating on the agent minutes faster and avoids burning the
collector's rate limits.

    python eval_local.py
    python eval_local.py --only sum percentage
    python eval_local.py --questions tds-p1-t2-2026-telegram-bot/evals/questions.json
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
from pathlib import Path

from dotenv import load_dotenv

import shape
from agent import Agent

load_dotenv()

DEFAULT_QUESTIONS = "tds-p1-t2-2026-telegram-bot/evals/questions.json"

GREEN, RED, YELLOW, DIM, RESET = "\033[32m", "\033[31m", "\033[33m", "\033[2m", "\033[0m"


async def run_question(agent: Agent, question: dict, log_url: str) -> dict:
    """Simulate one collected conversation and return a grade row."""
    messages = question["messages"]
    replies: list[str] = []
    started = time.monotonic()

    for i in range(len(messages)):
        turns = messages[: i + 1]          # the collector's view of the chat so far
        model_text = await agent.answer(turns)
        reply, _ = shape.finalise(model_text, turns, log_url)
        replies.append(reply)

    elapsed = time.monotonic() - started

    # grade.py's extract_answer + grade
    try:
        answer = json.loads(replies[-1].strip())
    except json.JSONDecodeError:
        return {"id": question["id"], "correct": False, "detail": "format_error",
                "got": replies[-1], "elapsed": elapsed}

    expected = question.get("expected")
    if expected == {"state": "REPLACE_ME"} or expected is None:
        return {"id": question["id"], "correct": None,
                "detail": "no expected value set in questions.json",
                "got": answer, "elapsed": elapsed}

    correct = answer == expected
    return {"id": question["id"], "correct": correct,
            "detail": "ok" if correct else f"expected {expected}, got {answer}",
            "got": answer, "elapsed": elapsed}


async def main_async(args) -> int:
    token = os.getenv("AIPIPE_TOKEN", "").strip()
    if not token:
        print("AIPIPE_TOKEN is not set (put it in .env)", file=sys.stderr)
        return 2

    log_url = os.getenv("LOG_URL", "").strip() or "https://example.invalid/run.jsonl"
    questions = json.loads(Path(args.questions).read_text())
    if args.only:
        questions = [q for q in questions if q["id"] in set(args.only)]
    if not questions:
        print("no questions selected", file=sys.stderr)
        return 2

    agent = Agent(token)
    print(f"running {len(questions)} question(s) against model="
          f"{os.getenv('MODEL', 'gpt-5-mini')}\n")

    sem = asyncio.Semaphore(args.concurrency)

    async def guarded(q):
        async with sem:
            row = await run_question(agent, q, log_url)
            mark = {True: f"{GREEN}PASS{RESET}", False: f"{RED}FAIL{RESET}",
                    None: f"{YELLOW}SKIP{RESET}"}[row["correct"]]
            print(f"{mark}  {row['id']:<28} {row['elapsed']:5.1f}s  {DIM}{row['detail']}{RESET}")
            return row

    rows = await asyncio.gather(*(guarded(q) for q in questions))

    graded = [r for r in rows if r["correct"] is not None]
    passed = sum(1 for r in graded if r["correct"])
    skipped = len(rows) - len(graded)
    slowest = max(r["elapsed"] for r in rows)

    print(f"\n{passed}/{len(graded)} correct" + (f", {skipped} skipped" if skipped else ""))
    print(f"slowest question: {slowest:.1f}s "
          f"(budget {questions[0].get('timeout_seconds', 300)}s)")

    if args.json:
        Path(args.json).write_text(json.dumps(rows, indent=2))
        print(f"wrote {args.json}")

    return 0 if passed == len(graded) else 1


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--questions", default=DEFAULT_QUESTIONS)
    ap.add_argument("--only", nargs="*", help="question ids to run")
    ap.add_argument("--concurrency", type=int, default=4)
    ap.add_argument("--json", help="write the grade rows to this path")
    return asyncio.run(main_async(ap.parse_args()))


if __name__ == "__main__":
    sys.exit(main())
