"""The data-analyst agent: turns a conversation into one JSON answer.

Design constraints come straight from how grading works:

  * The whole exchange has a per-question budget (~300s in evals/questions.json,
    "minutes not seconds"). Everything here runs under a hard deadline and
    degrades to a best-effort answer rather than blowing the budget.
  * A reply that never arrives is a terminal `timeout`. So no failure mode in
    this module raises - answer() always returns text.
  * Correctness is graded on exact value equality, so the model gets tools for
    arithmetic and for reading public datasets instead of doing mental maths.

Tools are optional (ENABLE_TOOLS=0 turns them off for a plain single-call agent).
"""

from __future__ import annotations

import asyncio
import json
import os
import subprocess
import sys
import tempfile
import textwrap
import time

from openai import AsyncOpenAI

AIPIPE_BASE_URL = os.getenv("AIPIPE_BASE_URL", "https://aipipe.org/openai/v1")
MODEL = os.getenv("MODEL", "gpt-5-mini")

# Budget knobs. The defaults leave a wide margin under a 300s question budget.
TOTAL_DEADLINE = float(os.getenv("AGENT_DEADLINE_SECONDS", "150"))
LLM_TIMEOUT = float(os.getenv("LLM_TIMEOUT_SECONDS", "60"))
LLM_RETRIES = int(os.getenv("LLM_RETRIES", "2"))
MAX_TOOL_STEPS = int(os.getenv("MAX_TOOL_STEPS", "4"))
PYTHON_TIMEOUT = float(os.getenv("PYTHON_TIMEOUT_SECONDS", "20"))
FETCH_TIMEOUT = float(os.getenv("FETCH_TIMEOUT_SECONDS", "25"))
FETCH_MAX_CHARS = int(os.getenv("FETCH_MAX_CHARS", "20000"))
ENABLE_TOOLS = os.getenv("ENABLE_TOOLS", "1") not in ("0", "false", "False", "")

SYSTEM_PROMPT = textwrap.dedent("""\
    You are a careful data analyst answering over Telegram.

    The user's LAST message is the question to answer. Earlier messages are
    context only (a multi-turn task may supply the data in one message and ask
    the question in the next).

    The message states the exact JSON shape it wants back. Obey it literally:
      * Use exactly the keys shown - no more, no fewer. Do not add keys such as
        "confidence", "explanation", "source", "reasoning" or "note".
      * Keep the nesting shown. If it shows {"answer": {"state": "..."}} then
        "answer" must be an object, not a bare string.
      * Placeholders like "<state name>" mark where YOUR value goes.
      * If the shape includes "log_url", put any string there - it is replaced
        with the real URL before sending. Never invent a different key for it.
      * Numbers must be JSON numbers, not strings. Booleans must be true/false.
      * Match the units, rounding and phrasing the question asks for. If it asks
        for a state/country name, give the plain conventional name with no
        qualifiers, abbreviations or annotations.

    Work out the real answer before formatting it. Use the tools for any
    arithmetic, statistics or aggregation rather than doing it in your head, and
    to read a public dataset the question points at. Prefer authoritative
    published figures (MOSPI, NFHS, Census, World Bank) for factual questions.

    Your FINAL message must be the raw JSON object and nothing else: no prose,
    no markdown, no code fences.
""")

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "run_python",
            "description": (
                "Execute Python 3 for computation, parsing or statistics and return "
                "its stdout. Use print() to emit results. The standard library, and "
                "any of pandas/numpy that happen to be installed, are available. No "
                "state persists between calls."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "code": {"type": "string", "description": "Python source to execute."}
                },
                "required": ["code"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "fetch_url",
            "description": (
                "HTTP GET a public URL and return the response body as text, truncated. "
                "Use for public datasets, CSVs or JSON APIs referenced by the question."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "url": {"type": "string", "description": "Absolute http(s) URL."}
                },
                "required": ["url"],
            },
        },
    },
]


# --------------------------------------------------------------------------
# tool implementations - each returns a string and never raises
# --------------------------------------------------------------------------

def _run_python_blocking(code: str) -> str:
    with tempfile.TemporaryDirectory() as workdir:
        path = os.path.join(workdir, "snippet.py")
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(code)
        try:
            proc = subprocess.run(
                [sys.executable, path],
                capture_output=True, text=True,
                timeout=PYTHON_TIMEOUT, cwd=workdir,
            )
        except subprocess.TimeoutExpired:
            return f"ERROR: execution exceeded {PYTHON_TIMEOUT}s"
        except Exception as exc:  # never let a tool kill the answer
            return f"ERROR: {type(exc).__name__}: {exc}"

    out = (proc.stdout or "").strip()
    err = (proc.stderr or "").strip()
    if proc.returncode != 0:
        return f"exit {proc.returncode}\nstdout:\n{out[:2000]}\nstderr:\n{err[:2000]}"
    if not out:
        return "(no stdout - remember to print() your result)"
    return out[:8000]


async def _tool_run_python(args: dict) -> str:
    code = args.get("code") or ""
    if not code.strip():
        return "ERROR: no code provided"
    return await asyncio.to_thread(_run_python_blocking, code)


async def _tool_fetch_url(args: dict) -> str:
    url = (args.get("url") or "").strip()
    if not url.lower().startswith(("http://", "https://")):
        return "ERROR: url must be an absolute http(s) URL"
    try:
        import httpx
    except ImportError:
        return "ERROR: httpx is not installed"
    try:
        async with httpx.AsyncClient(follow_redirects=True, timeout=FETCH_TIMEOUT) as http:
            resp = await http.get(url, headers={"User-Agent": "tds-data-analyst-bot/1.0"})
        body = resp.text[:FETCH_MAX_CHARS]
        return f"HTTP {resp.status_code}\n{body}"
    except Exception as exc:
        return f"ERROR: {type(exc).__name__}: {exc}"


TOOL_IMPLS = {"run_python": _tool_run_python, "fetch_url": _tool_fetch_url}


# --------------------------------------------------------------------------
# agent loop
# --------------------------------------------------------------------------

class Agent:
    def __init__(self, api_key: str, on_event=None):
        self.client = AsyncOpenAI(
            base_url=AIPIPE_BASE_URL,
            api_key=api_key,
            timeout=LLM_TIMEOUT,
            max_retries=0,  # retries are handled here so they respect the deadline
        )
        self.on_event = on_event or (lambda event: None)

    async def _complete(self, messages, use_tools: bool, deadline: float):
        """One chat completion with bounded retries, respecting the deadline."""
        last_error = None
        for attempt in range(LLM_RETRIES + 1):
            remaining = deadline - time.monotonic()
            if remaining <= 1:
                break
            kwargs = {"model": MODEL, "messages": messages}
            if use_tools:
                kwargs["tools"] = TOOLS
                kwargs["tool_choice"] = "auto"
            try:
                return await asyncio.wait_for(
                    self.client.chat.completions.create(**kwargs),
                    timeout=min(LLM_TIMEOUT, remaining),
                )
            except Exception as exc:
                last_error = exc
                self.on_event({
                    "type": "llm_error",
                    "attempt": attempt + 1,
                    "error": f"{type(exc).__name__}: {exc}",
                })
                # Brief backoff, but only if the deadline can absorb it.
                if deadline - time.monotonic() > 5:
                    await asyncio.sleep(min(2 * (attempt + 1), 5))
        if last_error:
            raise last_error
        raise TimeoutError("agent deadline exhausted before any LLM response")

    async def answer(self, user_messages: list[str]) -> str:
        """Return the model's final raw text. Never raises; '' means total failure."""
        deadline = time.monotonic() + TOTAL_DEADLINE
        messages = [{"role": "system", "content": SYSTEM_PROMPT}]
        messages += [{"role": "user", "content": text} for text in user_messages]

        try:
            for step in range(MAX_TOOL_STEPS + 1):
                # On the final step force a plain answer so we never end the loop
                # holding an unanswered tool call.
                use_tools = ENABLE_TOOLS and step < MAX_TOOL_STEPS and (
                    deadline - time.monotonic() > 30)

                response = await self._complete(messages, use_tools, deadline)
                choice = response.choices[0].message
                tool_calls = getattr(choice, "tool_calls", None) or []

                if not tool_calls:
                    text = (choice.content or "").strip()
                    self.on_event({"type": "llm_answer", "step": step, "text": text})
                    return text

                # Echo the assistant turn back verbatim, then answer each call.
                messages.append({
                    "role": "assistant",
                    "content": choice.content,
                    "tool_calls": [
                        {"id": tc.id, "type": "function",
                         "function": {"name": tc.function.name,
                                      "arguments": tc.function.arguments}}
                        for tc in tool_calls
                    ],
                })

                for tc in tool_calls:
                    name = tc.function.name
                    try:
                        args = json.loads(tc.function.arguments or "{}")
                    except json.JSONDecodeError:
                        args = {}
                    impl = TOOL_IMPLS.get(name)
                    result = (await impl(args)) if impl else f"ERROR: unknown tool {name}"
                    self.on_event({
                        "type": "tool_call", "step": step, "tool": name,
                        "args": args, "result": result[:2000],
                    })
                    messages.append({
                        "role": "tool", "tool_call_id": tc.id,
                        "name": name, "content": result,
                    })

            # Ran out of steps holding tool output - ask once for the answer.
            messages.append({
                "role": "user",
                "content": "Give the final JSON object now, in exactly the requested shape.",
            })
            response = await self._complete(messages, False, deadline)
            return (response.choices[0].message.content or "").strip()

        except Exception as exc:
            self.on_event({"type": "agent_failed", "error": f"{type(exc).__name__}: {exc}"})
            return ""
