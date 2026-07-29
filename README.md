# Data Analyst Telegram Bot

An LLM agent that answers data-analysis questions sent over Telegram and replies
with **exactly one JSON object**, shaped exactly as the question asked.

```
{"answer": {"state": "Assam"}, "log_url": "https://.../run.jsonl"}
```

## How it works

```
Telegram message
      │
   bot.py        per-chat history, per-chat ordering, always replies
      │
   agent.py      LLM loop with tools (run_python, fetch_url), hard deadline
      │
   shape.py      forces the reply into the exact JSON shape the question asked
      │
   runlog.py     append-only run.jsonl, optionally pushed so log_url stays live
      │
   reply
```

| File | Purpose |
| --- | --- |
| `bot.py` | Telegram entrypoint. Run this. |
| `agent.py` | The agent: prompt, tool loop, timeouts, retries. |
| `shape.py` | Parses the requested JSON shape and conforms the reply to it. |
| `runlog.py` | JSONL run log + publishing. `python runlog.py` validates the log. |
| `eval_local.py` | Runs the eval questions with no Telegram needed. |
| `smoke_test.py` | Checks the AI Pipe token and model are working. |
| `tests/test_shape.py` | Offline tests for the shape logic. |

## The rule everything is built around

The public grader compares whole objects:

```python
answer = json.loads(replies[-1].strip())
ok = answer == expected
```

So a right answer with one extra key scores **zero**. `shape.py` reads the
template in the question (`Reply with ONLY a JSON object like {"sum": 200}`),
takes its top-level keys, and emits precisely those — dropping keys the model
invented, and adding `log_url` **only when the question asked for one**.

## Setup

```bash
python3.12 -m venv venv
venv/bin/pip install -r requirements.txt
cp .env.example .env      # then fill in the three required values
```

Required in `.env`:

| Variable | Where it comes from |
| --- | --- |
| `TELEGRAM_BOT_TOKEN` | @BotFather → `/newbot` |
| `AIPIPE_TOKEN` | <https://aipipe.org/login> |
| `LOG_URL` | the **Raw** GitHub URL to `run.jsonl` in your public repo |

All other settings are optional and documented in `.env.example`.

## Verify before deploying

```bash
venv/bin/python smoke_test.py         # token + model reachable
venv/bin/python tests/test_shape.py   # shape logic (offline, no API)
venv/bin/python eval_local.py         # full agent vs the eval questions
venv/bin/python runlog.py             # every log line is valid JSON
venv/bin/python bot.py                # then message the bot from Telegram
```

`eval_local.py` reproduces the grader's comparison exactly, so a pass there is a
pass in the harness — it just skips Telegram, which makes iteration far faster.

## Testing against the real harness

```bash
cd tds-p1-t2-2026-telegram-bot
pip install -r requirements.txt      # needs telethon
python login.py                      # one-time; prints TELEGRAM_SESSION_STRING
python generate.py --students students.csv
python collect.py  --students students.csv
python grade.py    --students students.csv
```

`collect.py` only retries results whose status is `error`. **Delete `data/` before
re-collecting**, or previously recorded `ok` rows are reused and you will grade
stale replies.

## Deploying

The bot polls Telegram and listens on no port, so deploy it as a **worker**, not
a web service. `Procfile` already declares `worker: python bot.py`.

- **Render** — New → Background Worker, build `pip install -r requirements.txt`,
  start `python bot.py`, add the three env vars in the dashboard.
- **Railway** — same flow, deploy as a worker process.
- **VPS** — a systemd unit so it restarts on reboot.

Only one instance may poll a token at a time; a second one gets `Conflict` and
exits. Stop any local run before the deployed one matters.

## Publishing the log

`log_url` must be fetchable by a stranger with `wget` and no login. Commit
`run.jsonl` to the public repo and use its **Raw** URL. Check it from a private
browser window before grading day.

Set `PUBLISH_EVERY=N` to have the bot commit and push `run.jsonl` itself every N
events, if the host has push credentials. Otherwise push periodically by hand.

## Secrets

`.env` is gitignored and tokens are read only from the environment — nothing is
hardcoded and the bot never prints a token. Never commit a real token: anyone who
has it can hijack the bot or spend the AI credits.
