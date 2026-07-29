"""Smoke test: is the AI Pipe token valid and is the model reachable?

Run this first whenever the bot misbehaves - it separates "my credentials or the
upstream API are broken" from "my agent logic is broken".

    python smoke_test.py

Replaces the old test.py, which had a live AIPIPE token hardcoded in it.
"""

import os
import sys

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

BASE_URL = os.getenv("AIPIPE_BASE_URL", "https://aipipe.org/openai/v1")
MODEL = os.getenv("MODEL", "gpt-5-mini")


def token_expiry(token: str):
    """AI Pipe issues JWTs; decode the exp claim without verifying (no secret here)."""
    import base64
    import datetime
    import json
    try:
        payload = token.split(".")[1]
        payload += "=" * (-len(payload) % 4)
        claims = json.loads(base64.urlsafe_b64decode(payload))
        return claims.get("email"), datetime.datetime.fromtimestamp(claims["exp"])
    except Exception:
        return None, None


def main() -> int:
    token = os.getenv("AIPIPE_TOKEN", "").strip()
    if not token:
        print("AIPIPE_TOKEN is not set. Copy .env.example to .env and fill it in.",
              file=sys.stderr)
        return 2

    email, expires = token_expiry(token)
    if expires:
        import datetime
        left = expires - datetime.datetime.now()
        state = "EXPIRED" if left.total_seconds() < 0 else f"{left.days}d left"
        print(f"token: {email or 'unknown'}, expires {expires:%Y-%m-%d %H:%M} ({state})")
        if left.total_seconds() < 0:
            print("Get a fresh token at https://aipipe.org/login", file=sys.stderr)
            return 1

    client = OpenAI(base_url=BASE_URL, api_key=token, timeout=30)
    try:
        response = client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content": "Reply with the single word: ok"}],
        )
    except Exception as exc:
        print(f"API call failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1

    print(f"model {MODEL} replied: {response.choices[0].message.content!r}")
    print("smoke test passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
