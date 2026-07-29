"""Offline tests for the shape enforcement that the grader's exact-match depends on.

Run: python3 -m pytest tests/ -q     (or: python3 tests/test_shape.py)
No network, no Telegram, no API key needed.
"""

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from shape import (conform, finalise, parse_model_json, requested_keys,
                   shape_from_conversation)

LOG = "https://raw.githubusercontent.com/u/r/main/run.jsonl"


def check(name, got, want):
    if got != want:
        raise AssertionError(f"{name}\n  got:  {got!r}\n  want: {want!r}")
    print(f"  ok  {name}")


# --- requested_keys -------------------------------------------------------

def test_requested_keys():
    check("simple shape",
          requested_keys('Find the sum of 45, 78. Reply with ONLY a JSON object like {"sum": 200}'),
          ["sum"])
    check("nested shape keeps only top level",
          requested_keys('Reply with ONLY this JSON object: {"answer": {"state": "<state name>"}, "log_url": "<url>"}'),
          ["answer", "log_url"])
    check("placeholder values are not valid JSON but keys still parse",
          requested_keys('{"answer": <your answer>, "log_url": "<public wget-able URL>"}'),
          ["answer", "log_url"])
    check("braces inside strings do not break scanning",
          requested_keys('Reply like {"note": "use {curly} braces", "n": 1}'),
          ["note", "n"])
    check("last template wins",
          requested_keys('Not like {"old": 1}. Reply with ONLY {"new": 2}'),
          ["new"])
    check("no template", requested_keys("Remember these numbers: 12, 25, 48."), None)
    check("smart quotes",
          requested_keys('Reply with ONLY a JSON object like {“capital”: “Bengaluru”}'),
          ["capital"])


# --- multi-turn shape resolution -----------------------------------------

def test_multi_turn():
    turns = ["Remember these numbers: 12, 25, 48.",
             'Now tell me the largest one. Reply with ONLY a JSON object like {"largest": 48}']
    check("last turn is authoritative", shape_from_conversation(turns), (["largest"], False))

    turns2 = ['Use shape {"v": 1}', "Now answer it."]
    check("falls back to earlier turn", shape_from_conversation(turns2), (["v"], False))

    check("no template anywhere", shape_from_conversation(["hi", "there"]), (None, False))
    check("bare log_url mention", shape_from_conversation(["send log_url too"]), (None, True))


# --- model output parsing ------------------------------------------------

def test_parse_model_json():
    check("plain", parse_model_json('{"sum": 200}'), {"sum": 200})
    check("fenced", parse_model_json('```json\n{"sum": 200}\n```'), {"sum": 200})
    check("prose prefix", parse_model_json('Sure! Here you go: {"sum": 200}'), {"sum": 200})
    check("prose suffix", parse_model_json('{"sum": 200}\nHope that helps!'), {"sum": 200})
    check("nested", parse_model_json('{"answer": {"state": "Assam"}}'), {"answer": {"state": "Assam"}})
    check("garbage", parse_model_json("I cannot answer that."), None)
    check("empty", parse_model_json(""), None)


# --- conform: THE regression this project was failing on ------------------

def test_conform_drops_uninvited_log_url():
    """The original bug: log_url appended to every reply, failing exact match."""
    got = conform({"sum": 200, "log_url": LOG}, ["sum"], LOG, False)
    check("uninvited log_url is dropped", got, {"sum": 200})


def test_conform_adds_requested_log_url():
    got = conform({"answer": {"state": "Assam"}}, ["answer", "log_url"], LOG, True)
    check("requested log_url is added", got, {"answer": {"state": "Assam"}, "log_url": LOG})


def test_conform_overwrites_hallucinated_log_url():
    got = conform({"answer": 30, "log_url": "https://example.com/nope.jsonl"},
                  ["answer", "log_url"], LOG, True)
    check("model's fake log_url replaced with the real one", got, {"answer": 30, "log_url": LOG})


def test_conform_drops_extra_keys():
    got = conform({"sum": 200, "confidence": 0.9, "reasoning": "added them"}, ["sum"], LOG, False)
    check("extra keys dropped", got, {"sum": 200})


def test_conform_renames_near_miss_key():
    got = conform({"result": 200}, ["sum"], LOG, False)
    check("unambiguous rename rescued", got, {"sum": 200})


def test_conform_case_insensitive_match():
    got = conform({"Largest": 96}, ["largest"], LOG, False)
    check("case-insensitive key match", got, {"largest": 96})


def test_conform_key_order():
    got = conform({"log_url": LOG, "answer": 1}, ["answer", "log_url"], LOG, True)
    check("key order follows the template", list(got), ["answer", "log_url"])


def test_conform_bare_scalar():
    check("bare scalar wrapped", conform(200, ["sum"], LOG, False), {"sum": 200})
    check("bare scalar with log_url", conform(30, ["answer", "log_url"], LOG, True),
          {"answer": 30, "log_url": LOG})


def test_conform_unknown_shape_passthrough():
    got = conform({"whatever": 1}, None, LOG, False)
    check("unknown shape passes through", got, {"whatever": 1})


def test_conform_missing_key_is_null_not_crash():
    got = conform({}, ["a", "b"], LOG, False)
    check("missing keys become null", got, {"a": None, "b": None})


# --- finalise: end-to-end over the real eval questions --------------------

EVAL_CASES = [
    ('Which state has the highest maternal mortality rate based on MOSPI data? '
     'Reply with ONLY a JSON object like {"state": "<state name>"}',
     '{"state": "Assam", "log_url": "https://example.com/run.jsonl"}',
     {"state": "Assam"}),
    ('What is 15% of 200? Reply with ONLY a JSON object like {"answer": 30}',
     '```json\n{"answer": 30}\n```',
     {"answer": 30}),
    ('Is 97 a prime number? Reply with ONLY a JSON object like {"prime": true}',
     'Yes. {"prime": true, "explanation": "97 has no divisors"}',
     {"prime": True}),
    ('Which state has the highest maternal mortality rate based on MOSPI data? '
     'Reply with ONLY this JSON object and nothing else: '
     '{"answer": {"state": "<state name>"}, "log_url": "<public wget-able URL>"}',
     '{"answer": {"state": "Assam"}}',
     {"answer": {"state": "Assam"}, "log_url": LOG}),
]


def test_finalise_end_to_end():
    for question, model_text, want in EVAL_CASES:
        reply, _ = finalise(model_text, [question], LOG)
        check(f"finalise: {question[:44]}...", json.loads(reply), want)
        # The grader does json.loads(replies[-1].strip()) - the reply must be
        # exactly one JSON object with no surrounding prose.
        check("  reply is bare JSON", reply.strip()[0] + reply.strip()[-1], "{}")


def main():
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    print(f"running {len(tests)} shape tests\n")
    for t in tests:
        print(t.__name__)
        t()
    print(f"\nall {len(tests)} shape tests passed")


if __name__ == "__main__":
    main()
