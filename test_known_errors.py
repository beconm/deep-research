"""Regression test for is_fatal() against error shapes actually encountered
running this pipeline on free-tier keys.

Each string below is a real error this project hit at some point: a
deprecated model, a rate limit, a malformed tool call (see POSTMORTEM.md
for the full account of the first two). is_fatal() must keep classifying
each one correctly -- a fatal error should stop a sweep immediately rather
than retry uselessly; a recoverable one should not be treated as fatal, or
backoff and the fallback path never get a chance to run. If a future
change to is_fatal() or FATAL_STATUS_MARKERS breaks one of these, this
test catches it before a real run does -- the same pattern test_tiebreak.py
already uses for the tie-break regression.

Run directly:  python test_known_errors.py
Collected by check.py's check_tests() alongside the other test_*.py files.
"""
from __future__ import annotations

import sys

from researcher import is_fatal

FAIL = []


def check(label: str, condition: bool) -> None:
    print(f"  [{'ok' if condition else 'FAIL'}] {label}")
    if not condition:
        FAIL.append(label)


# --- fatal: must stop a sweep immediately, not retry ------------------------

GEMINI_MODEL_RETIRED = (
    "ChatGoogleGenerativeAIError: Error calling model 'gemini-2.5-flash-lite' "
    "(NOT_FOUND): 404 NOT_FOUND. {'error': {'code': 404, 'message': 'This "
    "model models/gemini-2.5-flash-lite is no longer available to new "
    "users. Please update your code to use models/gemini-3.5-flash-lite "
    "for the latest features and improvements.', 'status': 'NOT_FOUND'}}"
)

GROQ_MODEL_DEPRECATED = (
    "Error code: 404 - {'error': {'message': 'The model "
    "`llama-3.3-70b-versatile` does not exist or you do not have access "
    "to it.', 'type': 'invalid_request_error', 'code': 'model_not_found'}}"
)

MALFORMED_TOOL_CALL = (
    "BadRequestError: Error code: 400 - {'error': {'message': \"Tool call "
    "validation failed: tool call validation failed: parameters for tool "
    "web_search did not match schema: errors: [missing properties: "
    "'query']\", 'type': 'invalid_request_error', 'code': 'tool_use_failed', "
    "'failed_generation': '{\"name\": \"web_search\", \"arguments\": "
    "{\"cursor\": 2, \"id\": 0}}'}}"
)

# --- recoverable: must NOT be fatal, or backoff/fallback never runs --------

GROQ_TOKEN_RATE_LIMIT = (
    "RateLimitError: Error code: 429 - {'error': {'message': 'Rate limit "
    "reached for model `openai/gpt-oss-120b` in organization "
    "`org_01m0fxa07ze8yt66p9mrc5kzhk` service tier `on_demand` on tokens "
    "per day (TPD): Limit 200000, Used 198639, Requested 2326. Please try "
    "again in 6m56.88s.', 'type': 'tokens', 'code': 'rate_limit_exceeded'}}"
)


def main() -> int:
    print("Errors that must be classified fatal (stop the sweep immediately):")
    check("Gemini model retired for new API keys", is_fatal(GEMINI_MODEL_RETIRED))
    check("Groq model deprecated", is_fatal(GROQ_MODEL_DEPRECATED))
    check("Malformed tool call (bad arguments)", is_fatal(MALFORMED_TOOL_CALL))

    print("\nErrors that must NOT be classified fatal (retry/fallback should run):")
    check("Groq token-per-day rate limit", not is_fatal(GROQ_TOKEN_RATE_LIMIT))

    print()
    if FAIL:
        print(f"{len(FAIL)} check(s) failed:")
        for f in FAIL:
            print(f"  - {f}")
        return 1
    print("All known-error classifications still correct.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
