"""Tests for resilience.py and tracing.py. No API calls, no framework import.

Run directly:      python test_resilience.py
Collected by check.py's check_tests() alongside test_numerics.py,
test_tiebreak.py, and test_known_errors.py.
"""
from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

from resilience import try_with_fallback
from tracing import LocalTracer

FAIL = []


def check(label: str, condition: bool) -> None:
    print(f"  [{'ok' if condition else 'FAIL'}] {label}")
    if not condition:
        FAIL.append(label)


async def _ok(tag: str):
    async def call():
        return f"answer-from-{tag}"
    return call


async def _raises(tag: str, exc_type=RuntimeError):
    async def call():
        raise exc_type(f"{tag} is down")
    return call


async def test_primary_succeeds_no_fallback_needed():
    print("primary succeeds -> fallback never called")
    primary = await _ok("primary")

    async def fallback_should_not_run():
        raise AssertionError("fallback was called even though primary succeeded")

    result, outcome = await try_with_fallback(
        primary, fallback_should_not_run, primary_name="p", fallback_name="f"
    )
    check("result comes from primary", result == "answer-from-primary")
    check("outcome.model_used == primary name", outcome.model_used == "p")
    check("outcome.used_fallback is False", outcome.used_fallback is False)


async def test_primary_fails_fallback_succeeds():
    print("primary fails -> fallback takes over")
    primary = await _raises("primary")
    fallback = await _ok("fallback")

    result, outcome = await try_with_fallback(
        primary, fallback, primary_name="p", fallback_name="f"
    )
    check("result comes from fallback", result == "answer-from-fallback")
    check("outcome.model_used == fallback name", outcome.model_used == "f")
    check("outcome.used_fallback is True", outcome.used_fallback is True)
    check("primary_error was captured", outcome.primary_error is not None
          and "primary is down" in outcome.primary_error)


async def test_both_fail_raises_chained_error():
    print("primary and fallback both fail -> one raised error, both causes visible")
    primary = await _raises("primary")
    fallback = await _raises("fallback")

    try:
        await try_with_fallback(primary, fallback, primary_name="p", fallback_name="f")
        check("raised when both fail", False)
    except RuntimeError as exc:
        msg = str(exc)
        check("error message names the primary failure", "primary is down" in msg)
        check("error message names the fallback failure", "fallback is down" in msg)
        check("original fallback exception is chained (__cause__)", exc.__cause__ is not None)


async def test_no_fallback_configured_reraises_original():
    print("no fallback configured -> original exception propagates unchanged")
    primary = await _raises("primary", exc_type=ValueError)

    try:
        await try_with_fallback(primary, None, primary_name="p")
        check("raised when no fallback given", False)
    except ValueError as exc:
        check("original exception type preserved (not wrapped)", "primary is down" in str(exc))
    except RuntimeError:
        check("original exception type preserved (not wrapped)", False)


def test_local_tracer_writes_and_reads_spans(tmp_path: Path):
    print("LocalTracer: spans round-trip through the JSONL file")
    trace_file = tmp_path / "traces.jsonl"
    tracer = LocalTracer(path=str(trace_file))

    with tracer.span("unit.ok", qid="q1") as span:
        span.update(model_used="primary", success=True)

    try:
        with tracer.span("unit.fail", qid="q2") as span:
            span.update(model_used="primary")
            raise ValueError("boom")
    except ValueError:
        pass

    records = tracer.read_all()
    check("two spans recorded", len(records) == 2)
    ok_rec = next(r for r in records if r["name"] == "unit.ok")
    fail_rec = next(r for r in records if r["name"] == "unit.fail")
    check("success span has no error", ok_rec["error"] is None)
    check("success span carries update() fields", ok_rec.get("success") is True
          and ok_rec.get("model_used") == "primary")
    check("failed span records the exception", fail_rec["error"] is not None
          and "boom" in fail_rec["error"])
    check("failed span still carries fields set before the raise",
          fail_rec.get("model_used") == "primary")

    with open(trace_file) as f:
        lines = [json.loads(line) for line in f if line.strip()]
    check("file on disk is valid line-delimited JSON", len(lines) == 2)


async def _run_async_tests():
    await test_primary_succeeds_no_fallback_needed()
    await test_primary_fails_fallback_succeeds()
    await test_both_fail_raises_chained_error()
    await test_no_fallback_configured_reraises_original()


def main() -> int:
    import tempfile

    asyncio.run(_run_async_tests())
    with tempfile.TemporaryDirectory() as d:
        test_local_tracer_writes_and_reads_spans(Path(d))

    print()
    if FAIL:
        print(f"{len(FAIL)} check(s) failed:")
        for f in FAIL:
            print(f"  - {f}")
        return 1
    print("All resilience/tracing checks passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
