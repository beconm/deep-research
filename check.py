"""Verify the environment without spending anything: python check.py

Checks, in the order they break:
  1. the right interpreter (are you actually in the venv?)
  2. analysis packages
  3. agent packages, and whether the agent factory resolves
  4. project modules
  5. .env keys present and non-empty
  6. .gitignore protects the decrypted benchmark
  7. the numerical tests

Exits 0 if everything the pipeline needs is present, 1 otherwise.
"""
from __future__ import annotations

import importlib
import os
import subprocess
import sys
from pathlib import Path

OK, BAD, WARN = "  OK  ", " FAIL ", " WARN "
failures: list[str] = []
warnings_: list[str] = []


def line(status: str, label: str, detail: str = "") -> None:
    print(f"[{status}] {label:<28} {detail}")


def check_interpreter() -> None:
    in_venv = sys.prefix != sys.base_prefix or bool(os.getenv("CONDA_PREFIX"))
    ver = ".".join(map(str, sys.version_info[:3]))
    if not in_venv:
        failures.append(
            "Not inside a virtualenv. Run:  source ~/.venv/bin/activate")
        line(BAD, "virtualenv", f"python {ver} at {sys.prefix}")
    elif sys.version_info[:2] < (3, 10):
        failures.append(f"Python {ver} is too old; need 3.10+")
        line(BAD, "virtualenv", f"python {ver}")
    else:
        line(OK, "virtualenv", f"python {ver}")
        line(OK, "interpreter", sys.executable)


def check_packages(names: list[str], group: str) -> None:
    for name in names:
        try:
            mod = importlib.import_module(name)
            line(OK, name, getattr(mod, "__version__", "installed"))
        except ImportError as exc:
            failures.append(f"{group}: {name} missing ({exc})")
            line(BAD, name, "not installed")


def check_agent_factory() -> None:
    try:
        from researcher import _load_agent_factory
        f = _load_agent_factory()
        line(OK, "agent factory", f"{f.__module__}.{f.__name__}")
    except Exception as exc:  # noqa: BLE001
        failures.append(f"agent factory unresolved: {exc}")
        line(BAD, "agent factory", str(exc).splitlines()[0][:60])


def check_project_modules() -> None:
    mods = ["config", "benchmark", "cache", "calibration", "ensemble",
            "grader", "plots", "sweep", "researcher"]
    missing = []
    for m in mods:
        try:
            importlib.import_module(m)
        except Exception as exc:  # noqa: BLE001
            missing.append(f"{m} ({exc})")
    if missing:
        failures.append("project modules failed to import: " + ", ".join(missing))
        line(BAD, "project modules", missing[0][:60])
    else:
        line(OK, "project modules", f"{len(mods)} modules")


def check_env() -> None:
    env_path = Path.cwd() / ".env"
    if not env_path.exists():
        failures.append(
            ".env not found. Create it:\n"
            "      printf 'OPENAI_API_KEY=sk-...\\nTAVILY_API_KEY=tvly-...\\n' > .env")
        line(BAD, ".env file", "missing")
        return
    line(OK, ".env file", str(env_path))

    try:
        from dotenv import load_dotenv
        load_dotenv(env_path, override=False)
    except ImportError:
        # python-dotenv missing is already reported above; fall back to a
        # minimal parse so the key check still runs and reports something useful.
        for raw in env_path.read_text().splitlines():
            raw = raw.strip()
            if raw and not raw.startswith("#") and "=" in raw:
                k, _, v = raw.partition("=")
                os.environ.setdefault(k.strip(), v.strip())
    import config
    keymap = {"openai": "OPENAI_API_KEY", "google_genai": "GOOGLE_API_KEY",
              "groq": "GROQ_API_KEY", "anthropic": "ANTHROPIC_API_KEY",
              "ollama": None}
    required = {keymap.get(config.provider(m)) for m in
                (config.settings.researcher_model, config.settings.grader_model)}
    required.discard(None)
    if config.settings.search_backend == "tavily":
        required.add("TAVILY_API_KEY")
    if not required:
        line(OK, "API keys", "none needed (local models + keyless search)")
    for key in sorted(required):
        val = (os.getenv(key) or "").strip()
        if not val or val.endswith("...") or val in {"sk-", "tvly-"}:
            failures.append(f"{key} is empty or still a placeholder in .env")
            line(BAD, key, "empty / placeholder")
        else:
            line(OK, key, f"{val[:6]}...{val[-4:]}  ({len(val)} chars)")
    if os.getenv("LANGSMITH_API_KEY"):
        line(OK, "LANGSMITH_API_KEY", "set (optional)")


def check_live_apis(skip: bool = False) -> None:
    """Actually call both APIs. This is the check that matters.

    Keys being present is not the same as keys being valid. A placeholder key
    passes every static check, then fails on all N x M research runs -- which
    completes, costs nothing, and leaves a dataset of blanks. Two tiny calls
    here cost a fraction of a cent and rule that out.
    """
    if skip:
        line(WARN, "live API test", "skipped (--no-live)")
        return

    import config
    model_id = config.settings.researcher_model
    backend = config.settings.search_backend

    # --- model ------------------------------------------------------------
    try:
        from langchain.chat_models import init_chat_model
        model = init_chat_model(model_id, temperature=0)
        resp = model.invoke("Reply with the single word: ok")
        text = resp.content if isinstance(resp.content, str) else str(resp.content)
        tag = "FREE" if config.is_free(model_id) else "paid"
        line(OK, "model call", f"{model_id} [{tag}] -> {text.strip()[:16]!r}")
    except Exception as exc:  # noqa: BLE001
        name = type(exc).__name__
        key = {"openai": "OPENAI_API_KEY", "google_genai": "GOOGLE_API_KEY",
               "groq": "GROQ_API_KEY", "anthropic": "ANTHROPIC_API_KEY"
               }.get(config.provider(model_id), "the provider key")
        hint = (f"key invalid or expired -- check {key} in .env"
                if "Auth" in name or "401" in str(exc) or "API key" in str(exc)
                else str(exc)[:70])
        failures.append(f"model call failed ({name}): {hint}")
        line(BAD, "model call", f"{name}: {hint[:44]}")

    # --- search -----------------------------------------------------------
    try:
        import asyncio
        from researcher import BACKENDS
        call = BACKENDS[backend](max_results=1)
        out = asyncio.run(call("capital of France"))
        line(OK, f"{backend} search", f"{len(str(out))} chars returned")
    except Exception as exc:  # noqa: BLE001
        name = type(exc).__name__
        failures.append(f"{backend} search failed ({name}): {str(exc)[:70]}")
        line(BAD, f"{backend} search", f"{name}: {str(exc)[:44]}")


def check_stale_runs() -> None:
    """Warn if the runs file is poisoned with failures.

    Resume logic skips any (qid, member) already recorded, so a file full of
    errored rows makes a re-run silently do nothing.
    """
    import json
    p = Path("runs/ensemble_runs.jsonl")
    if not p.exists():
        return
    total = failed = 0
    with open(p) as f:
        for l in f:
            if l.strip():
                total += 1
                if json.loads(l).get("error"):
                    failed += 1
    if failed:
        warnings_.append(
            f"runs/ensemble_runs.jsonl holds {failed}/{total} FAILED runs. "
            "Resume will skip them and your re-run will do nothing. Clear it:\n"
            "      python -c \"from researcher import prune_failed_runs as p; "
            "p('runs/ensemble_runs.jsonl')\"\n"
            "      rm -f runs/grader_cache.json")
        line(WARN, "existing runs", f"{failed}/{total} failed records")
    else:
        line(OK, "existing runs", f"{total} records, none failed")


def check_gitignore() -> None:
    """data/questions.jsonl holds decrypted BrowseComp problems and answers."""
    if not (Path.cwd() / ".git").is_dir():
        line(OK, ".gitignore", "not a git repo yet")
        return
    gi = Path.cwd() / ".gitignore"
    covered = gi.exists() and any(
        l.strip() in {"data/", "data", "data/*"} for l in gi.read_text().splitlines())
    if covered:
        line(OK, ".gitignore", "data/ excluded")
    else:
        warnings_.append(
            "git repo without 'data/' in .gitignore -- questions.jsonl contains "
            "DECRYPTED BrowseComp answers and must not be committed")
        line(WARN, ".gitignore", "data/ NOT excluded")


def check_tests() -> None:
    for t in ("test_numerics.py", "test_tiebreak.py"):
        if not Path(t).exists():
            warnings_.append(f"{t} not found")
            line(WARN, t, "not found")
            continue
        r = subprocess.run([sys.executable, t], capture_output=True, text=True)
        if r.returncode == 0:
            line(OK, t, r.stdout.strip().splitlines()[-1][:50])
        else:
            failures.append(f"{t} failed")
            line(BAD, t, (r.stderr.strip().splitlines() or [""])[-1][:60])


def main() -> int:
    if not (Path.cwd() / "config.py").exists():
        print(f"config.py not found in {Path.cwd()}")
        print("Run this from the deep-research directory:  cd ~/deep-research")
        return 1

    print(f"\nchecking {Path.cwd()}\n" + "-" * 62)
    check_interpreter()
    print("-" * 62)
    check_packages(["numpy", "scipy", "sklearn", "matplotlib", "pandas",
                    "dotenv", "tqdm"], "analysis")
    print("-" * 62)
    check_packages(["langchain", "langgraph", "langchain_openai",
                    "langchain_tavily"], "agent")
    check_agent_factory()
    print("-" * 62)
    check_project_modules()
    check_env()
    check_gitignore()
    check_stale_runs()
    print("-" * 62)
    check_live_apis(skip="--no-live" in sys.argv or bool(failures))
    print("-" * 62)
    check_tests()
    print("-" * 62)

    for w in warnings_:
        print(f"\nWARNING: {w}")
    if failures:
        print(f"\n{len(failures)} problem(s):\n")
        for f in failures:
            print(f"  - {f}")
        return 1
    print("\nEverything checks out. Next:")
    print("    jupyter lab pipeline.ipynb      (kernel: drc)")
    print("Step 2 of the notebook prices the full run before you commit to it.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
