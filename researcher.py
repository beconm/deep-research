"""One research run: a ReAct agent with a cached web-search tool.

This is the only module that touches the LangChain/LangGraph API surface. That
is deliberate -- the framework moves fast, and everything that constitutes the
actual contribution (ensemble aggregation, calibration, the sweep) is plain
numpy downstream. If a release breaks something, you fix this file and nothing
else.

API note: LangChain 1.0 (Oct 2025) removed `initialize_agent` / `AgentExecutor`.
`create_agent` is the current idiom and LangGraph is the runtime beneath it.
"""
from __future__ import annotations

import asyncio
import re
from dataclasses import dataclass, field

from cache import SearchCache
from config import settings

# Mirrors the official BrowseComp query template closely enough that the parsed
# fields line up with the reference grader -- including the confidence field,
# which gives the verbalised-confidence baseline for free.
RATE_LIMIT_MARKERS = ("429", "rate limit", "ratelimit", "quota", "resource_exhausted",
                      "too many requests")


async def _with_retry(coro_factory, what: str = "call", attempts: int | None = None):
    """Exponential backoff on rate limits.

    Free tiers are capped per minute -- Gemini at 15 RPM, Groq at 30 -- so 429s
    are the normal case rather than an exception, and a run without backoff
    turns them into a wall of empty results. Fatal errors are re-raised
    immediately; retrying an invalid key just wastes the afternoon.
    """
    import asyncio
    import random as _random

    attempts = attempts or settings.max_retries
    delay = 2.0
    last: Exception | None = None
    for attempt in range(attempts):
        try:
            return await coro_factory()
        except Exception as exc:  # noqa: BLE001
            last = exc
            msg = f"{type(exc).__name__}: {exc}".lower()
            if is_fatal(type(exc).__name__):
                raise
            if not any(m in msg for m in RATE_LIMIT_MARKERS) and attempt >= 1:
                raise
            await asyncio.sleep(delay + _random.uniform(0, 1))
            delay *= 2
    raise last  # type: ignore[misc]


def _load_agent_factory():
    """Locate the agent constructor across LangChain versions.

    `create_agent` moved during the 1.x line -- it was reported missing from
    `langchain.agents` in 1.1.0 with no note in the release notes -- so try the
    known locations rather than betting on one and failing with a bare
    ImportError halfway through a paid run.
    """
    import importlib

    attempts = [
        ("langchain.agents", "create_agent"),
        ("langchain.agents.factory", "create_agent"),
        ("langgraph.prebuilt", "create_react_agent"),
    ]
    problems = []
    for mod_name, attr in attempts:
        try:
            mod = importlib.import_module(mod_name)
            return getattr(mod, attr)
        except (ImportError, AttributeError) as exc:
            problems.append(f"  {mod_name}.{attr}: {exc}")
    raise ImportError(
        "No agent factory found. Tried:\n" + "\n".join(problems) +
        "\n\nInstall the agent stack:  python -m pip install -r requirements.txt"
    )


def _build_agent(factory, model, tools, prompt):
    """Call the factory with whichever prompt keyword it accepts.

    `create_agent` takes `system_prompt=`; LangGraph's `create_react_agent`
    takes `prompt=`. Inspect rather than guess.
    """
    import inspect

    params = inspect.signature(factory).parameters
    kw = {"model": model, "tools": tools}
    if "system_prompt" in params:
        kw["system_prompt"] = prompt
    elif "prompt" in params:
        kw["prompt"] = prompt
    return factory(**kw)


QUERY_TEMPLATE = """{problem}

Your response must end with exactly this format:

Explanation: {{your reasoning}}
Exact Answer: {{succinct, standalone answer}}
Confidence: {{your confidence as a percentage, 0-100}}

The Exact Answer must be as short as possible -- a name, a number, a date.
Do not hedge in the Exact Answer field; put uncertainty in Confidence.
"""

DIVERSITY_HINTS = [
    "",
    "Begin from primary sources and official records.",
    "Begin from secondary literature and press coverage.",
    "Work backwards from the constraints in the question, narrowing candidates.",
    "Enumerate plausible candidates first, then eliminate.",
    "Prioritise recent sources over older ones.",
    "Prioritise authoritative or archival sources over recent ones.",
    "Cross-check any candidate answer against a second independent source.",
]


# Errors that will never resolve by retrying. Hitting one of these means the
# configuration is wrong, and continuing just burns wall-clock producing empty
# records -- which is exactly what a 2400-run AuthenticationError sweep looks
# like: it completes, costs nothing, and yields a dataset of blanks.
FATAL_ERRORS = (
    "AuthenticationError", "PermissionDeniedError", "NotFoundError",
    "APIStatusError", "BadRequestError", "InvalidRequestError",
    "MissingAPIKeyError", "UnauthorizedError",
)


def is_fatal(err: str | None) -> bool:
    return bool(err) and err.split(":")[0].strip() in FATAL_ERRORS


@dataclass
class ResearchRun:
    qid: str
    member: int
    exact_answer: str
    explanation: str
    stated_confidence: float | None  # 0-1, from the model's own mouth
    input_tokens: int = 0
    output_tokens: int = 0
    search_calls: int = 0
    billable_searches: int = 0
    error: str | None = None
    raw: str = field(default="", repr=False)

    @property
    def fatal(self) -> bool:
        return is_fatal(self.error)


def _parse(text: str) -> tuple[str, str, float | None]:
    def grab(label: str) -> str:
        m = re.search(rf"{label}\s*:\s*(.+?)(?=\n\s*(?:Explanation|Exact Answer|Confidence)\s*:|\Z)",
                      text, re.S | re.I)
        return m.group(1).strip() if m else ""

    explanation = grab("Explanation")
    answer = grab("Exact Answer")
    conf_raw = grab("Confidence")
    m = re.search(r"(\d+(?:\.\d+)?)", conf_raw)
    conf = min(max(float(m.group(1)) / 100.0, 0.0), 1.0) if m else None
    return answer, explanation, conf


def _tavily_backend(max_results: int = 5):
    from langchain_tavily import TavilySearch
    tool = TavilySearch(max_results=max_results)

    async def call(query: str) -> str:
        return str(await tool.ainvoke({"query": query}))
    return call


def _duckduckgo_backend(max_results: int = 5):
    """Keyless, free, and noticeably worse than Tavily.

    No API key and no per-search charge, which is what makes a zero-budget run
    possible at all. The tradeoff is real: snippets are shorter, ranking is
    weaker, and DuckDuckGo rate-limits aggressively -- so retrieval quality,
    not the model, may become the thing your accuracy is measuring. Say so in
    the writeup rather than comparing these numbers to a Tavily run.
    """
    import asyncio

    try:
        from ddgs import DDGS
    except ImportError:  # older package name
        from duckduckgo_search import DDGS  # type: ignore

    def _sync(query: str) -> str:
        with DDGS() as ddgs:
            hits = list(ddgs.text(query, max_results=max_results))
        if not hits:
            return "No results."
        return "\n\n".join(
            f"{h.get('title','')}\n{h.get('href','')}\n{h.get('body','')}"
            for h in hits
        )

    async def call(query: str) -> str:
        # DDGS is synchronous; keep the event loop free.
        return await asyncio.to_thread(_sync, query)
    return call


BACKENDS = {"tavily": _tavily_backend, "duckduckgo": _duckduckgo_backend}


def make_search_tool(cache: SearchCache, counters: dict):
    """Web search, disk-cached, over whichever backend is configured."""
    name = settings.search_backend
    if name not in BACKENDS:
        raise ValueError(f"unknown search_backend {name!r}; choose from {list(BACKENDS)}")
    backend = BACKENDS[name](max_results=5)

    async def search(query: str) -> str:
        """Search the web for information. Returns titles, URLs and snippets."""
        counters["search_calls"] += 1
        if counters["search_calls"] > settings.max_search_calls:
            return "Search budget exhausted. Answer with what you have."
        hit = cache.get(query, tool=name, max_results=5)
        if hit is not None:
            return hit
        text = await _with_retry(lambda: backend(query), what="search")
        cache.put(query, text, tool=name, max_results=5)
        counters["billable_searches"] += 1
        return text

    search.__name__ = "web_search"
    return search


async def run_one(
    qid: str, problem: str, member: int, cache: SearchCache
) -> ResearchRun:
    # Imported here, not at module scope, so the numerical tests stay runnable
    # without the framework installed.
    from langchain.chat_models import init_chat_model

    create_agent = _load_agent_factory()
    counters = {"search_calls": 0, "billable_searches": 0}
    tool = make_search_tool(cache, counters)

    model = init_chat_model(
        settings.researcher_model, temperature=settings.temperature
    )
    prompt = "You are a meticulous research assistant. Search before answering."
    if settings.use_diversity_prompts:
        prompt += " " + DIVERSITY_HINTS[member % len(DIVERSITY_HINTS)]

    agent = _build_agent(create_agent, model, [tool], prompt)

    try:
        result = await _with_retry(
            lambda: agent.ainvoke(
                {"messages": [{"role": "user",
                               "content": QUERY_TEMPLATE.format(problem=problem)}]}
            ),
            what="agent",
        )
    except Exception as exc:  # noqa: BLE001 - one bad run must not kill the sweep
        return ResearchRun(qid, member, "", "", None,
                           search_calls=counters["search_calls"],
                           billable_searches=counters["billable_searches"],
                           error=f"{type(exc).__name__}: {exc}")

    messages = result["messages"]
    text = messages[-1].content
    if isinstance(text, list):  # content blocks
        text = "".join(b.get("text", "") for b in text if isinstance(b, dict))

    inp = out = 0
    for m in messages:
        usage = getattr(m, "usage_metadata", None) or {}
        inp += usage.get("input_tokens", 0)
        out += usage.get("output_tokens", 0)

    answer, explanation, conf = _parse(text)
    return ResearchRun(
        qid=qid, member=member, exact_answer=answer, explanation=explanation,
        stated_confidence=conf, input_tokens=inp, output_tokens=out,
        search_calls=counters["search_calls"],
        billable_searches=counters["billable_searches"], raw=text,
    )


async def run_ensemble(
    qid: str, problem: str, n: int, cache: SearchCache
) -> list[ResearchRun]:
    """N independent researchers on the SAME question -- this is the change.

    Upstream, the supervisor splits a brief into subtopics and runs one agent per
    subtopic. Here every member answers the whole question, so their answers are
    comparable and their agreement is a signal rather than a coincidence.
    """
    sem = asyncio.Semaphore(settings.concurrency)

    async def guarded(member: int):
        async with sem:
            return await run_one(qid, problem, member, cache)

    return list(await asyncio.gather(*(guarded(i) for i in range(n))))


def prune_failed_runs(path) -> tuple[int, int]:
    """Drop errored records from the runs JSONL so a re-run actually re-runs.

    The resume logic skips any (qid, member) already present in the file. That
    is what you want for a genuine interruption -- and exactly wrong after a
    misconfigured sweep, because every question looks "done" and the re-run is
    a no-op. Strip the failures first.

    Returns (kept, dropped).
    """
    import json
    from pathlib import Path

    path = Path(path)
    if not path.exists():
        return (0, 0)

    kept, dropped = [], 0
    with open(path) as f:
        for line in f:
            if not line.strip():
                continue
            rec = json.loads(line)
            if rec.get("error"):
                dropped += 1
            else:
                kept.append(line)

    if dropped:
        backup = path.with_suffix(path.suffix + ".bak")
        path.replace(backup)
        with open(path, "w") as f:
            f.writelines(kept)
        print(f"pruned {dropped} failed runs, kept {len(kept)}"
              f" (original saved as {backup.name})")
    return (len(kept), dropped)
