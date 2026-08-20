"""Central configuration. Everything tunable lives here, nothing is hard-coded downstream."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

# Standard search: walks up from the working directory.
load_dotenv()

def _find_root() -> Path:
    """Locate the project root without assuming how deep this file is buried.
    """
    env = os.getenv("DRC_ROOT")
    if env:
        return Path(env).expanduser().resolve()
    here = Path(__file__).resolve()
    for parent in (here.parent, *here.parents):
        if (parent / "pyproject.toml").exists() or (parent / ".git").is_dir():
            return parent
    return Path.cwd().resolve()


ROOT = _find_root()

# Belt and braces: also load the .env sitting next to these files, in case the
# working directory is elsewhere (a script run from a parent, say). override
# is False, so anything already in the real environment still wins.
load_dotenv(ROOT / ".env", override=False)

DATA = ROOT / "data"
RUNS = ROOT / "runs"
FIGS = ROOT / "figures"
CACHE = ROOT / ".cache"


def ensure_dirs() -> None:
    """Create the output directories. Called before writing, never on import.
    """
    for d in (DATA, RUNS, FIGS, CACHE):
        try:
            d.mkdir(parents=True, exist_ok=True)
        except PermissionError as exc:
            raise PermissionError(
                f"Cannot create {d} (project root resolved to {ROOT}).\n"
                f"Point the project somewhere writable:\n"
                f"    export DRC_ROOT=$PWD\n"
                f"or in the notebook, before importing drc:\n"
                f"    import os; os.environ['DRC_ROOT'] = str(Path.cwd())"
            ) from exc

BROWSECOMP_URL = (
    "https://openaipublic.blob.core.windows.net/simple-evals/browse_comp_test_set.csv"
)
SIMPLEQA_URL = (
    "https://openaipublic.blob.core.windows.net/simple-evals/simple_qa_test_set.csv"
)

# USD per 1M tokens. Update these; they are the only place cost is defined.
# Free-tier models are priced at 0 -- the binding constraint there is requests
# per day, not dollars, so see FREE_TIER_RPD below.
PRICING: dict[str, tuple[float, float]] = {
    # --- free tiers ------------------------------------------------------
    "google_genai:gemini-2.5-flash": (0.0, 0.0),
    "google_genai:gemini-2.5-flash-lite": (0.0, 0.0),
    "groq:llama-3.3-70b-versatile": (0.0, 0.0),
    "groq:llama-3.1-8b-instant": (0.0, 0.0),
    "ollama:llama3.1": (0.0, 0.0),          # fully local, no quota at all
    # --- paid ------------------------------------------------------------
    "openai:gpt-4.1-mini": (0.40, 1.60),
    "openai:gpt-4.1": (2.00, 8.00),
    "anthropic:claude-haiku-4-5": (1.00, 5.00),
    "anthropic:claude-sonnet-4-5": (3.00, 15.00),
}

# Requests-per-day caps on the free tiers. These are the real budget when the
# dollar cost is zero: the study is limited by wall-clock days, not money.
# Verify current values before planning -- providers change them often.
FREE_TIER_RPD: dict[str, int] = {
    "google_genai": 1500,   # Flash / Flash-Lite; 15 RPM
    "groq": 1000,           # most models; 30 RPM
    "ollama": 10**9,        # local: unlimited, bounded by your hardware
}

# Per-search cost. Zero for the keyless DuckDuckGo backend.
SEARCH_COST_USD = {"tavily": 0.008, "duckduckgo": 0.0}


@dataclass
class Settings:
    # --- benchmark ----------------------------------------------------------
    # "simpleqa"   short verifiable questions; a weak/free model lands in a
    #              measurable accuracy band. The right default for a free run.
    # "browsecomp" deliberately brutal; a free model scores near zero, which
    #              makes calibration unmeasurable -- every point in one bin.
    benchmark: str = os.getenv("BENCHMARK", "simpleqa")

    # --- models -------------------------------------------------------------
    researcher_model: str = os.getenv("RESEARCHER_MODEL",
                                      "google_genai:gemini-2.5-flash-lite")
    grader_model: str = os.getenv("GRADER_MODEL",
                                  "google_genai:gemini-2.5-flash")

    # --- search -------------------------------------------------------------
    # "duckduckgo" keyless and free, lower quality, rate-limited
    # "tavily"     better results, needs a key, ~$0.008/search
    search_backend: str = os.getenv("SEARCH_BACKEND", "duckduckgo")

    # Deliberately weak researcher. If accuracy is ~95% every point lands in the
    # top reliability bin and calibration becomes unmeasurable. 45-65% is the
    # best-conditioned regime for this experiment -- and the cheapest.
    temperature: float = 1.0

    # --- ensemble -----------------------------------------------------------
    n_max: int = int(os.getenv("N_MAX", 5))  # runs actually executed per question
    max_search_calls: int = 6                # per researcher, hard stop
    # Free tiers cap requests per minute (Gemini 15 RPM, Groq 30 RPM). Running
    # 8 researchers at once guarantees 429s, so keep this low and let the
    # retry/backoff in researcher.py absorb the rest.
    concurrency: int = int(os.getenv("CONCURRENCY", 2))
    max_retries: int = 5                     # on rate limits / transient errors

    # --- benchmark ----------------------------------------------------------
    n_questions: int = int(os.getenv("N_QUESTIONS", 150))
    calib_frac: float = 0.4                  # rest is held-out test
    split_seed: int = 0

    # --- analysis -----------------------------------------------------------
    n_bins: int = 10
    bin_strategy: str = "quantile"           # "quantile" or "uniform"
    n_bootstrap: int = 2000
    sweep_sizes: tuple[int, ...] = (1, 2, 3, 4, 5, 6, 7, 8)
    n_subsamples: int = 200                  # random n-subsets drawn per question

    # --- diversity ----------------------------------------------------------
    # Temperature sampling alone is the clean experiment: members differ only by
    # sampling noise. Prompt-based diversity raises spread but confounds the
    # confidence signal with prompt effects. Off by default; report both if used.
    use_diversity_prompts: bool = False

    paths: dict = field(default_factory=lambda: {
        "questions": DATA / "questions.jsonl",
        "runs": RUNS / "ensemble_runs.jsonl",
        "graded": RUNS / "graded.jsonl",
        "results": RUNS / "results.json",
    })


settings = Settings()


def provider(model_id: str) -> str:
    """'google_genai:gemini-2.5-flash' -> 'google_genai'."""
    return model_id.split(":", 1)[0]


def is_free(model_id: str) -> bool:
    return PRICING.get(model_id, (1.0, 1.0)) == (0.0, 0.0)


def estimate_days(n_questions: int, n_max: int, calls_per_run: float = 6.0,
                  model_id: str | None = None) -> float:
    """Wall-clock days to finish, given a free tier's requests-per-day cap.

    When the model is free the budget stops being money and becomes time. A
    150-question study at N=5 is roughly 4,500 requests; on Gemini's 1,500/day
    that is three days of running, not an afternoon. Better to know that up
    front than to discover it on day two.
    """
    model_id = model_id or settings.researcher_model
    rpd = FREE_TIER_RPD.get(provider(model_id))
    if not rpd:
        return 0.0
    return n_questions * n_max * calls_per_run / rpd
