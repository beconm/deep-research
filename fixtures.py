"""Test fixtures ONLY. Not part of the pipeline; never imported by the notebook.

Generates ensemble runs with known ground truth so the numerical code
(aggregation, calibration, sweep) can be tested without API calls. Nothing here
produces results -- it exists so `test_numerics.py` and `test_tiebreak.py` have
data with properties we control.

The generator deliberately includes the feature that makes this problem hard:

    **correlated errors.** Wrong members draw from a small pool of plausible
    distractors rather than each inventing a unique wrong answer. Pool size is
    what makes a question easy or treacherous: a one-distractor question puts
    high agreement behind a wrong answer, which is precisely the regime where
    agreement-as-confidence breaks.

    An earlier version of this generator gave every wrong member a unique
    answer. That made a single correct member win the plurality 56% of the time
    at 1/8 correct, which decoupled agreement from accuracy entirely and drove
    the point-biserial correlation to zero. The sanity check in the notebook
    caught it. Keep that check.

Everything produced here is labelled `synthetic=True` so it cannot be mistaken
for a real run downstream.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class SyntheticRun:
    qid: str
    member: int
    exact_answer: str
    explanation: str = ""
    stated_confidence: float = 0.5
    input_tokens: int = 0
    output_tokens: int = 0
    search_calls: int = 0
    billable_searches: int = 0
    error: str | None = None
    synthetic: bool = True
    raw: str = field(default="", repr=False)


def make_runs(
    n_questions: int = 300,
    n_max: int = 8,
    seed: int = 0,
    p_shared_error: float = 0.20,
    overconfidence: float = 0.55,
) -> tuple[dict[str, list[SyntheticRun]], dict[str, str], dict[str, str]]:
    """Return (runs_by_qid, truth_by_qid, split_by_qid).

    Parameters
    ----------
    p_shared_error
        Probability that a question has a *narrow* distractor pool (1-2 wrong
        candidates), so errors are strongly correlated. Otherwise the pool is
        wide (3 to n_max) and wrong answers scatter.
    overconfidence
        How much the members' *stated* confidence is inflated. The stated
        signal is mapped as p -> p**overconfidence, so values below 1 push
        confidence upward and reproduce the miscalibration the real agent shows.
    """
    rng = np.random.default_rng(seed)
    runs: dict[str, list[SyntheticRun]] = {}
    truth: dict[str, str] = {}
    split: dict[str, str] = {}

    for q in range(n_questions):
        qid = f"syn-{q:05d}"
        # Per-question difficulty: probability an individual member gets it right.
        # Beta(1.6, 1.6) gives a broad spread, so reliability bins get populated
        # across the whole [0, 1] range rather than piling up at one end.
        skill = float(rng.beta(1.6, 1.6))

        # Distractor pool. Narrow pool -> members that are wrong are wrong
        # together, and agreement stops tracking correctness.
        if rng.random() < p_shared_error:
            n_distractors = int(rng.integers(1, 3))
        else:
            n_distractors = int(rng.integers(3, n_max + 1))
        pool = [f"WRONG-{q}-{d}" for d in range(n_distractors)]
        # Skewed draw: some distractors are more tempting than others.
        weights = rng.dirichlet(np.full(n_distractors, 0.7))

        members = []
        for m in range(n_max):
            correct = rng.random() < skill
            if correct:
                answer = "TRUE-ANSWER"
            else:
                answer = pool[int(rng.choice(n_distractors, p=weights))]

            # Stated confidence: inflated, and only weakly informative.
            base = skill if correct else max(0.0, skill - 0.15)
            stated = float(np.clip(base ** overconfidence
                                   + rng.normal(0, 0.08), 0.01, 0.99))

            members.append(SyntheticRun(
                qid=qid, member=m, exact_answer=answer,
                stated_confidence=stated,
                input_tokens=int(rng.normal(9000, 1800)),
                output_tokens=int(rng.normal(700, 150)),
                search_calls=int(rng.integers(3, 12)),
                billable_searches=int(rng.integers(1, 6)),
            ))

        runs[qid] = members
        truth[qid] = "TRUE-ANSWER"
        split[qid] = "calib" if q < int(0.4 * n_questions) else "test"

    return runs, truth, split


def make_grader(truth: dict[str, str]):
    """A perfect grader for synthetic data.

    Real grading is noisy and that noise caps every calibration claim you make.
    Here it is exact by construction, which is the point: any miscalibration you
    see in demo mode comes from the aggregation, not from the grader.
    """
    def lookup(qid: str, answer: str) -> bool:
        return answer == truth.get(qid)
    return lookup
