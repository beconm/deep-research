"""Estimate performance at ensemble size n by subsampling the N_max runs.

The naive sweep runs every condition separately: for n in (1,2,3,4,5,6,7,8) that
is 36 research runs per question. Instead run N_max=8 once and draw random
n-subsets from those 8. Cost drops from 36 to 8 per question -- a 4.5x saving --
and averaging over many draws gives error bars for free.

The estimate is unbiased for sampling n members without replacement from the
same distribution, which is exactly what a real n-member ensemble is. What it
cannot capture is any interaction between ensemble size and the search cache:
a genuine 8-member run warms the cache more than a genuine 3-member run, so the
*cost* side of the curve must be modelled per-member, not measured from the
N_max=8 totals. That is handled explicitly below.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ensemble import aggregate


@dataclass
class SweepPoint:
    n: int
    accuracy: float
    accuracy_se: float
    mean_agreement: float
    tie_rate: float
    input_tokens: float      # per question, mean
    output_tokens: float
    billable_searches: float
    usd: float


def subsample_predictions(
    runs_by_q: dict[str, list],
    n: int,
    n_draws: int = 200,
    seed: int = 0,
) -> dict[str, list]:
    """For each question, draw `n_draws` random n-subsets and aggregate each.

    Returns {qid: [Aggregate, ...]} with one Aggregate per draw.
    """
    rng = np.random.default_rng(seed)
    out: dict[str, list] = {}
    for qid, runs in runs_by_q.items():
        n_avail = len(runs)
        if n > n_avail:
            continue
        draws = []
        if n == n_avail:
            draws.append(aggregate(runs, qid))          # only one subset exists
        else:
            for _ in range(n_draws):
                idx = rng.choice(n_avail, size=n, replace=False)
                draws.append(aggregate([runs[i] for i in idx], qid))
        out[qid] = draws
    return out


def sweep(
    runs_by_q: dict[str, list],
    correct_fn,
    sizes: tuple[int, ...] = (1, 2, 3, 4, 5, 6, 7, 8),
    n_draws: int = 200,
    price_in: float = 0.0,
    price_out: float = 0.0,
    search_cost: float = 0.0,
    seed: int = 0,
) -> list[SweepPoint]:
    """Accuracy and cost as a function of ensemble size.

    `correct_fn(qid, answer) -> bool` looks up the graded verdict. Grading every
    subsampled plurality answer with an LLM would cost more than the experiment,
    so grade the *distinct* answers once (see grader.cache_verdicts) and look up.

    Cost is modelled as n x the per-member mean rather than read off the N_max
    totals, because a smaller ensemble would have had a colder cache. This
    slightly over-states the cost of large ensembles, which is the conservative
    direction for a claim of the form "N=3 is enough".
    """
    points: list[SweepPoint] = []
    per_member = _per_member_cost(runs_by_q)

    for n in sizes:
        sub = subsample_predictions(runs_by_q, n, n_draws, seed)
        if not sub:
            continue
        per_q_acc, per_q_agree, per_q_tie = [], [], []
        for qid, draws in sub.items():
            per_q_acc.append(np.mean([correct_fn(qid, d.answer) for d in draws]))
            per_q_agree.append(np.mean([d.agreement for d in draws]))
            per_q_tie.append(np.mean([d.is_tie for d in draws]))
        acc = float(np.mean(per_q_acc))
        se = float(np.std(per_q_acc, ddof=1) / np.sqrt(len(per_q_acc)))
        usd = n * (
            per_member["input_tokens"] * price_in / 1e6
            + per_member["output_tokens"] * price_out / 1e6
            + per_member["billable_searches"] * search_cost
        )
        points.append(
            SweepPoint(
                n=n, accuracy=acc, accuracy_se=se,
                mean_agreement=float(np.mean(per_q_agree)),
                tie_rate=float(np.mean(per_q_tie)),
                input_tokens=n * per_member["input_tokens"],
                output_tokens=n * per_member["output_tokens"],
                billable_searches=n * per_member["billable_searches"],
                usd=usd,
            )
        )
    return points


def _per_member_cost(runs_by_q: dict[str, list]) -> dict[str, float]:
    inp = out = sea = k = 0
    for runs in runs_by_q.values():
        for r in runs:
            inp += r.input_tokens
            out += r.output_tokens
            sea += r.billable_searches
            k += 1
    k = max(k, 1)
    return {"input_tokens": inp / k, "output_tokens": out / k,
            "billable_searches": sea / k}


def knee(points: list[SweepPoint], z: float = 1.0) -> int:
    """Smallest n statistically indistinguishable from the best observed.

    A fixed tolerance ("within 1 accuracy point of best") reads noise as
    signal: on the synthetic fixture it returned n=6 from a curve whose points
    were all within about two standard errors of each other. Compare against
    the paired uncertainty instead -- n qualifies when

        best - acc_n  <=  z * sqrt(se_n^2 + se_best^2)

    so the answer degrades gracefully when the sweep simply cannot resolve a
    knee, rather than inventing one.
    """
    best_pt = max(points, key=lambda p: p.accuracy)
    for p in sorted(points, key=lambda p: p.n):
        margin = z * float(np.hypot(p.accuracy_se, best_pt.accuracy_se))
        if best_pt.accuracy - p.accuracy <= margin:
            return p.n
    return best_pt.n
