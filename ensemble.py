"""Turn N independent answers into one answer plus a confidence estimate.

The confidence signal under test is the *agreement fraction*: the share of
ensemble members whose answer falls in the largest cluster. On BrowseComp,
answers are short factual strings, so clustering reduces to string matching --
but "1997", "in 1997" and "1997." must land together or the signal is noise.
Normalisation handles the easy cases; an optional LLM pass handles the rest.
"""
from __future__ import annotations

import hashlib
import random
import re
import string
import unicodedata
from collections import Counter
from dataclasses import dataclass

_ARTICLES = {"a", "an", "the"}
_PUNCT = str.maketrans("", "", string.punctuation)


def normalise(ans: str) -> str:
    """Aggressive but order-preserving normalisation for short factual answers."""
    s = unicodedata.normalize("NFKD", ans).encode("ascii", "ignore").decode()
    s = s.lower().strip()
    s = re.sub(r"^(the answer is|answer:)\s*", "", s)
    s = s.translate(_PUNCT)
    s = " ".join(w for w in s.split() if w not in _ARTICLES)
    return s


@dataclass
class Aggregate:
    qid: str
    n: int
    answer: str              # plurality answer, in its original surface form
    agreement: float         # |largest cluster| / n  -- the raw confidence signal
    stated_confidence: float | None   # mean of members' self-reported confidence
    n_valid: int
    cluster_sizes: list[int]
    input_tokens: int
    output_tokens: int
    billable_searches: int
    is_tie: bool = False     # top cluster was not unique -- see tie_break below


def cluster(answers: list[str]) -> dict[str, list[int]]:
    """Group member indices by normalised answer. Blank answers are dropped."""
    groups: dict[str, list[int]] = {}
    for i, a in enumerate(answers):
        key = normalise(a)
        if not key:
            continue
        groups.setdefault(key, []).append(i)
    return groups


def aggregate(
    runs: list, qid: str | None = None, tie_break: str = "random"
) -> Aggregate:
    """Plurality vote with an explicit tie-break policy.

    Tie-breaking is not cosmetic, and the default matters. Measured on the
    synthetic fixture at N_max=8, the tie rate is 0.54 at n=2, 0.20 at n=4 and
    0.16 at n=8 -- so at even n a large share of "ensemble" answers are decided
    by the tie-break rule, not by agreement.

    That makes `tie_break="confidence"` actively dangerous for this experiment:
    it decides those cases using the members' verbalised confidence, which is
    the very signal the ensemble is being compared against. It also inflated
    even-n accuracy in the sweep, producing a spurious even/odd oscillation.

      - "random"     (default) seeded from the qid and the tied answers, so it
                     is reproducible without importing information from another
                     signal. Use this for the headline numbers.
      - "confidence" break by the tied members' mean stated confidence. Only
                     for a deliberate ablation, never the main result.
      - "abstain"    return an empty answer, scored as incorrect. The most
                     conservative reading: no plurality means no answer.
    """
    answers = [r.exact_answer for r in runs]
    groups = cluster(answers)
    n = len(runs)
    n_valid = sum(len(v) for v in groups.values())

    if not groups:
        return Aggregate(
            qid or runs[0].qid, n, "", 0.0, None, 0, [],
            sum(r.input_tokens for r in runs), sum(r.output_tokens for r in runs),
            sum(r.billable_searches for r in runs),
        )

    top = max(len(v) for v in groups.values())
    tied = sorted(k for k, v in groups.items() if len(v) == top)
    is_tie = len(tied) > 1

    if not is_tie:
        best_key = tied[0]
    elif tie_break == "abstain":
        stated_ = [r.stated_confidence for r in runs
                   if r.stated_confidence is not None]
        return Aggregate(
            qid or runs[0].qid, n, "", top / n,
            sum(stated_) / len(stated_) if stated_ else None,
            n_valid, sorted((len(v) for v in groups.values()), reverse=True),
            sum(r.input_tokens for r in runs), sum(r.output_tokens for r in runs),
            sum(r.billable_searches for r in runs), is_tie=True,
        )
    elif tie_break == "confidence":
        def conf_of(k: str) -> float:
            cs = [runs[i].stated_confidence for i in groups[k]
                  if runs[i].stated_confidence is not None]
            return sum(cs) / len(cs) if cs else 0.0
        best_key = max(tied, key=conf_of)
    else:  # "random", seeded from content so repeated runs agree
        seed = int(hashlib.sha256(
            ((qid or runs[0].qid) + "|" + "|".join(tied)).encode()
        ).hexdigest()[:8], 16)
        best_key = tied[random.Random(seed).randrange(len(tied))]

    winners = groups[best_key]

    # Denominator is n, not n_valid: a member that failed to produce an answer is
    # evidence of difficulty, and hiding it would inflate confidence exactly on
    # the hard questions where calibration matters most.
    agreement = len(winners) / n

    stated = [r.stated_confidence for r in runs if r.stated_confidence is not None]
    return Aggregate(
        qid=qid or runs[0].qid,
        n=n,
        answer=answers[winners[0]],
        agreement=agreement,
        stated_confidence=sum(stated) / len(stated) if stated else None,
        n_valid=n_valid,
        cluster_sizes=sorted((len(v) for v in groups.values()), reverse=True),
        input_tokens=sum(r.input_tokens for r in runs),
        output_tokens=sum(r.output_tokens for r in runs),
        billable_searches=sum(r.billable_searches for r in runs),
        is_tie=is_tie,
    )


def entropy_confidence(agg: Aggregate) -> float:
    """Alternative signal: 1 - normalised Shannon entropy of the cluster sizes.

    Distinguishes 4/1/1/1/1 from 4/4 (both agreement 0.5 at n=8) -- the first is
    a clear plurality against scattered noise, the second a genuine standoff.
    Worth reporting alongside plain agreement; sometimes it calibrates better.
    """
    import math

    sizes = agg.cluster_sizes
    if not sizes or len(sizes) == 1:
        return 1.0
    total = sum(sizes)
    probs = [s / total for s in sizes]
    h = -sum(p * math.log(p) for p in probs)
    return 1.0 - h / math.log(len(sizes))
