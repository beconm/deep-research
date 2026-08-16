"""Calibration statistics. Pure numpy/sklearn -- no LLM, no network, testable.

The structure mirrors temperature scaling: fit a one-dimensional mapping on a
calibration split, then report on a test split that the fit never saw. The only
difference is that the mapping here is isotonic or Platt rather than a single
temperature, and the input is ensemble agreement rather than a logit.

Report the raw signal too. Agreement fraction is already a probability estimate
with zero fitted parameters; if it beats verbalised confidence untouched, that
is the cleaner result and it costs nothing to state.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression


# --------------------------------------------------------------------------
# metrics
# --------------------------------------------------------------------------

def bin_edges(p: np.ndarray, n_bins: int, strategy: str = "quantile") -> np.ndarray:
    """Bin boundaries for a reliability analysis.

    Strategies:
      - "levels"   one bin per distinct value. Use this for AGREEMENT, which at
                   ensemble size N can only take N+1 values (7 observed at
                   N=8). Forcing 10 quantile bins onto 7 levels puts several
                   bins on identical values, which makes the reliability curve
                   double back on itself and the ECE partly an artefact of the
                   binning. Falls back to quantile above ~25 distinct values.
      - "quantile" (default) equal-mass bins, right for continuous signals.
      - "uniform"  equal-width bins; leaves sparse bins that dominate the ECE.
    """
    p = np.asarray(p, float)
    if strategy == "levels":
        vals = np.unique(p)
        if len(vals) > 25:
            strategy = "quantile"
        else:
            mids = (vals[:-1] + vals[1:]) / 2
            return np.concatenate(([0.0], mids, [1.0]))
    if strategy == "uniform":
        return np.linspace(0.0, 1.0, n_bins + 1)
    qs = np.linspace(0.0, 1.0, n_bins + 1)
    edges = np.unique(np.quantile(p, qs))
    edges[0], edges[-1] = 0.0, 1.0
    return edges


def reliability(
    p: np.ndarray, y: np.ndarray, n_bins: int = 10, strategy: str = "quantile"
) -> dict:
    """Per-bin mean confidence, observed accuracy and count."""
    p, y = np.asarray(p, float), np.asarray(y, float)
    edges = bin_edges(p, n_bins, strategy)
    idx = np.clip(np.digitize(p, edges[1:-1], right=False), 0, len(edges) - 2)

    conf, acc, cnt = [], [], []
    for b in range(len(edges) - 1):
        m = idx == b
        cnt.append(int(m.sum()))
        conf.append(float(p[m].mean()) if m.any() else np.nan)
        acc.append(float(y[m].mean()) if m.any() else np.nan)
    return {
        "edges": edges, "conf": np.array(conf),
        "acc": np.array(acc), "count": np.array(cnt),
    }


def ece(p: np.ndarray, y: np.ndarray, n_bins: int = 10,
        strategy: str = "quantile") -> float:
    """Expected calibration error: count-weighted mean |accuracy - confidence|."""
    r = reliability(p, y, n_bins, strategy)
    m = r["count"] > 0
    w = r["count"][m] / r["count"][m].sum()
    return float(np.sum(w * np.abs(r["acc"][m] - r["conf"][m])))


def mce(p: np.ndarray, y: np.ndarray, n_bins: int = 10,
        strategy: str = "quantile") -> float:
    """Maximum calibration error -- the worst bin, not the average."""
    r = reliability(p, y, n_bins, strategy)
    m = r["count"] > 0
    return float(np.max(np.abs(r["acc"][m] - r["conf"][m])))


def brier(p: np.ndarray, y: np.ndarray) -> float:
    return float(np.mean((np.asarray(p, float) - np.asarray(y, float)) ** 2))


def brier_decomposition(p: np.ndarray, y: np.ndarray, n_bins: int = 10) -> dict:
    """Murphy decomposition: Brier = reliability - resolution + uncertainty.

    Worth reporting because it separates 'the numbers are miscalibrated'
    (reliability) from 'the signal does not discriminate at all' (resolution).
    A signal can be perfectly calibrated and useless: always saying 0.5 when the
    base rate is 0.5 scores reliability 0 and resolution 0.
    """
    p, y = np.asarray(p, float), np.asarray(y, float)
    r = reliability(p, y, n_bins)
    m = r["count"] > 0
    n = r["count"][m].sum()
    ybar = y.mean()
    rel = float(np.sum(r["count"][m] * (r["conf"][m] - r["acc"][m]) ** 2) / n)
    res = float(np.sum(r["count"][m] * (r["acc"][m] - ybar) ** 2) / n)
    unc = float(ybar * (1 - ybar))
    return {"reliability": rel, "resolution": res, "uncertainty": unc,
            "brier": rel - res + unc}


# --------------------------------------------------------------------------
# fitted mappings
# --------------------------------------------------------------------------

@dataclass
class Calibrator:
    kind: str
    model: object

    def predict(self, p: np.ndarray) -> np.ndarray:
        p = np.asarray(p, float).reshape(-1)
        if self.kind == "identity":
            return p
        if self.kind == "isotonic":
            return np.clip(self.model.predict(p), 0.0, 1.0)
        if self.kind == "platt":
            return self.model.predict_proba(_logit(p).reshape(-1, 1))[:, 1]
        raise ValueError(self.kind)


def _logit(p: np.ndarray, eps: float = 1e-3) -> np.ndarray:
    p = np.clip(np.asarray(p, float), eps, 1 - eps)
    return np.log(p / (1 - p))


def fit_calibrator(p: np.ndarray, y: np.ndarray, kind: str = "isotonic") -> Calibrator:
    """Fit on the CALIBRATION split only. Never on test."""
    p, y = np.asarray(p, float), np.asarray(y, float)
    if kind == "identity":
        return Calibrator("identity", None)
    if kind == "isotonic":
        m = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
        m.fit(p, y)
        return Calibrator("isotonic", m)
    if kind == "platt":
        m = LogisticRegression(C=1e6, solver="lbfgs")
        m.fit(_logit(p).reshape(-1, 1), y)
        return Calibrator("platt", m)
    raise ValueError(kind)


# --------------------------------------------------------------------------
# uncertainty on the metrics themselves
# --------------------------------------------------------------------------

def bootstrap_ci(
    p: np.ndarray, y: np.ndarray, stat=ece, n_boot: int = 2000,
    alpha: float = 0.05, seed: int = 0, **kw,
) -> tuple[float, float, float]:
    """Percentile bootstrap CI. With ~180 test questions, ECE differences of
    0.02 are frequently not significant -- publish the interval, not the point."""
    rng = np.random.default_rng(seed)
    p, y = np.asarray(p, float), np.asarray(y, float)
    n = len(p)
    vals = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, n, n)
        vals[b] = stat(p[idx], y[idx], **kw)
    lo, hi = np.quantile(vals, [alpha / 2, 1 - alpha / 2])
    return float(stat(p, y, **kw)), float(lo), float(hi)


def paired_permutation_test(
    p_a: np.ndarray, p_b: np.ndarray, y: np.ndarray, stat=ece,
    n_perm: int = 10000, seed: int = 0, **kw,
) -> float:
    """Two-sided p-value for 'signal A calibrates better than signal B'.

    Paired: both signals score the same questions, so swap them per question
    under the null rather than shuffling labels.
    """
    rng = np.random.default_rng(seed)
    p_a, p_b, y = (np.asarray(v, float) for v in (p_a, p_b, y))
    observed = abs(stat(p_a, y, **kw) - stat(p_b, y, **kw))
    n = len(y)
    count = 0
    for _ in range(n_perm):
        swap = rng.random(n) < 0.5
        a = np.where(swap, p_b, p_a)
        b = np.where(swap, p_a, p_b)
        if abs(stat(a, y, **kw) - stat(b, y, **kw)) >= observed:
            count += 1
    return (count + 1) / (n_perm + 1)
