"""The two figures the project exists to produce."""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from calibration import ece, reliability


def reliability_diagram(
    signals: dict[str, np.ndarray], y: np.ndarray, n_bins: int = 10,
    strategy: str = "quantile", ax=None,
):
    """Three curves on one axis: verbalised confidence, raw agreement, calibrated.

    The diagonal is perfect calibration. Below it means overconfident. The whole
    argument of the project is visible in which curve sits closest to the line.
    """
    ax = ax or plt.subplots(figsize=(5.2, 5))[1]
    ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.5, zorder=1, label="perfect")

    for name, p in signals.items():
        r = reliability(p, y, n_bins, strategy)
        m = r["count"] > 0
        e = ece(p, y, n_bins, strategy)
        ax.plot(r["conf"][m], r["acc"][m], "o-", ms=5,
                label=f"{name}  (ECE {e:.3f})", zorder=3)

    ax.set_xlabel("stated confidence")
    ax.set_ylabel("observed accuracy")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.set_aspect("equal")
    ax.legend(loc="upper left", fontsize=8, frameon=False)
    ax.set_title("Reliability (held-out test split)", fontsize=10)
    return ax


def confidence_histogram(signals: dict[str, np.ndarray], ax=None):
    """Companion to the reliability diagram: where the mass actually sits.

    A reliability curve hugging the diagonal means little if 90% of the points
    are in one bin. Always publish this next to it.
    """
    ax = ax or plt.subplots(figsize=(5.2, 2.4))[1]
    for name, p in signals.items():
        ax.hist(p, bins=np.linspace(0, 1, 21), histtype="step", lw=1.5, label=name)
    ax.set_xlabel("stated confidence"); ax.set_ylabel("questions")
    ax.legend(fontsize=8, frameon=False)
    return ax


def cost_frontier(points, knee_n: int | None = None, ax=None):
    """Accuracy against dollars, annotated by ensemble size."""
    ax = ax or plt.subplots(figsize=(5.6, 4.2))[1]
    usd = [p.usd for p in points]
    acc = [p.accuracy for p in points]
    se = [p.accuracy_se for p in points]

    ax.errorbar(usd, acc, yerr=se, fmt="o-", capsize=3, ms=5)
    for p in points:
        ax.annotate(f"N={p.n}", (p.usd, p.accuracy),
                    textcoords="offset points", xytext=(6, -10), fontsize=8)
    if knee_n is not None:
        kp = next(p for p in points if p.n == knee_n)
        ax.axvline(kp.usd, color="crimson", ls=":", lw=1.2)
        ax.annotate(f"knee at N={knee_n}", (kp.usd, min(acc)),
                    color="crimson", fontsize=8,
                    textcoords="offset points", xytext=(6, 4))

    ax.set_xlabel("USD per question")
    ax.set_ylabel("accuracy (plurality answer)")
    ax.set_title("Cost / quality frontier", fontsize=10)
    return ax
