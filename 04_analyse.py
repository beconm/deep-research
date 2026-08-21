"""Fit on calib, report on test, sweep the ensemble size, write the figures.

This is the whole result. Everything before it was data collection.
"""
import json
from collections import defaultdict
from types import SimpleNamespace

import matplotlib.pyplot as plt
import numpy as np

from benchmark import read_questions
from calibration import (bootstrap_ci, brier, brier_decomposition, ece,
                             fit_calibrator, paired_permutation_test)
from config import FIGS, PRICING, SEARCH_COST_USD, settings
from ensemble import aggregate
from grader import Grader
from plots import confidence_histogram, cost_frontier, reliability_diagram
from sweep import knee, sweep


def load_runs():
    by_q = defaultdict(list)
    with open(settings.paths["runs"]) as f:
        for line in f:
            if line.strip():
                by_q[json.loads(line)["qid"]].append(SimpleNamespace(**json.loads(line)))
    return dict(by_q)


def main():
    questions = {q.qid: q for q in read_questions()}
    runs_by_q = load_runs()
    grader = Grader()

    # --- assemble per-question signals at full N -------------------------
    rows = []
    for qid, runs in runs_by_q.items():
        q = questions.get(qid)
        if q is None:
            continue
        agg = aggregate(runs, qid, tie_break="random")
        rows.append({
            "qid": qid, "split": q.split,
            "agreement": agg.agreement,
            "stated": agg.stated_confidence if agg.stated_confidence is not None else 0.5,
            "correct": float(grader.lookup(qid, agg.answer)),
        })

    calib = [r for r in rows if r["split"] == "calib"]
    test = [r for r in rows if r["split"] == "test"]
    print(f"calib {len(calib)}   test {len(test)}   "
          f"base accuracy (test) {np.mean([r['correct'] for r in test]):.3f}")

    arr = lambda rs, k: np.array([r[k] for r in rs], float)  # noqa: E731

    # --- fit the mapping on calib, apply to test -------------------------
    iso = fit_calibrator(arr(calib, "agreement"), arr(calib, "correct"), "isotonic")
    platt = fit_calibrator(arr(calib, "agreement"), arr(calib, "correct"), "platt")

    y = arr(test, "correct")
    signals = {
        "verbalised confidence": arr(test, "stated"),
        "raw agreement": arr(test, "agreement"),
        "agreement + isotonic": iso.predict(arr(test, "agreement")),
        "agreement + Platt": platt.predict(arr(test, "agreement")),
    }

    # --- metrics with intervals ------------------------------------------
    print(f"\n{'signal':<24} {'ECE':>18} {'Brier':>8} {'resolution':>11}")
    results = {}
    for name, p in signals.items():
        strat = "quantile" if name.startswith("verbalised") else "levels"
        e, lo, hi = bootstrap_ci(p, y, ece, settings.n_bootstrap,
                                 n_bins=settings.n_bins, strategy=strat)
        d = brier_decomposition(p, y, settings.n_bins)
        results[name] = {"ece": e, "ece_lo": lo, "ece_hi": hi,
                         "brier": brier(p, y), "brier_reconstructed": d["brier"],
                         "reliability": d["reliability"],
                         "resolution": d["resolution"],
                         "uncertainty": d["uncertainty"]}
        print(f"{name:<24} {e:.3f} [{lo:.3f},{hi:.3f}] "
              f"{brier(p, y):>8.3f} {d['resolution']:>11.4f}")

    p_val = paired_permutation_test(
        signals["raw agreement"], signals["verbalised confidence"], y,
        ece, n_perm=10000, n_bins=settings.n_bins, strategy="levels")
    print(f"\nagreement vs verbalised, paired permutation on ECE: p = {p_val:.4f}")

    # --- the sweep --------------------------------------------------------
    price_in, price_out = PRICING.get(settings.researcher_model, (0.0, 0.0))
    pts = sweep(runs_by_q, grader.lookup, settings.sweep_sizes,
                settings.n_subsamples, price_in, price_out, SEARCH_COST_USD)
    k = knee(pts, z=1.0)
    print(f"\n{'N':>3} {'accuracy':>10} {'+/-':>7} {'agreement':>10} {'USD/q':>9}")
    for p in pts:
        print(f"{p.n:>3} {p.accuracy:>10.3f} {p.accuracy_se:>7.3f} "
              f"{p.mean_agreement:>10.3f} {p.usd:>9.4f}")
    print(f"\nknee (within 1 point of best): N = {k}")

    # --- figures ----------------------------------------------------------
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.8),
                           gridspec_kw={"width_ratios": [1, 1.1]})
    reliability_diagram(signals, y, settings.n_bins, "levels", ax=ax[0])
    cost_frontier(pts, knee_n=k, ax=ax[1])
    plt.tight_layout()
    fig.savefig(FIGS / "headline.png", dpi=180)

    fig2, ax2 = plt.subplots(figsize=(6, 2.6))
    confidence_histogram(signals, ax=ax2)
    plt.tight_layout()
    fig2.savefig(FIGS / "confidence_histogram.png", dpi=180)

    with open(settings.paths["results"], "w") as f:
        json.dump({"calibration": results, "permutation_p": p_val,
                   "knee": k, "sweep": [vars(p) for p in pts]}, f, indent=2)
    print(f"\nfigures -> {FIGS}    table -> {settings.paths['results']}")


if __name__ == "__main__":
    main()
