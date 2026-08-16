"""Smoke-test the numerical modules on synthetic data (no network, no API)."""
import numpy as np
from types import SimpleNamespace
from calibration import (ece, mce, brier, brier_decomposition, reliability,
                             fit_calibrator, bootstrap_ci, paired_permutation_test)
from ensemble import aggregate, normalise, cluster, entropy_confidence
from sweep import sweep, knee, subsample_predictions

rng = np.random.default_rng(0)

# ---- 1. calibration math on a known-overconfident signal ----------------
n = 400
true_p = rng.uniform(0.05, 0.95, n)
y = (rng.random(n) < true_p).astype(float)
overconf = np.clip(true_p ** 0.45, 0, 1)        # systematically too high

print("well-calibrated  ECE:", round(ece(true_p, y), 4))
print("overconfident    ECE:", round(ece(overconf, y), 4))
assert ece(overconf, y) > ece(true_p, y), "overconfident signal must score worse"

# fit on first half, evaluate on second
cal = fit_calibrator(overconf[:200], y[:200], "isotonic")
fixed = cal.predict(overconf[200:])
print("after isotonic   ECE:", round(ece(fixed, y[200:]), 4),
      " (was", round(ece(overconf[200:], y[200:]), 4), ")")

d = brier_decomposition(true_p, y)
assert abs(d["brier"] - (d["reliability"] - d["resolution"] + d["uncertainty"])) < 1e-9
print("Murphy decomposition closes:", {k: round(v, 4) for k, v in d.items()})

e, lo, hi = bootstrap_ci(overconf, y, ece, n_boot=300)
assert lo <= e <= hi
print(f"bootstrap CI: {e:.4f} [{lo:.4f}, {hi:.4f}]")
p = paired_permutation_test(true_p, overconf, y, ece, n_perm=500)
print(f"permutation p (calibrated vs overconfident): {p:.4f}")

# ---- 2. answer clustering ----------------------------------------------
assert normalise("The 1997.") == normalise("1997")
assert normalise("  Marie  Curie ") == "marie curie"
g = cluster(["1997", "in 1997", "1998", "", "1997."])
print("clusters:", {k: v for k, v in g.items()})
assert len(g["1997"]) == 2 and len(g["in 1997"]) == 1

def mk(qid, member, ans, conf=0.9, correct=None):
    return SimpleNamespace(qid=qid, member=member, exact_answer=ans,
                           stated_confidence=conf, input_tokens=1000,
                           output_tokens=200, billable_searches=3)

runs = [mk("q1", i, "1997" if i < 6 else "1998") for i in range(8)]
agg = aggregate(runs)
print(f"agreement={agg.agreement}  answer={agg.answer}  clusters={agg.cluster_sizes}")
assert agg.agreement == 0.75 and agg.answer == "1997"
assert 0 <= entropy_confidence(agg) <= 1

# blank answers must depress confidence, not be hidden
runs_b = [mk("q2", i, "1997" if i < 4 else "") for i in range(8)]
assert aggregate(runs_b).agreement == 0.5, "blanks must count against agreement"
print("blank handling ok")

# ---- 3. sweep + subsampling --------------------------------------------
runs_by_q = {}
for q in range(30):
    k = rng.integers(2, 9)            # how many members get it right
    runs_by_q[f"q{q}"] = [mk(f"q{q}", i, "GOOD" if i < k else f"BAD{i}")
                          for i in range(8)]
truth = lambda qid, ans: ans == "GOOD"

sub = subsample_predictions(runs_by_q, 3, n_draws=50)
assert all(len(v) == 50 for v in sub.values())
assert len(subsample_predictions(runs_by_q, 8)["q0"]) == 1, "n==N_max has one subset"

pts = sweep(runs_by_q, truth, sizes=(1,2,3,5,8), n_draws=100,
            price_in=0.4, price_out=1.6, search_cost=0.008)
print(f"\n{'N':>3} {'acc':>7} {'se':>6} {'USD/q':>8}")
for p in pts:
    print(f"{p.n:>3} {p.accuracy:>7.3f} {p.accuracy_se:>6.3f} {p.usd:>8.4f}")
assert pts[-1].usd > pts[0].usd, "cost must rise with N"
assert pts[-1].accuracy >= pts[0].accuracy - 0.05, "plurality should not hurt much"
print("knee:", knee(pts))
print("\nALL CHECKS PASSED")
