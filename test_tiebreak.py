import numpy as np
from types import SimpleNamespace
from ensemble import aggregate
from sweep import sweep, knee

rng = np.random.default_rng(0)
def mk(qid, member, ans, conf):
    return SimpleNamespace(qid=qid, member=member, exact_answer=ans,
                           stated_confidence=conf, input_tokens=1000,
                           output_tokens=200, billable_searches=3)

# GOOD members are more confident on average -- the realistic case, and the one
# the old alphabetical tie-break actively worked against.
runs_by_q = {}
for q in range(60):
    k = int(rng.integers(2, 9))
    runs_by_q[f"q{q}"] = [
        mk(f"q{q}", i, "GOOD" if i < k else f"BAD{i}",
           float(rng.uniform(0.6, 0.95)) if i < k else float(rng.uniform(0.3, 0.7)))
        for i in range(8)]
truth = lambda qid, ans: ans == "GOOD"

pts = sweep(runs_by_q, truth, sizes=(1,2,3,4,5,6,7,8), n_draws=200,
            price_in=0.4, price_out=1.6, search_cost=0.008)
print(f"{'N':>3} {'acc':>7} {'se':>6} {'agree':>7} {'USD/q':>8}")
for p in pts:
    print(f"{p.n:>3} {p.accuracy:>7.3f} {p.accuracy_se:>6.3f} "
          f"{p.mean_agreement:>7.3f} {p.usd:>8.4f}")

accs = [p.accuracy for p in pts]
assert accs[1] >= accs[0] - 0.03, f"n=2 must not collapse: {accs[:2]}"
assert all(accs[i] <= accs[i+1] + 0.06 for i in range(len(accs)-1)), "roughly monotone"
print("\nknee:", knee(pts))
print("tie-break fix verified")

# --- regression: the tie-break must not leak the stated-confidence signal ----
from ensemble import aggregate as _agg
from fixtures import make_runs as _mk, make_grader as _mg
from sweep import subsample_predictions as _sub

_runs, _truth, _ = _mk(300, 8, 0)
_g = _mg(_truth)


def _acc(policy, n):
    import numpy as _np
    out = []
    for qid, rs in _runs.items():
        n_avail = len(rs)
        rng = _np.random.default_rng(0)
        for _ in range(40):
            idx = rng.choice(n_avail, size=n, replace=False)
            a = _agg([rs[i] for i in idx], qid, tie_break=policy)
            out.append(_g(qid, a.answer))
    return float(_np.mean(out))


_rand2, _conf2 = _acc("random", 2), _acc("confidence", 2)
print(f"n=2 accuracy -- random {_rand2:.3f} vs confidence {_conf2:.3f}")
assert _conf2 - _rand2 > 0.05, (
    "expected the confidence tie-break to inflate n=2; if this stops holding, "
    "the leak characterisation in the README needs revisiting")
assert _agg(_runs["syn-00000"], "syn-00000", tie_break="random").answer == \
       _agg(_runs["syn-00000"], "syn-00000", tie_break="random").answer, \
       "random tie-break must be reproducible across calls"
print("tie-break leak regression OK")
