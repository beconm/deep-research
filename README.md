# Ensemble Agreement as a Calibrated Confidence Signal

A deep-research agent that reports how likely its own answer is to be true — and
a measurement of whether that number can be trusted.

Built on [`langchain-ai/deep_research_from_scratch`](https://github.com/langchain-ai/deep_research_from_scratch),
with one architectural change and a great deal of measurement around it.

---

## The idea

The BrowseComp paper reports two things and then leaves them sitting next to each
other. First, best-of-N aggregation reliably improves a browsing agent's
accuracy. Second, that agent's stated confidence is *not* well calibrated in
absolute terms, while still carrying a real internal signal — the model often
"knows" when it is right without being able to express that as a probability.

Nobody closed the gap between those two observations. This repository does:

**If the ensemble already agrees more often when it is right, that agreement is a
probability estimate. Fit it, measure it, and price it.**

Three questions, in order:

1. **Is ensemble agreement better calibrated than the model's own verbalised
   confidence?** Not "does it feel better" — expected calibration error on a
   held-out split, with a bootstrap interval and a paired permutation test.
2. **Does fitting a mapping beat the raw signal?** The agreement fraction is
   already a probability with zero fitted parameters. If isotonic regression
   barely improves on it, the honest headline is that nothing needed fitting — a
   stronger claim, not a weaker one.
3. **What does the confidence cost?** Ensemble size is a dial. Sweep it and find
   where accuracy stops paying for itself, expressed as money per accuracy point
   rather than a ratio of N.

**Nothing here is trained.** The model weights sit frozen behind an API; the only
fitted object in the project is a one-dimensional monotone mapping with no more
freedom than a temperature parameter. This is a measurement study, not a
modelling one.

---

## The architectural change

Upstream, the supervisor decomposes a research brief into independent subtopics
and dispatches one researcher per subtopic. Parallelism there is **work
splitting** — each agent answers a different question, so their outputs cannot be
compared to each other.

Here, N researchers answer **the same** question under temperature sampling.
Their answers become comparable, and the share falling in the largest answer
cluster stops being a coincidence and becomes a signal.

```
          upstream                            this repo

      ┌── researcher → subtopic A         ┌── researcher ─┐
brief ├── researcher → subtopic B         ├── researcher ─┤
      └── researcher → subtopic C    Q ───┼── researcher ─┼── agreement → p(correct)
                 ↓                        ├── researcher ─┤
             one report                   └── researcher ─┘

     parallelism = throughput              parallelism = evidence
```

---

## Pipeline

```
benchmark ──► ensemble ──► grade ──► aggregate ──► calibrate ──► sweep
    │             │           │           │             │           │
 SimpleQA /   N agents on  verdict vs  cluster       fit on      subsample
 BrowseComp   the same Q,  reference,  answers,      calib,      n from N,
 frozen       cached       cached per  agreement     report on   accuracy
 calib/test   search       answer      fraction      test        vs cost
```

**Benchmark.** Short, single, verifiable answers, so correctness is a label
rather than a judgement call. The calibration/test split is drawn once and
frozen; redrawing it after seeing test numbers would leak the test set.

**Ensemble.** N independent research runs on the same question. Search is cached
to disk — partly for cost, mostly because the live web drifting between
conditions would confound the ensemble-size sweep with *when* each condition ran.

**Grading.** Distinct `(question, answer)` pairs are graded once and cached. The
sweep evaluates tens of thousands of subsampled predictions; grading each one
would cost more than generating them.

**Aggregation.** Agreement fraction = share of members in the largest answer
cluster. The denominator is N, not the number of parseable answers — a member
that failed is evidence of difficulty, and excluding it would inflate confidence
on exactly the hard questions where calibration matters most.

**Calibration.** Fit isotonic and Platt mappings on the calibration split, report
on the test split. Four signals land on one reliability diagram: verbalised
confidence, raw agreement, and both fitted variants.

**Sweep.** The largest ensemble is run once; smaller ones are estimated by
drawing random n-subsets from those runs. Cost drops from 36 research runs per
question to 8, and averaging over draws yields error bars for free.

---

## What it reports

- **Reliability diagram** — observed accuracy against stated confidence, four
  signals, with a confidence histogram beside it. A curve hugging the diagonal
  means little if most of the mass sits in one bin.
- **ECE with bootstrap intervals** and a **paired permutation test** for
  "agreement calibrates better than verbalised confidence". At a few hundred test
  questions, ECE gaps below ~0.02 are frequently not significant, and the test
  says so rather than letting the point estimate imply otherwise.
- **Murphy decomposition** of the Brier score, separating *miscalibrated* from
  *uninformative*. A signal that always reports the base rate is perfectly
  calibrated and completely useless.
- **Cost/quality frontier** — accuracy against dollars, annotated by ensemble
  size, with a standard-error-aware knee.

---

## Three decisions that carry the experiment

**A deliberately weak researcher model.** If the agent is 95% accurate, every
point lands in the top reliability bin and calibration cannot be measured. The
45–65% band is the best-conditioned regime — and it is also the cheapest, so the
scientific choice and the budget choice coincide. The same logic drives benchmark
selection: BrowseComp is built so that strong agents fail most of the time, which
pushes a weak model to near-zero accuracy and makes the measurement degenerate.
SimpleQA has the same verifiable-answer shape at a difficulty a free model can
actually spread across.

**Run the largest ensemble once, then subsample.** Estimating every ensemble size
separately costs 4.5× running the largest once and drawing subsets from it.

**Confine the framework.** All LangChain surface lives in one module, and the
agent factory is resolved across its known locations at call time — `create_agent`
moved during the 1.x line. Aggregation, calibration and the sweep are plain
numpy, so a framework release cannot invalidate a result.

---

## Four things that would have quietly broken the result

Each produced plausible-looking output while being wrong. They are documented
because the failure modes are the interesting part.

**The tie-break leaked the signal under test.** At even ensemble sizes ties are
common — 56% of decisions at n=2, 15% at n=8. Breaking them by the members' mean
stated confidence therefore decided a large share of "ensemble" answers using the
very signal the ensemble was being compared against. It inflated n=2 accuracy
from 0.507 to 0.636 and produced a spurious even/odd sawtooth across the entire
sweep. Ties now break by seeded coin flip, and `test_tiebreak.py` locks the
finding in as a regression.

**Agreement is discrete and was binned as though continuous.** At ensemble size N
it takes only N+1 values — 7 distinct values at N=8. Forcing 10 quantile bins
onto 7 levels puts several bins on identical values, which makes the reliability
curve double back on itself and turns part of the ECE into a binning artefact.

**The knee was read out of noise.** A fixed "within one accuracy point of best"
rule returned n=6 from a curve whose points all sat within about two standard
errors of one another. Knee detection now compares against the paired standard
error and degrades to a larger n — or none — when the sweep cannot resolve one.

**An invalid API key produced a complete-looking run of nothing.** 2,400 member
runs, zero tokens, `$0.00` estimated cost, accuracy `0.000`, and the pipeline
carried on through grading and aggregation before dying at the Platt fit with
"needs samples of at least 2 classes". The run now halts at the first question
whose members all fail unrecoverably, the spend check refuses to pass on zero
tokens, and failed records can be pruned so a re-run actually re-runs.

---

## Limitations

- **Correlated errors are the blind spot.** When members are wrong *together* —
  converging on the same plausible distractor — agreement is high and accuracy is
  low, and no calibration mapping fixes that. The test fixture models this
  explicitly via distractor-pool size, because a fixture without it would make
  the method look better than it is.
- **Grader accuracy caps every claim.** An LLM grader that is 90% accurate puts a
  hard ceiling on any calibration result derived from its verdicts.
- **The search backend is part of what accuracy measures.** The keyless
  DuckDuckGo backend has weaker snippets and ranking than Tavily. Runs across
  backends are not comparable, and the difference cannot be attributed to the
  model.
- **Caching removes retrieval diversity.** Members issuing identical queries see
  identical evidence, so the agreement signal measures reasoning diversity, not
  retrieval diversity. Intended, but a scope limit.
- **Benchmark contamination is live.** Frontier models have been documented
  recognising BrowseComp mid-evaluation and reconstructing its decryption scheme
  to look answers up. Suspiciously high accuracy is a finding to investigate, not
  a win.
- **One answer, not many claims.** These benchmarks have single short answers, so
  "per-claim confidence" collapses to "per-answer confidence". Extending to
  multi-claim reports needs claim extraction and per-claim labels.

---

## Resilience and tracing

Two additions on top of the pipeline described above, neither of which changes a result -- both are about not silently losing one.

**A second free-tier model as fallback.** `_with_retry` already absorbs ordinary rate-limit backoff. What it cannot fix is the primary model becoming genuinely unavailable mid-sweep -- an exhausted quota, a provider outage, a deprecated model ID. `resilience.try_with_fallback` tries `fallback_researcher_model` (a different provider by default, so one outage doesn't take out both) once before a member's run is recorded as failed. Off by default in effect: point it at the same model as `researcher_model`, or leave its key unset, and `run_one` behaves exactly as before. `ResearchRun.model_used` records which model actually answered.

**Groq as researcher, Gemini as fallback.** Gemini's newest Flash-Lite generation ignores `temperature`, `top_p`, and `top_k` entirely -- a deliberate change on Google's part, confirmed via a live API call rather than documentation alone -- which removes the sampling diversity the ensemble's confidence signal depends on. Groq's open-weight models carry no equivalent restriction, which is why they are the researcher default here rather than the fallback. The trade-off: Groq's free tier is a per-day token budget rather than a per-day request count, and BrowseComp-style questions are token-heavy enough that a full sweep can exhaust a day's budget within a handful of questions -- worth sizing `N_QUESTIONS` and `N_MAX` against that, or against SimpleQA's far lighter per-question cost, before committing to a long run. See `POSTMORTEM.md` for the incident that surfaced both of these.

**Every model and grader call is traced.** `tracing.get_tracer()` writes structured spans -- which model, latency, success, the exact error on failure -- to `traces.jsonl` by default, or to Langfuse if `LANGFUSE_PUBLIC_KEY` / `LANGFUSE_SECRET_KEY` are set. On a free tier with an opaque per-minute or per-day cap, this is the difference between "the sweep is slow" and knowing exactly which provider is throttling and how often the fallback had to cover for it.

**One thing this does not fix.** `04_analyse.py` prices every token at `PRICING[settings.researcher_model]`. If the fallback is exercised at volume, its tokens are still priced at the primary's rate -- invisible while both are free-tier `(0.0, 0.0)`, but wrong the moment either model is paid.

## A live endpoint on top of the batch pipeline

Everything above runs as an offline sweep against a fixed benchmark. `api.py` wraps the same `researcher.run_one()` and `ensemble.aggregate()` used by `02_run_ensemble.py` and `04_analyse.py` behind a single HTTP endpoint, so the calibrated confidence signal this project measures can also answer one live question instead of three hundred benchmark ones.

`POST /research` streams each ensemble member's answer as it completes (Server-Sent Events, not raw token streaming -- an ensemble's natural unit of progress is "one member finished," not "one token generated"), then a final event with the aggregated answer and its agreement score. A request here does not touch `runs/ensemble_runs.jsonl` or the grading path; it shares only the search cache with the batch pipeline, so overlapping queries benefit from cached results without mixing live traffic into benchmark data.

The confidence returned is raw agreement, not the isotonic- or Platt-fitted probability `04_analyse.py` reports -- a completed run's fitted calibrator is not currently persisted to disk, only its metrics. Saving that mapping and loading it here to convert raw agreement into a properly calibrated probability is a natural next step and is not yet built.

`Dockerfile` and `requirements-service.txt` package the service for deployment; `.github/workflows/deploy.yml` builds, tests, and deploys it to Cloud Run on push to `main`, separately from `.github/workflows/ci.yml`'s lighter test-only run on every push. Because `api.py` calls the real research pipeline rather than a stand-in, the deployed image needs the full dependency set from `requirements.txt` -- there is no smaller "just the API" install here, unlike a service that only wraps a single model call.

`POSTMORTEM.md` documents two real failures hit while getting this pipeline running on free-tier keys -- a retired model and an exhausted quota -- each diagnosed with tools this project already has: `check.py`'s live test, a diversity probe, and the trace log. `test_known_errors.py` locks both error shapes in as a regression test, the same pattern `test_tiebreak.py` already uses for the tie-break finding.

## Layout

| File | Role |
|---|---|
| `pipeline.ipynb` | driver notebook — every cell is an import and a call |
| `config.py` | every dial: models, benchmark, search backend, pricing |
| `benchmark.py` | dataset download, decryption, frozen calibration/test split |
| `researcher.py` | one research run; the only LangChain-facing module |
| `resilience.py` | retry-then-fallback for a single model call, framework-free |
| `tracing.py` | per-call tracing -- Langfuse if configured, local JSONL otherwise |
| `ensemble.py` | answer normalisation, clustering, agreement, tie-break policy |
| `grader.py` | verdicts against the reference answer, cached per distinct answer |
| `calibration.py` | ECE, Brier, Murphy decomposition, isotonic/Platt, bootstrap, permutation test |
| `sweep.py` | subsampled ensemble-size sweep, standard-error-aware knee |
| `plots.py` | reliability diagram, confidence histogram, cost frontier |
| `cache.py` | disk-backed search cache |
| `fixtures.py` | test fixtures only — never imported by the pipeline |
| `check.py` | environment check, including live API calls |
| `api.py` | live HTTP endpoint wrapping the same researcher/aggregate functions |
| `Dockerfile`, `requirements-service.txt` | container image for `api.py` |
| `POSTMORTEM.md` | two real free-tier failures hit while getting this running |
| `test_*.py` | numerical tests, no API calls (includes `test_resilience.py`, `test_known_errors.py`) |

Setup and dependencies: [SETUP.md](SETUP.md).

---

## Status

The pipeline is complete and the numerical components are tested. The measurement
run is in progress; the table below is filled from `runs/results.json`.

| Signal | ECE [95% CI] | Brier | Resolution |
|---|---|---|---|
| verbalised confidence | — | — | — |
| raw agreement | — | — | — |
| agreement + isotonic | — | — | — |
| agreement + Platt | — | — | — |

---

Benchmarks from [`openai/simple-evals`](https://github.com/openai/simple-evals) (MIT).
