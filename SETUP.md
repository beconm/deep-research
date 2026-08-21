# Setup

Requires Python 3.10+. All files sit in one directory; there is no packaging step.

## Free by default

The defaults cost nothing: **Gemini Flash-Lite** (free tier) as the researcher,
**DuckDuckGo** (keyless) as search, **SimpleQA** as the benchmark. One key, no
credit card.

```bash
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
bash setup.sh
```

`setup.sh` installs dependencies, registers a Jupyter kernel called `drc`,
resolves the LangChain agent factory to prove the import works, writes `.env` and
`.gitignore` if missing, and runs the tests.

Then put a key in `.env` — from [aistudio.google.com/apikey](https://aistudio.google.com/apikey):

```
GOOGLE_API_KEY=your-key-here
```

Alternatives: `GROQ_API_KEY` from [console.groq.com/keys](https://console.groq.com/keys)
(faster, lower daily cap), or Ollama for a fully local run with no quota at all.
`OPENAI_API_KEY` and `TAVILY_API_KEY` are only needed if you switch to the paid
configuration in `config.py`.

## Manual install

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m ipykernel install --user --name drc --display-name "drc"
```

Use `python -m pip`, not bare `pip` — on shared machines bare `pip` sometimes
resolves outside the active venv. `requirements.txt` is a package list, not a
script; `python requirements.txt` fails with `NameError: name 'numpy' is not
defined`.

## Check before running

```bash
python check.py
```

Walks the environment in the order things break — interpreter, packages, agent
factory, project modules, `.env` keys, `.gitignore`, stale run files — then makes
a **live model call and a live search**. Keys being present is not the same as
keys being valid, and that distinction has already cost one full run.

## Run

```bash
jupyter lab pipeline.ipynb         # launch FROM this directory
```

Kernel → Change Kernel → **drc**, then work down the notebook. Launching Jupyter
from a parent directory is the one thing a flat layout is sensitive to, because
`import config` then resolves against the wrong working directory; cell 1 checks
and stops with a clear message.

## Budgeting a free run

On a free tier the constraint is time, not money. Gemini Flash allows roughly
1,500 requests/day at 15/minute; Groq roughly 1,000/day at 30/minute. Each agent
run costs several requests.

| questions | N | agent runs | ≈ requests | ≈ days at 1,500/day |
|---|---|---|---|---|
| 300 | 8 | 2,400 | 14,400 | ~10 |
| 150 | 5 | 750 | 4,500 | ~3 |
| 100 | 5 | 500 | 3,000 | ~2 |
| 60 | 5 | 300 | 1,800 | ~1.2 |

Cell 1 prints this estimate for your settings; the defaults are 150 × 5. The run
is resumable and searches are cached, so spreading it across days costs nothing
but patience. Keep `concurrency` at 2 — the per-minute cap bites first, and the
backoff in `researcher.py` absorbs the rest.

Verify the accuracy your configuration actually reaches before scaling up. Under
~15% or over ~85% and the calibration measurement is poorly conditioned; change
the benchmark or the model rather than collecting more data at the wrong
difficulty.

## If a run produced nothing

An invalid key does not crash the pipeline — it produces records that look
complete while being empty. Symptoms: `$0.00` estimated cost, zero tokens,
accuracy `0.000`, `point-biserial r = nan`.

```bash
python check.py                    # live call; identifies the bad key
# fix .env, restart the kernel
python -c "from researcher import prune_failed_runs as p; p('runs/ensemble_runs.jsonl')"
rm -f runs/grader_cache.json
```

The prune step is not optional. Resume skips any `(question, member)` already
recorded, so a file full of failures makes the re-run a silent no-op. The
original is kept as `.bak`.

## Do not commit the decrypted benchmark

BrowseComp is distributed XOR-encrypted so it stays out of training corpora.
`data/questions.jsonl` holds the plaintext. The generated `.gitignore` excludes
`data/`, and `save_questions()` warns if it detects a git repo without that rule.
Check `git status` before your first push. (SimpleQA ships unencrypted, so this
applies to BrowseComp runs.)

## Shared or quota-limited machines

```bash
export PIP_CACHE_DIR=/tmp/$USER-pipcache      # if home is quota-limited
export DRC_ROOT=/scratch/$USER/deep-research  # put outputs on scratch
```
