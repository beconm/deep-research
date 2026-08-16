#!/usr/bin/env bash
# One-shot setup. From this directory, with a virtualenv activated:
#     bash setup.sh
set -euo pipefail

if [[ -z "${VIRTUAL_ENV:-}" && -z "${CONDA_PREFIX:-}" ]]; then
  echo "!! No virtualenv active. Create one first:"
  echo "     python3 -m venv .venv && source .venv/bin/activate"
  exit 1
fi

PY=$(python -c 'import sys; print(".".join(map(str,sys.version_info[:3])))')
OK=$(python -c 'import sys; print(1 if sys.version_info[:2] >= (3,10) else 0)')
[[ "$OK" == "1" ]] || { echo "!! Python $PY too old; need 3.10+"; exit 1; }
echo "-> python $PY"

python -m pip install --upgrade pip -q
echo "-> installing dependencies (this pulls the LangChain stack; a few minutes)"
python -m pip install -q -r requirements.txt

echo "-> registering Jupyter kernel 'drc'"
python -m ipykernel install --user --name drc --display-name "drc" >/dev/null

echo "-> verifying analysis modules"
python -c "import config, calibration, ensemble, sweep, plots; print('   analysis OK')"

echo "-> verifying agent stack"
python - <<'PY'
import importlib
for m in ("langchain", "langgraph", "langchain_openai", "langchain_tavily"):
    mod = importlib.import_module(m)
    print(f"   {m:<20} {getattr(mod, '__version__', 'installed')}")
from researcher import _load_agent_factory
print("   agent factory      ", _load_agent_factory().__name__)
PY

echo "-> running tests"
python test_numerics.py | tail -1
python test_tiebreak.py | tail -1

# --- dotfiles are generated here, not shipped -----------------------------
# Downloading files individually silently drops leading-dot files, and a
# missing .gitignore is not cosmetic: data/questions.jsonl holds DECRYPTED
# BrowseComp problems and answers. Committing that to a public repo leaks a
# benchmark whose whole point is staying out of training corpora.
if [[ ! -f .gitignore ]]; then
  cat > .gitignore <<'GITEOF'
# Secrets
.env

# DECRYPTED BENCHMARK -- never commit. BrowseComp ships encrypted so it stays
# out of training corpora; questions.jsonl is the plaintext.
data/
.cache/

# Run artefacts
runs/
figures/

# Python
__pycache__/
*.egg-info/
.venv/
.ipynb_checkpoints/
GITEOF
  echo "-> wrote .gitignore (protects the decrypted benchmark and your keys)"
fi

if [[ ! -f .env ]]; then
  cat > .env <<'ENVEOF'
# Required -- every pipeline step calls a live API.
OPENAI_API_KEY=
TAVILY_API_KEY=

# Optional: trace UI for debugging the agent loop.
# LANGSMITH_API_KEY=
# LANGSMITH_TRACING=true
# LANGSMITH_PROJECT=deep-research-calibration

# Experiment overrides (defaults live in config.py)
RESEARCHER_MODEL=openai:gpt-4.1-mini
GRADER_MODEL=openai:gpt-4.1
N_MAX=8
N_QUESTIONS=300
ENVEOF
  echo "-> wrote .env skeleton"
  echo
  echo "NEXT: open .env and fill in OPENAI_API_KEY and TAVILY_API_KEY"
else
  KEYS_OK=1
  grep -q "^OPENAI_API_KEY=.\+" .env || KEYS_OK=0
  grep -q "^TAVILY_API_KEY=.\+" .env || KEYS_OK=0
  if [[ "$KEYS_OK" == "1" ]]; then
    echo ".env found with both keys set."
  else
    echo ".env found but a key is still blank -- fill in OPENAI_API_KEY / TAVILY_API_KEY"
  fi
fi
echo "Then: jupyter lab pipeline.ipynb   (kernel: drc)"
