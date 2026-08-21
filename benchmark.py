"""Load BrowseComp and split it into a calibration set and a held-out test set.

BrowseComp ships encrypted to keep it out of training corpora. Each row carries
its own plaintext `canary`, which is the password for that row's `problem` and
`answer` fields. Scheme (from openai/simple-evals, MIT): SHA-256 the password,
tile the digest to the ciphertext length, XOR, decode UTF-8.

Keep the decrypted CSV out of version control -- see .gitignore.
"""
from __future__ import annotations

import base64
import hashlib
import json
import random
import urllib.request
from dataclasses import asdict, dataclass
from pathlib import Path

import pandas as pd

from config import (BROWSECOMP_URL, SIMPLEQA_URL, CACHE, ROOT,
                    ensure_dirs, settings)


def derive_key(password: str, length: int) -> bytes:
    key = hashlib.sha256(password.encode()).digest()
    return (key * (length // len(key) + 1))[:length]


def decrypt(ciphertext_b64: str, password: str) -> str:
    encrypted = base64.b64decode(ciphertext_b64)
    key = derive_key(password, len(encrypted))
    return bytes(a ^ b for a, b in zip(encrypted, key)).decode()


@dataclass
class Question:
    qid: str
    problem: str
    answer: str
    split: str  # "calib" | "test"


DATASETS = {
    # name: (url, encrypted?)
    "browsecomp": (BROWSECOMP_URL, True),
    "simpleqa": (SIMPLEQA_URL, False),
}


def _download(name: str) -> Path:
    ensure_dirs()
    url, _ = DATASETS[name]
    dest = CACHE / url.rsplit("/", 1)[-1]
    if not dest.exists():
        print(f"downloading {name} -> {dest}")
        urllib.request.urlretrieve(url, dest)
    return dest


def load_questions(
    n: int | None = None,
    calib_frac: float | None = None,
    seed: int | None = None,
    name: str | None = None,
) -> list[Question]:
    """Return a deterministic, stratification-free random subset, split in two.

    The split is drawn once from `seed` and never re-drawn. If you change the
    seed after looking at test results you have leaked the test set; don't.
    """
    n = n or settings.n_questions
    calib_frac = settings.calib_frac if calib_frac is None else calib_frac
    seed = settings.split_seed if seed is None else seed
    name = (name or settings.benchmark).lower()
    if name not in DATASETS:
        raise ValueError(f"unknown benchmark {name!r}; choose from {list(DATASETS)}")
    _, encrypted = DATASETS[name]

    df = pd.read_csv(_download(name))
    need = {"problem", "answer"} | ({"canary"} if encrypted else set())
    missing = need - set(df.columns)
    if missing:
        raise ValueError(f"unexpected {name} schema, missing {missing}")

    prefix = {"browsecomp": "bc", "simpleqa": "sq"}[name]
    rows = []
    for i, row in df.iterrows():
        if encrypted:
            problem = decrypt(row["problem"], row["canary"])
            answer = decrypt(row["answer"], row["canary"])
        else:
            problem, answer = str(row["problem"]), str(row["answer"])
        rows.append({"qid": f"{prefix}-{i:05d}", "problem": problem,
                     "answer": answer})

    rng = random.Random(seed)
    rng.shuffle(rows)
    rows = rows[:n]

    n_calib = int(round(calib_frac * len(rows)))
    out = [
        Question(**r, split="calib" if k < n_calib else "test")
        for k, r in enumerate(rows)
    ]
    return out


def save_questions(questions: list[Question], path: Path | None = None) -> Path:
    ensure_dirs()
    path = path or settings.paths["questions"]
    gitignore = ROOT / ".gitignore"
    if (ROOT / ".git").is_dir():
        covered = gitignore.exists() and any(
            line.strip() in {"data/", "data", "data/*"}
            for line in gitignore.read_text().splitlines()
        )
        if not covered and settings.benchmark == "browsecomp":
            print(
                "!! WARNING: this is a git repo and .gitignore does not exclude "
                "data/.\n"
                f"   {path.name} contains DECRYPTED BrowseComp problems and "
                "answers.\n"
                "   Add 'data/' to .gitignore before committing -- the benchmark "
                "is\n"
                "   distributed encrypted to keep it out of training corpora."
            )
    with open(path, "w") as f:
        for q in questions:
            f.write(json.dumps(asdict(q)) + "\n")
    return path


def read_questions(path: Path | None = None) -> list[Question]:
    path = path or settings.paths["questions"]
    with open(path) as f:
        return [Question(**json.loads(line)) for line in f if line.strip()]
