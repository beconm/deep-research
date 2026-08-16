"""Run N_MAX independent researchers per question. This is the expensive step.

Resumable: already-completed (qid, member) pairs are skipped, so an interrupted
run costs nothing to restart. Check spend after ~10 questions before committing
to the full set.
"""
import argparse
import asyncio
import json
from dataclasses import asdict

from tqdm import tqdm

from benchmark import read_questions
from cache import SearchCache
from config import settings
from researcher import run_ensemble


def load_done(path):
    done = set()
    if path.exists():
        with open(path) as f:
            for line in f:
                if line.strip():
                    r = json.loads(line)
                    done.add((r["qid"], r["member"]))
    return done


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=settings.n_max)
    ap.add_argument("--limit", type=int, default=None, help="first K questions only")
    ap.add_argument("--split", default=None, choices=[None, "calib", "test"])
    args = ap.parse_args()

    questions = read_questions()
    if args.split:
        questions = [q for q in questions if q.split == args.split]
    if args.limit:
        questions = questions[: args.limit]

    out_path = settings.paths["runs"]
    done = load_done(out_path)
    cache = SearchCache()

    with open(out_path, "a") as f:
        for q in tqdm(questions, desc="questions"):
            if all((q.qid, m) in done for m in range(args.n)):
                continue
            runs = await run_ensemble(q.qid, q.problem, args.n, cache)
            for r in runs:
                if (r.qid, r.member) in done:
                    continue
                d = asdict(r)
                d.pop("raw", None)          # keep the log small
                f.write(json.dumps(d) + "\n")
            f.flush()

    print(f"\nsearch cache: {cache.stats()}")
    print(f"billable searches this session: {cache.billable_searches}")


if __name__ == "__main__":
    asyncio.run(main())
