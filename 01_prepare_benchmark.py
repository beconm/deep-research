"""Download, decrypt and split BrowseComp. Run once; the split is then frozen."""
import argparse
from collections import Counter

from benchmark import load_questions, save_questions
from config import settings


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=settings.n_questions)
    ap.add_argument("--calib-frac", type=float, default=settings.calib_frac)
    ap.add_argument("--seed", type=int, default=settings.split_seed)
    args = ap.parse_args()

    qs = load_questions(args.n, args.calib_frac, args.seed)
    path = save_questions(qs)

    counts = Counter(q.split for q in qs)
    print(f"wrote {len(qs)} questions -> {path}")
    print(f"  calib {counts['calib']}   test {counts['test']}   seed {args.seed}")
    print(f"\nexample ({qs[0].qid}):\n  {qs[0].problem[:180]}...")
    print(f"  answer: {qs[0].answer}")


if __name__ == "__main__":
    main()
