"""Grade every distinct answer produced by any ensemble member.

Grading distinct (qid, answer) pairs rather than draws is what keeps the sweep
affordable: 300 questions x 8 members yields at most 2400 grader calls, and in
practice far fewer because members agree.
"""
import asyncio
import json
from collections import defaultdict

from benchmark import read_questions
from config import settings
from grader import Grader


async def main():
    questions = {q.qid: q for q in read_questions()}
    runs = defaultdict(list)
    with open(settings.paths["runs"]) as f:
        for line in f:
            if line.strip():
                r = json.loads(line)
                runs[r["qid"]].append(r)

    items, seen = [], set()
    for qid, rs in runs.items():
        q = questions.get(qid)
        if q is None:
            continue
        for r in rs:
            key = (qid, r["exact_answer"].strip().lower())
            if key in seen:
                continue
            seen.add(key)
            items.append((qid, q.problem, q.answer, r["exact_answer"]))

    grader = Grader()
    todo = [it for it in items if grader._key(it[0], it[3]) not in grader.cache]
    print(f"{len(items)} distinct answers, {len(todo)} need grading")
    await grader.grade_all(todo)
    print(f"grader calls: {grader.calls}   cache size: {len(grader.cache)}")


if __name__ == "__main__":
    asyncio.run(main())
