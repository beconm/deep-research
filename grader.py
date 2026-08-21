"""Grade predicted answers against BrowseComp reference answers.

Grading is the one place where sloppiness silently corrupts everything: a grader
that is itself 90% accurate puts a hard ceiling on any calibration claim you
make. Two mitigations here:

1. A strong grader model, distinct from the (deliberately weak) researcher.
2. Verdicts cached by (qid, normalised answer), so the same distinct string is
   graded exactly once no matter how many subsample draws produce it. Without
   this the sweep would issue tens of thousands of grader calls.

Report grader agreement with your own labels on a sample of ~50. If it is below
~95%, say so in the limitations rather than letting it sit unstated.
"""
from __future__ import annotations

import asyncio
import json
import re
from pathlib import Path

from config import RUNS, ensure_dirs, settings
from ensemble import normalise
from tracing import get_tracer

GRADER_TEMPLATE = """Judge whether the predicted answer matches the correct answer.

Question: {question}
Correct answer: {correct_answer}
Predicted answer: {predicted_answer}

The prediction is correct if it names the same entity, quantity or date as the
correct answer, allowing for differences in phrasing, word order, abbreviation,
or added detail. It is incorrect if it names something different, is empty, or
refuses to commit.

Reply with exactly one word: correct or incorrect.
"""


class Grader:
    def __init__(self, cache_path: Path | None = None):
        self.cache_path = cache_path or (RUNS / "grader_cache.json")
        self.cache: dict[str, bool] = {}
        if self.cache_path.exists():
            self.cache = json.loads(self.cache_path.read_text())
        self._model = None
        self.calls = 0

    @property
    def model(self):
        """Built on first use. The analysis path only calls lookup(), which
        reads the cache -- it must not require an API key or the framework."""
        if self._model is None:
            from langchain.chat_models import init_chat_model
            self._model = init_chat_model(settings.grader_model, temperature=0)
        return self._model

    @staticmethod
    def _key(qid: str, answer: str) -> str:
        return f"{qid}||{normalise(answer)}"

    def lookup(self, qid: str, answer: str) -> bool:
        """Sync lookup for the sweep. Unseen answers count as incorrect --
        they should not exist if you graded before sweeping."""
        return self.cache.get(self._key(qid, answer), False)

    async def grade(self, qid: str, question: str, correct: str, predicted: str) -> bool:
        key = self._key(qid, predicted)
        if key in self.cache:
            return self.cache[key]
        if not normalise(predicted):
            self.cache[key] = False
            return False

        tracer = get_tracer()
        with tracer.span("grader.grade", qid=qid, model=settings.grader_model) as span:
            resp = await self.model.ainvoke(
                GRADER_TEMPLATE.format(question=question, correct_answer=correct,
                                       predicted_answer=predicted)
            )
            text = resp.content if isinstance(resp.content, str) else str(resp.content)
            verdict = bool(re.search(r"\bcorrect\b", text.strip(), re.I)) and not \
                re.search(r"\bincorrect\b", text.strip(), re.I)
            span.update(verdict=verdict)

        self.cache[key] = verdict
        self.calls += 1
        return verdict

    async def grade_all(self, items: list[tuple[str, str, str, str]],
                        concurrency: int = 8) -> None:
        """items: (qid, question, correct_answer, predicted_answer)."""
        sem = asyncio.Semaphore(concurrency)

        async def one(it):
            async with sem:
                return await self.grade(*it)

        await asyncio.gather(*(one(it) for it in items))
        self.save()

    def save(self) -> None:
        ensure_dirs()
        self.cache_path.write_text(json.dumps(self.cache, indent=0))
