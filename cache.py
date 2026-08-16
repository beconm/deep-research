"""Disk-backed cache for search results.

Two reasons this exists, and the second one matters more than the first:

1. Cost. Ensemble members re-issue overlapping queries; paying once is enough.
2. Reproducibility. The live web drifts. Without a cache, a sweep over ensemble
   size N is confounded with the web changing underneath it, and the cost curve
   becomes partly a measurement of when you ran each condition. The cache pins
   the corpus for the duration of the experiment.

Caveat worth stating in the writeup: caching also removes retrieval-side
diversity between members. Two members that issue an identical query see
identical evidence. That is the intended behaviour -- diversity should come from
the sampling temperature, not from search API nondeterminism -- but it does mean
your agreement signal measures reasoning diversity, not retrieval diversity.
"""
from __future__ import annotations

import hashlib
import json
import sqlite3
import threading
from pathlib import Path
from typing import Any

from config import CACHE, ensure_dirs


class SearchCache:
    def __init__(self, path: Path | None = None):
        ensure_dirs()
        self.path = path or (CACHE / "search.sqlite")
        self._local = threading.local()
        with self._conn() as c:
            c.execute(
                "CREATE TABLE IF NOT EXISTS kv ("
                " k TEXT PRIMARY KEY, v TEXT NOT NULL, hits INTEGER DEFAULT 0)"
            )
        self.misses = 0
        self.hits = 0

    def _conn(self) -> sqlite3.Connection:
        if not hasattr(self._local, "conn"):
            self._local.conn = sqlite3.connect(self.path, check_same_thread=False)
        return self._local.conn

    @staticmethod
    def _key(query: str, **kw: Any) -> str:
        blob = json.dumps({"q": query, **kw}, sort_keys=True)
        return hashlib.sha256(blob.encode()).hexdigest()

    def get(self, query: str, **kw: Any):
        k = self._key(query, **kw)
        cur = self._conn().execute("SELECT v FROM kv WHERE k = ?", (k,))
        row = cur.fetchone()
        if row is None:
            self.misses += 1
            return None
        self.hits += 1
        self._conn().execute("UPDATE kv SET hits = hits + 1 WHERE k = ?", (k,))
        self._conn().commit()
        return json.loads(row[0])

    def put(self, query: str, value: Any, **kw: Any) -> None:
        k = self._key(query, **kw)
        conn = self._conn()
        conn.execute(
            "INSERT OR REPLACE INTO kv (k, v) VALUES (?, ?)", (k, json.dumps(value))
        )
        conn.commit()

    @property
    def billable_searches(self) -> int:
        """Only misses cost money. This is what feeds the cost curve."""
        return self.misses

    def stats(self) -> dict:
        total = self.hits + self.misses
        return {
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate": self.hits / total if total else 0.0,
        }
