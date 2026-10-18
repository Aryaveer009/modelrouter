"""SQLite request log: one row per routed request, with cost and latency."""

from __future__ import annotations

import sqlite3
import threading
import time
from pathlib import Path
from typing import Any

from .router import RouteResult

SCHEMA = """
CREATE TABLE IF NOT EXISTS requests (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts REAL NOT NULL,
    query TEXT,
    p_hard REAL NOT NULL,
    threshold REAL NOT NULL,
    initial_tier TEXT NOT NULL,
    final_tier TEXT NOT NULL,
    final_model TEXT NOT NULL,
    escalated INTEGER NOT NULL,
    escalation_reason TEXT,
    forced INTEGER NOT NULL DEFAULT 0,
    prompt_tokens INTEGER NOT NULL,
    completion_tokens INTEGER NOT NULL,
    cost_usd REAL NOT NULL,
    baseline_strong_cost_usd REAL NOT NULL,
    latency_ms REAL NOT NULL,
    mock INTEGER NOT NULL DEFAULT 0
);
CREATE INDEX IF NOT EXISTS idx_requests_ts ON requests(ts);
"""


class RequestLog:
    def __init__(self, path: str | Path, *, log_queries: bool = True) -> None:
        self.path = str(path)
        self.log_queries = log_queries
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(self.path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        with self._lock:
            self._conn.execute("PRAGMA journal_mode=WAL")
            self._conn.executescript(SCHEMA)
            self._conn.commit()

    def record(self, result: RouteResult, *, mock: bool = False) -> int:
        usage = result.usage
        query = result.query[:300] if self.log_queries else None
        with self._lock:
            cur = self._conn.execute(
                "INSERT INTO requests (ts, query, p_hard, threshold, initial_tier, final_tier,"
                " final_model, escalated, escalation_reason, forced, prompt_tokens,"
                " completion_tokens, cost_usd, baseline_strong_cost_usd, latency_ms, mock)"
                " VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (
                    time.time(),
                    query,
                    result.p_hard,
                    result.confidence_threshold,
                    result.initial_tier,
                    result.final_tier,
                    result.final.model,
                    int(result.escalated),
                    result.escalation_reason,
                    int(result.forced),
                    usage.prompt_tokens,
                    usage.completion_tokens,
                    result.cost_usd,
                    result.baseline_strong_cost_usd,
                    result.latency_s * 1000.0,
                    int(mock),
                ),
            )
            self._conn.commit()
            return int(cur.lastrowid or 0)

    def recent(self, limit: int = 50) -> list[dict[str, Any]]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT * FROM requests ORDER BY id DESC LIMIT ?", (limit,)
            ).fetchall()
        return [dict(r) for r in rows]

    def all_rows(self) -> list[dict[str, Any]]:
        with self._lock:
            rows = self._conn.execute("SELECT * FROM requests ORDER BY id").fetchall()
        return [dict(r) for r in rows]

    def summary(self) -> dict[str, Any]:
        with self._lock:
            row = self._conn.execute(
                "SELECT COUNT(*) AS n, COALESCE(SUM(cost_usd),0) AS cost,"
                " COALESCE(SUM(baseline_strong_cost_usd),0) AS baseline,"
                " COALESCE(SUM(final_tier='cheap'),0) AS n_cheap,"
                " COALESCE(SUM(escalated),0) AS n_escalated,"
                " COALESCE(AVG(latency_ms),0) AS avg_latency_ms,"
                " COALESCE(SUM(mock),0) AS n_mock FROM requests"
            ).fetchone()
            latencies = [
                r[0] for r in self._conn.execute("SELECT latency_ms FROM requests ORDER BY 1")
            ]
        n = int(row["n"])
        cost, baseline = float(row["cost"]), float(row["baseline"])
        return {
            "requests": n,
            "cost_usd": cost,
            "baseline_strong_cost_usd": baseline,
            "saved_usd": baseline - cost,
            "saved_pct": (baseline - cost) / baseline * 100 if baseline > 0 else 0.0,
            "pct_answered_by_cheap": row["n_cheap"] / n * 100 if n else 0.0,
            "escalation_rate_pct": row["n_escalated"] / n * 100 if n else 0.0,
            "avg_latency_ms": float(row["avg_latency_ms"]),
            "p50_latency_ms": _percentile(latencies, 50),
            "p95_latency_ms": _percentile(latencies, 95),
            "mock_requests": int(row["n_mock"]),
        }

    def close(self) -> None:
        with self._lock:
            self._conn.close()


def _percentile(sorted_values: list[float], pct: float) -> float:
    if not sorted_values:
        return 0.0
    k = (len(sorted_values) - 1) * pct / 100
    lo, hi = int(k), min(int(k) + 1, len(sorted_values) - 1)
    return sorted_values[lo] + (sorted_values[hi] - sorted_values[lo]) * (k - lo)
