"""Reconcile measured cost against Snowflake's own billing record.

``show_details => TRUE`` gives token counts at call time, which is what the
scheduler needs live. It is not the billing record. The authoritative source is
``SNOWFLAKE.ACCOUNT_USAGE.CORTEX_FUNCTIONS_USAGE_HISTORY`` (and its per-query
sibling), which reports the credits actually consumed — but those views lag real
time by up to a few hours.

So the benchmark works in two passes: run now with token-based costs, then come
back later and stamp the real credits onto the saved result. The headline number
in a submitted result should always come from this second pass.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

USAGE_SQL = """
SELECT
    QUERY_ID,
    MODEL_NAME,
    FUNCTION_NAME,
    TOKENS,
    TOKEN_CREDITS,
    START_TIME
FROM SNOWFLAKE.ACCOUNT_USAGE.CORTEX_FUNCTIONS_QUERY_USAGE_HISTORY
WHERE START_TIME >= DATEADD('hour', -%s, CURRENT_TIMESTAMP())
"""

AGGREGATE_SQL = """
SELECT
    MODEL_NAME,
    FUNCTION_NAME,
    SUM(TOKENS)        AS TOKENS,
    SUM(TOKEN_CREDITS) AS CREDITS
FROM SNOWFLAKE.ACCOUNT_USAGE.CORTEX_FUNCTIONS_USAGE_HISTORY
WHERE START_TIME >= DATEADD('hour', -%s, CURRENT_TIMESTAMP())
GROUP BY 1, 2
ORDER BY CREDITS DESC
"""


@dataclass(slots=True)
class UsageRow:
    query_id: str
    model: str
    function: str
    tokens: int
    credits: float


def fetch(conn, lookback_hours: int = 24) -> dict[str, UsageRow]:
    """Pull per-query Cortex usage, keyed by query id."""
    rows: dict[str, UsageRow] = {}
    with conn.cursor() as cur:
        cur.execute(USAGE_SQL, (lookback_hours,))
        for query_id, model, function, tokens, credits, _start in cur.fetchall():
            rows[query_id] = UsageRow(
                query_id=query_id,
                model=model,
                function=function,
                tokens=int(tokens or 0),
                credits=float(credits or 0.0),
            )
    return rows


def reconcile(result_path: str | Path, conn, lookback_hours: int = 24) -> dict:
    """Stamp real credits onto a saved benchmark result, in place.

    Returns a short report of how many calls were matched. Unmatched calls are
    reported rather than silently zeroed — a partially reconciled run must not
    look like a cheap one.
    """
    path = Path(result_path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    usage = fetch(conn, lookback_hours)

    report = {"matched": 0, "unmatched": 0, "credits": {}}
    for policy in ("baseline", "mavis"):
        total = 0.0
        for trace in payload["traces"][policy]:
            for step in trace["steps"]:
                cost = step.get("cost")
                if not cost or not cost.get("query_id"):
                    continue
                row = usage.get(cost["query_id"])
                if row is None:
                    report["unmatched"] += 1
                    continue
                cost["credits"] = row.credits
                cost["estimated"] = False
                total += row.credits
                report["matched"] += 1
        payload[policy]["credits"] = total
        payload[policy]["credits_measured"] = report["unmatched"] == 0
        report["credits"][policy] = total

    b, m = report["credits"]["baseline"], report["credits"]["mavis"]
    payload["cost_reduction"] = (b - m) / b if b else float("nan")
    payload["cost_basis"] = "credits"
    payload["estimated"] = report["unmatched"] > 0
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return report
