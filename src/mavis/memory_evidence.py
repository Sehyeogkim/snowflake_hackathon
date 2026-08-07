from __future__ import annotations

import json
import math
import re
from collections import Counter
from pathlib import Path
from typing import Any

from .policy import MemoryEvidence, RuntimeAction

_CLIP_IN_SESSION = re.compile(r"mavis-seed-([0-9a-f]{16})-[0-9a-f]{8}")
_EXPERIENCE_IN_SESSION = re.compile(r"mavis-memory-v3-([0-9a-f]{24})")
_EXPERIENCE_IN_TEXT = re.compile(
    r"experience(?: with (?:the )?identifier)?\s+([0-9a-f]{24})", re.IGNORECASE
)
_TOKEN = re.compile(r"[a-z0-9_]+")


def _data(response: dict[str, Any]) -> dict[str, Any]:
    value = response.get("data", response)
    return value if isinstance(value, dict) else {}


def case_references(response: dict[str, Any]) -> list[tuple[str, str, float]]:
    """Return (case_id, clip_id, retrieval weight) from EverOS cases."""
    cases = _data(response).get("agent_cases") or []
    raw: list[tuple[str, str, float]] = []
    for case in cases:
        if not isinstance(case, dict):
            continue
        match = _CLIP_IN_SESSION.search(str(case.get("session_id", "")))
        if not match:
            continue
        score = max(0.0, float(case.get("score", 0.0)))
        quality = max(0.0, float(case.get("quality_score", 1.0)))
        raw.append((str(case.get("id", "")), match.group(1), score * quality))
    return [(case_id, clip_id, weight if weight > 0 else 0.05) for case_id, clip_id, weight in raw]


def episode_references(response: dict[str, Any]) -> list[tuple[str, str, float]]:
    """Return (episode_id, experience_id, retrieval weight) from EverOS episodes."""
    episodes = _data(response).get("episodes") or response.get("episodes") or []
    output: list[tuple[str, str, float]] = []
    for episode in episodes:
        if not isinstance(episode, dict):
            continue
        session = str(episode.get("session_id", ""))
        match = _EXPERIENCE_IN_SESSION.search(session)
        if not match:
            searchable = " ".join(
                str(episode.get(field, "")) for field in ("summary", "subject", "episode")
            )
            match = _EXPERIENCE_IN_TEXT.search(searchable)
        if not match:
            continue
        score = max(0.0, float(episode.get("score", 0.0))) or 0.05
        output.append((str(episode.get("id", "")), match.group(1), score))
    return output


def _actual_or_default_cost(observation: dict[str, Any], default: float) -> float:
    actual = observation.get("actual_credits")
    return max(float(actual), 1e-9) if actual is not None else default


def evidence_from_experience_payload(
    payload: dict[str, Any], retrieval_weight: float, source_id: str | None = None
) -> list[MemoryEvidence]:
    """Convert one portable, GT-evaluated Experience into routing evidence."""
    outcome = str(payload.get("outcome", ""))
    best_action = str(payload.get("best_action", ""))
    observations = [row for row in payload.get("observations", []) if isinstance(row, dict)]
    metadata = payload.get("metadata") or {}
    unsafe_false_negative = bool(metadata.get("unsafe_false_negative")) or any(
        bool(row.get("unsafe_false_negative")) for row in observations
    )
    escalation = "escalat" in best_action or "escalation" in outcome

    if outcome == "cheap_strategy_verified" or best_action.startswith("flash_lite"):
        cheap = observations[0] if observations else {}
        return [
            MemoryEvidence(
                action=RuntimeAction.ACCEPT_CHEAP,
                retrieval_weight=retrieval_weight,
                correct=bool(cheap.get("correct", True)),
                information_gain=0.0,
                cost_credits=_actual_or_default_cost(cheap, 0.15),
                source_case_id=source_id,
            )
        ]

    action = (
        RuntimeAction.STRONG_MULTI
        if unsafe_false_negative or "temporal" in best_action or "dense" in best_action
        else RuntimeAction.STRONG_SINGLE
    )
    strong_observation = next(
        (
            row
            for row in observations
            if str(row.get("action", "")).startswith("pro_")
            or row.get("action") in {"strong_single", "strong_multi"}
        ),
        {},
    )
    gain = max(0.0, float(metadata.get("gt_log_loss_increase") or 0.0))
    if gain == 0.0:
        gain = 1.0 if unsafe_false_negative else 0.5
    return [
        MemoryEvidence(
            action=action,
            retrieval_weight=retrieval_weight,
            # This is correctness of the evaluated routing lesson, not a claim
            # that an unobserved Snowflake model prediction was correct.
            correct=bool(payload.get("task_completed", True)),
            information_gain=gain,
            cost_credits=_actual_or_default_cost(
                strong_observation, 2.2 if action == RuntimeAction.STRONG_MULTI else 1.0
            ),
            source_case_id=source_id,
            unsafe_false_negative=unsafe_false_negative,
            recommends_escalation=escalation,
        )
    ]


class PortableExperienceIndex:
    """Small deterministic local index mirroring the 300 EverOS episodes."""

    def __init__(self, rows: list[dict[str, Any]], limit: int | None = None) -> None:
        # Produce nested, class/outcome-balanced 50/100/300 ablation subsets
        # instead of taking a path-sorted prefix with accidental class skew.
        buckets: dict[tuple[str, str], list[dict[str, Any]]] = {}
        for row in rows:
            payload = row.get("payload", row)
            key = (str(payload.get("label", "")), str(payload.get("outcome", "")))
            buckets.setdefault(key, []).append(row)
        for bucket in buckets.values():
            bucket.sort(key=lambda row: str(row.get("experience_id", "")))
        balanced: list[dict[str, Any]] = []
        while buckets:
            empty = []
            for key in sorted(buckets):
                bucket = buckets[key]
                if bucket:
                    balanced.append(bucket.pop(0))
                if not bucket:
                    empty.append(key)
            for key in empty:
                buckets.pop(key)
        selected = balanced if limit is None else balanced[: max(0, limit)]
        self.payloads: dict[str, dict[str, Any]] = {}
        self.documents: dict[str, Counter[str]] = {}
        document_frequency: Counter[str] = Counter()
        for row in selected:
            payload = row.get("payload", row)
            experience_id = str(row.get("experience_id") or payload.get("experience_id") or "")
            if not experience_id or not isinstance(payload, dict):
                continue
            self.payloads[experience_id] = payload
            searchable = " ".join(
                str(payload.get(field, ""))
                for field in ("scene", "text", "key_insight", "label", "outcome", "best_action")
            )
            tokens = Counter(_TOKEN.findall(searchable.casefold()))
            self.documents[experience_id] = tokens
            document_frequency.update(tokens)
        count = max(1, len(self.documents))
        self.idf = {
            token: math.log((count + 1) / (frequency + 1)) + 1.0
            for token, frequency in document_frequency.items()
        }

    @classmethod
    def from_jsonl(cls, path: Path, limit: int | None = None) -> PortableExperienceIndex:
        rows = [
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        return cls(rows, limit=limit)

    def _similarity(self, query: Counter[str], document: Counter[str]) -> float:
        common = set(query) & set(document)
        numerator = sum(
            query[token] * document[token] * self.idf.get(token, 1.0) ** 2 for token in common
        )
        query_norm = math.sqrt(
            sum((count * self.idf.get(token, 1.0)) ** 2 for token, count in query.items())
        )
        document_norm = math.sqrt(
            sum((count * self.idf.get(token, 1.0)) ** 2 for token, count in document.items())
        )
        return numerator / max(query_norm * document_norm, 1e-12)

    def local_references(self, query: str, top_k: int = 8) -> list[tuple[str, str, float]]:
        query_tokens = Counter(_TOKEN.findall(query.casefold()))
        scored = [
            (f"local:{experience_id}", experience_id, self._similarity(query_tokens, document))
            for experience_id, document in self.documents.items()
        ]
        scored.sort(key=lambda row: (-row[2], row[1]))
        return [row for row in scored[:top_k] if row[2] > 0]

    def evidence(
        self, references: list[tuple[str, str, float]]
    ) -> list[MemoryEvidence]:
        output: list[MemoryEvidence] = []
        for source_id, experience_id, weight in references:
            payload = self.payloads.get(experience_id)
            if payload:
                output.extend(evidence_from_experience_payload(payload, weight, source_id))
        return output


def evidence_from_payloads(
    references: list[tuple[str, str, float]],
    payloads_by_clip: dict[str, dict[str, Any]],
) -> list[MemoryEvidence]:
    """Join EverOS qualitative retrieval to local structured, auditable trajectories."""
    output: list[MemoryEvidence] = []
    for case_id, clip_id, retrieval_weight in references:
        payload = payloads_by_clip.get(clip_id)
        if not payload:
            continue
        for observation in payload.get("observations", []):
            action_name = observation.get("action")
            if action_name not in {RuntimeAction.STRONG_SINGLE, RuntimeAction.STRONG_MULTI}:
                continue
            credits = observation.get("actual_credits")
            if credits is None:
                continue
            output.append(
                MemoryEvidence(
                    action=RuntimeAction(action_name),
                    retrieval_weight=retrieval_weight,
                    correct=bool(observation.get("correct", False)),
                    information_gain=float(observation.get("information_gain", 0.0)),
                    cost_credits=float(credits),
                    source_case_id=case_id,
                )
            )
    return output
