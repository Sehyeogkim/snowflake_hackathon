from __future__ import annotations

import json
import math
import threading
from collections import Counter
from collections.abc import Callable, Iterable
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Protocol

from .backends.base import InferenceBackend
from .candidates import extract_candidates
from .config import Settings
from .labels import normalize_label
from .memory_evidence import PortableExperienceIndex, episode_references
from .models import ClipRecord, InferenceResult
from .policy import MavisPolicy, MemoryEvidence, PolicyDecision, RuntimeAction
from .visual_memory import VisualMemoryIndex, fingerprint_query


class EvidenceRetriever(Protocol):
    def retrieve(
        self, observation: InferenceResult, clip: ClipRecord, top_k: int
    ) -> list[MemoryEvidence]: ...


class NoMemoryRetriever:
    def retrieve(
        self, observation: InferenceResult, clip: ClipRecord, top_k: int
    ) -> list[MemoryEvidence]:
        del observation, clip, top_k
        return []


def memory_query(observation: InferenceResult) -> str:
    return (
        f"factory CCTV scene: {observation.scene}; "
        f"cheap_prediction={observation.prediction}; "
        f"need_temporal_context={observation.need_temporal_context}; "
        "find ground-truth-evaluated routing lessons for similar visual evidence"
    )


class LocalMemoryRetriever:
    def __init__(self, index: PortableExperienceIndex) -> None:
        self.index = index

    def retrieve(
        self, observation: InferenceResult, clip: ClipRecord, top_k: int
    ) -> list[MemoryEvidence]:
        del clip
        references = self.index.local_references(memory_query(observation), top_k=top_k)
        return self.index.evidence(references)


class EverOSMemoryRetriever:
    """Use EverOS for similarity, then join to the local auditable payload ledger."""

    def __init__(self, index: PortableExperienceIndex, client) -> None:
        self.index = index
        self.client = client

    def retrieve(
        self, observation: InferenceResult, clip: ClipRecord, top_k: int
    ) -> list[MemoryEvidence]:
        del clip
        response = self.client.search(memory_query(observation), top_k=top_k, scope="user")
        return self.index.evidence(episode_references(response))


class VisualMemoryRetriever:
    def __init__(self, payload_index: PortableExperienceIndex, visual_index: VisualMemoryIndex) -> None:
        self.payload_index = payload_index
        self.visual_index = visual_index

    def retrieve(
        self, observation: InferenceResult, clip: ClipRecord, top_k: int
    ) -> list[MemoryEvidence]:
        del observation
        references = self.visual_index.references(
            fingerprint_query(clip),
            top_k=top_k,
            allowed_experience_ids=set(self.payload_index.payloads),
        )
        return self.payload_index.evidence(references)


class HybridMemoryRetriever:
    def __init__(
        self,
        visual: VisualMemoryRetriever,
        everos: EverOSMemoryRetriever,
        everos_weight: float = 0.25,
    ) -> None:
        self.visual = visual
        self.everos = everos
        self.everos_weight = everos_weight

    def retrieve(
        self, observation: InferenceResult, clip: ClipRecord, top_k: int
    ) -> list[MemoryEvidence]:
        visual = self.visual.retrieve(observation, clip, top_k)
        text = [
            replace(item, retrieval_weight=item.retrieval_weight * self.everos_weight)
            for item in self.everos.retrieve(observation, clip, min(top_k, 8))
        ]
        return visual + text


@dataclass(frozen=True)
class BenchmarkFailure:
    clip_id: str
    path: str
    error: str


@dataclass(frozen=True)
class BenchmarkRow:
    run_id: str
    clip_id: str
    split: str
    label: str
    baseline: InferenceResult
    cheap: InferenceResult
    optimized_final: InferenceResult
    policy: PolicyDecision
    memory_sources: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        optimized_runs = [self.cheap]
        if self.optimized_final.query_id != self.cheap.query_id:
            optimized_runs.append(self.optimized_final)
        return {
            "run_id": self.run_id,
            "clip_id": self.clip_id,
            "split": self.split,
            "label": self.label,
            "baseline": self.baseline.to_dict(),
            "optimized_runs": [result.to_dict() for result in optimized_runs],
            "optimized_final": self.optimized_final.to_dict(),
            "policy": {
                **asdict(self.policy),
                "action": self.policy.action.value,
            },
            "memory_sources": list(self.memory_sources),
        }


@dataclass(frozen=True)
class BenchmarkReport:
    rows: tuple[BenchmarkRow, ...]
    failures: tuple[BenchmarkFailure, ...]
    metrics: dict[str, object]


def _safe_div(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator else 0.0


def _classification_metrics(
    labels: tuple[str, ...], expected: list[str], predicted: list[str]
) -> tuple[float, float]:
    accuracy = _safe_div(
        sum(normalize_label(left) == normalize_label(right) for left, right in zip(expected, predicted)),
        len(expected),
    )
    f1_values = []
    for label in labels:
        true_positive = sum(e == label and p == label for e, p in zip(expected, predicted))
        false_positive = sum(e != label and p == label for e, p in zip(expected, predicted))
        false_negative = sum(e == label and p != label for e, p in zip(expected, predicted))
        precision = _safe_div(true_positive, true_positive + false_positive)
        recall = _safe_div(true_positive, true_positive + false_negative)
        f1_values.append(_safe_div(2 * precision * recall, precision + recall))
    return accuracy, _safe_div(sum(f1_values), len(f1_values))


def _unsafe_recall(
    expected: list[str], predicted: list[str], unsafe_labels: frozenset[str]
) -> float:
    unsafe_indices = [index for index, label in enumerate(expected) if label in unsafe_labels]
    return _safe_div(
        sum(predicted[index] in unsafe_labels for index in unsafe_indices), len(unsafe_indices)
    )


def _cost(results: list[InferenceResult], field: str) -> tuple[float | None, bool]:
    values: list[float] = []
    complete = True
    for result in results:
        if result.error:
            complete = False
            continue
        value = getattr(result, field)
        if value is None or not math.isfinite(float(value)):
            complete = False
        else:
            values.append(float(value))
    return (sum(values) if complete else None), complete


def benchmark_row_from_dict(payload: dict[str, object]) -> BenchmarkRow:
    baseline = InferenceResult(**payload["baseline"])
    optimized_runs = [InferenceResult(**row) for row in payload["optimized_runs"]]
    cheap = next(result for result in optimized_runs if result.action == "cheap_single")
    final_payload = payload["optimized_final"]
    final = InferenceResult(**final_payload)
    policy_payload = dict(payload["policy"])
    policy_payload["action"] = RuntimeAction(policy_payload["action"])
    return BenchmarkRow(
        run_id=str(payload["run_id"]),
        clip_id=str(payload["clip_id"]),
        split=str(payload["split"]),
        label=str(payload["label"]),
        baseline=baseline,
        cheap=cheap,
        optimized_final=final,
        policy=PolicyDecision(**policy_payload),
        memory_sources=tuple(str(value) for value in payload.get("memory_sources", [])),
    )


def load_benchmark_rows(path: Path) -> list[BenchmarkRow]:
    return [
        benchmark_row_from_dict(json.loads(line))
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def attach_actual_costs(
    rows: Iterable[BenchmarkRow], costs_by_query_id: dict[str, float]
) -> list[BenchmarkRow]:
    def reconciled(result: InferenceResult) -> InferenceResult:
        if not result.query_id or result.query_id not in costs_by_query_id:
            return result
        return replace(result, actual_credits=float(costs_by_query_id[result.query_id]))

    output = []
    for row in rows:
        output.append(
            replace(
                row,
                baseline=reconciled(row.baseline),
                cheap=reconciled(row.cheap),
                optimized_final=reconciled(row.optimized_final),
            )
        )
    return output


def write_benchmark_rows(rows: Iterable[BenchmarkRow], path: Path) -> None:
    path = path.resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row.to_dict(), ensure_ascii=False) + "\n")


def benchmark_metrics(
    rows: Iterable[BenchmarkRow], settings: Settings, evidence_source: str
) -> dict[str, object]:
    records = list(rows)
    expected = [normalize_label(row.label) for row in records]
    baseline_predicted = [normalize_label(row.baseline.prediction) for row in records]
    optimized_predicted = [normalize_label(row.optimized_final.prediction) for row in records]
    baseline_accuracy, baseline_macro_f1 = _classification_metrics(
        settings.labels, expected, baseline_predicted
    )
    optimized_accuracy, optimized_macro_f1 = _classification_metrics(
        settings.labels, expected, optimized_predicted
    )
    baseline_unsafe = _unsafe_recall(expected, baseline_predicted, settings.unsafe_labels)
    optimized_unsafe = _unsafe_recall(expected, optimized_predicted, settings.unsafe_labels)

    baseline_runs = [row.baseline for row in records]
    optimized_runs = []
    for row in records:
        optimized_runs.append(row.cheap)
        if row.optimized_final.query_id != row.cheap.query_id:
            optimized_runs.append(row.optimized_final)
    baseline_actual, baseline_actual_complete = _cost(baseline_runs, "actual_credits")
    optimized_actual, optimized_actual_complete = _cost(optimized_runs, "actual_credits")
    actual_complete = baseline_actual_complete and optimized_actual_complete
    actual_reduction = None
    if actual_complete and baseline_actual is not None and optimized_actual is not None:
        actual_reduction = 100 * _safe_div(baseline_actual - optimized_actual, baseline_actual)

    baseline_estimated, _ = _cost(baseline_runs, "estimated_credits")
    optimized_estimated, _ = _cost(optimized_runs, "estimated_credits")
    actions = Counter(row.policy.action.value for row in records)
    recall_delta_pp = 100 * (optimized_unsafe - baseline_unsafe)
    guardrail_pass = recall_delta_pp >= -settings.safety_recall_tolerance_pp
    return {
        "clips": len(records),
        "evidence_source": evidence_source,
        "baseline": {
            "accuracy": baseline_accuracy,
            "macro_f1": baseline_macro_f1,
            "unsafe_recall": baseline_unsafe,
            "latency_ms": sum(row.baseline.latency_ms for row in records),
            "actual_credits": baseline_actual,
            "estimated_credits": baseline_estimated,
            "queries": sum(bool(row.baseline.query_id) for row in records),
        },
        "optimized": {
            "accuracy": optimized_accuracy,
            "macro_f1": optimized_macro_f1,
            "unsafe_recall": optimized_unsafe,
            "latency_ms": sum(result.latency_ms for result in optimized_runs),
            "actual_credits": optimized_actual,
            "estimated_credits": optimized_estimated,
            "queries": sum(bool(result.query_id) for result in optimized_runs),
            "actions": dict(sorted(actions.items())),
        },
        "unsafe_recall_delta_pp": recall_delta_pp,
        "unsafe_recall_guardrail_pass": guardrail_pass,
        "actual_credits_complete": actual_complete,
        "actual_cost_reduction_pct": actual_reduction,
        "claim_ready": evidence_source == "snowflake" and actual_complete and guardrail_pass,
    }


class RuntimeBenchmark:
    def __init__(
        self,
        settings: Settings,
        baseline_backend_factory: Callable[[], InferenceBackend],
        optimized_backend_factory: Callable[[], InferenceBackend],
        retriever: EvidenceRetriever,
        work_dir: Path,
        run_id: str,
        memory_top_k: int = 8,
        workers: int = 4,
        policy: MavisPolicy | None = None,
    ) -> None:
        self.settings = settings
        self.baseline_backend_factory = baseline_backend_factory
        self.optimized_backend_factory = optimized_backend_factory
        self.retriever = retriever
        self.work_dir = work_dir.resolve()
        self.run_id = run_id
        self.memory_top_k = memory_top_k
        self.workers = workers
        self.policy = policy or MavisPolicy()
        self._local = threading.local()
        self._backends: list[InferenceBackend] = []
        self._backend_lock = threading.Lock()

    def _backends_for_thread(self) -> tuple[InferenceBackend, InferenceBackend]:
        pair = getattr(self._local, "backends", None)
        if pair is None:
            pair = (self.baseline_backend_factory(), self.optimized_backend_factory())
            self._local.backends = pair
            with self._backend_lock:
                self._backends.extend(pair)
        return pair

    def _run_one(self, clip: ClipRecord) -> BenchmarkRow:
        candidates = extract_candidates(
            clip,
            self.work_dir / "candidates",
            analysis_fps=self.settings.analysis_fps,
            scan_width=self.settings.scan_width,
            jpeg_quality=self.settings.jpeg_quality,
            include_crop=self.settings.crop_strong,
        )
        baseline_backend, optimized_backend = self._backends_for_thread()
        baseline = baseline_backend.infer(clip, candidates, "strong_multi")
        cheap = optimized_backend.infer(clip, candidates, "cheap_single")
        evidence = self.retriever.retrieve(cheap, clip, top_k=self.memory_top_k)
        decision = self.policy.decide(
            cheap_scores=cheap.scores,
            labels=self.settings.labels,
            risk=cheap.risk,
            need_temporal_context=cheap.need_temporal_context,
            evidence=evidence,
            unsafe_labels=self.settings.unsafe_labels,
        )
        if decision.action == RuntimeAction.ACCEPT_CHEAP:
            final = cheap
        else:
            final = optimized_backend.infer(clip, candidates, decision.action.value)
        return BenchmarkRow(
            run_id=self.run_id,
            clip_id=clip.clip_id,
            split=clip.split,
            label=clip.label,
            baseline=baseline,
            cheap=cheap,
            optimized_final=final,
            policy=decision,
            memory_sources=tuple(
                sorted({item.source_case_id for item in evidence if item.source_case_id})
            ),
        )

    def run(
        self,
        clips: Iterable[ClipRecord],
        output_jsonl: Path,
        progress: Callable[[int, int, str], None] | None = None,
    ) -> BenchmarkReport:
        records = list(clips)
        rows: list[BenchmarkRow] = []
        failures: list[BenchmarkFailure] = []
        try:
            with ThreadPoolExecutor(max_workers=max(1, self.workers)) as executor:
                futures = {executor.submit(self._run_one, clip): clip for clip in records}
                for completed, future in enumerate(as_completed(futures), start=1):
                    clip = futures[future]
                    try:
                        rows.append(future.result())
                        status = "completed"
                    except Exception as exc:
                        failures.append(
                            BenchmarkFailure(
                                clip_id=clip.clip_id,
                                path=str(clip.path),
                                error=f"{type(exc).__name__}: {exc}",
                            )
                        )
                        status = "failed"
                    if progress:
                        progress(completed, len(records), f"{clip.clip_id} {status}")
        finally:
            for backend in self._backends:
                backend.close()

        rows.sort(key=lambda row: row.clip_id)
        write_benchmark_rows(rows, output_jsonl)
        source = self._backends[0].evidence_source if self._backends else "unknown"
        metrics = benchmark_metrics(rows, self.settings, source)
        return BenchmarkReport(tuple(rows), tuple(failures), metrics)
