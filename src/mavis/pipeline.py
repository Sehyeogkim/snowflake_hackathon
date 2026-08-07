from __future__ import annotations

import json
import threading
from collections.abc import Callable, Iterable
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, replace
from pathlib import Path

from .backends.base import InferenceBackend
from .candidates import extract_candidates
from .config import Settings
from .evaluation import build_experience, evaluate_actions
from .models import ClipRecord, Experience
from .store import RunStore


@dataclass(frozen=True)
class PipelineFailure:
    clip_id: str
    path: str
    error: str


@dataclass(frozen=True)
class PipelineReport:
    created: int
    skipped: int
    failures: tuple[PipelineFailure, ...]


class SeedPipeline:
    def __init__(
        self,
        settings: Settings,
        backend_factory: Callable[[], InferenceBackend],
        store: RunStore,
        work_dir: Path,
        workers: int | None = None,
    ) -> None:
        self.settings = settings
        self.backend_factory = backend_factory
        self.store = store
        self.work_dir = work_dir.resolve()
        self.workers = workers or settings.max_workers
        self._local = threading.local()

    def _backend(self) -> InferenceBackend:
        backend = getattr(self._local, "backend", None)
        if backend is None:
            backend = self.backend_factory()
            self._local.backend = backend
        return backend

    def _run_one(self, clip: ClipRecord) -> Experience:
        candidates = extract_candidates(
            clip,
            self.work_dir / "candidates",
            analysis_fps=self.settings.analysis_fps,
            scan_width=self.settings.scan_width,
            jpeg_quality=self.settings.jpeg_quality,
            include_crop=self.settings.crop_strong,
        )
        actions = list(self.settings.actions)
        if self.settings.crop_strong and "crop_strong" not in actions:
            actions.append("crop_strong")
        backend = self._backend()
        results = [backend.infer(clip, candidates, action) for action in actions]
        evaluated = evaluate_actions(clip.label, results, self.settings.labels)
        experience = build_experience(
            clip_id=clip.clip_id,
            label=clip.label,
            split=clip.split,
            observations=evaluated,
            prompt_version=self.settings.prompt_version,
        )
        experience = replace(
            experience,
            metadata={
                **experience.metadata,
                "evidence_source": backend.evidence_source,
                "actual_costs_complete": all(
                    result.actual_credits is not None for result in results
                ),
            },
        )
        self.store.save_experience(experience)
        return experience

    def run(
        self,
        clips: Iterable[ClipRecord],
        output_jsonl: Path | None = None,
        progress: Callable[[int, int, str], None] | None = None,
    ) -> PipelineReport:
        records = list(clips)
        todo: list[ClipRecord] = []
        skipped = 0
        for clip in records:
            if self.store.is_complete(clip.clip_id, self.settings.prompt_version):
                skipped += 1
            else:
                todo.append(clip)

        output_handle = None
        if output_jsonl is not None:
            output_jsonl.parent.mkdir(parents=True, exist_ok=True)
            output_handle = output_jsonl.open("a", encoding="utf-8", newline="\n")

        created = 0
        failures: list[PipelineFailure] = []
        try:
            with ThreadPoolExecutor(max_workers=max(1, self.workers)) as executor:
                futures = {executor.submit(self._run_one, clip): clip for clip in todo}
                for completed, future in enumerate(as_completed(futures), start=1):
                    clip = futures[future]
                    try:
                        experience = future.result()
                        created += 1
                        if output_handle is not None:
                            output_handle.write(
                                json.dumps(experience.to_dict(), ensure_ascii=False) + "\n"
                            )
                            output_handle.flush()
                        status = "created"
                    except Exception as exc:
                        failures.append(
                            PipelineFailure(
                                clip_id=clip.clip_id,
                                path=str(clip.path),
                                error=f"{type(exc).__name__}: {exc}",
                            )
                        )
                        status = "failed"
                    if progress:
                        progress(completed, len(todo), f"{clip.clip_id} {status}")
        finally:
            if output_handle is not None:
                output_handle.close()
        return PipelineReport(created=created, skipped=skipped, failures=tuple(failures))
