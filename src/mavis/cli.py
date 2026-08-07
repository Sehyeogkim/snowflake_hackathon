from __future__ import annotations

import argparse
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict
from pathlib import Path

from .backends.mock import MockInferenceBackend
from .candidates import extract_candidates
from .config import Settings, load_dotenv
from .everos import EverOSClient
from .fingerprints import fingerprint_many, near_duplicate_pairs
from .gemini_teacher import (
    GeminiAnalyzer,
    GeminiTeacherStore,
    build_proxies,
    export_experiences,
    load_experiences,
    run_cheap_pass,
    run_strong_counterfactuals,
)
from .jobs import plan_seed_jobs
from .manifest import (
    assign_grouped_splits,
    discover_clips,
    merge_duplicate_groups,
    read_manifest,
    summarize,
    write_manifest,
)
from .memory_evidence import PortableExperienceIndex
from .pipeline import SeedPipeline
from .runtime import (
    EverOSMemoryRetriever,
    HybridMemoryRetriever,
    LocalMemoryRetriever,
    NoMemoryRetriever,
    RuntimeBenchmark,
    VisualMemoryRetriever,
    attach_actual_costs,
    benchmark_metrics,
    load_benchmark_rows,
    write_benchmark_rows,
)
from .store import RunStore
from .visual_memory import VisualMemoryIndex, build_visual_memory_index

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "default.json"


def _json(value: object) -> None:
    print(json.dumps(value, ensure_ascii=False, indent=2))


def _settings(path: str) -> Settings:
    return Settings.load(Path(path).resolve())


def cmd_doctor(args: argparse.Namespace) -> int:
    load_dotenv(Path(args.env))
    checks = {
        "python": sys.version.split()[0],
        "config_exists": Path(args.config).exists(),
        "snowflake_credentials_present": all(
            os.getenv(name)
            for name in (
                "SNOWFLAKE_ACCOUNT",
                "SNOWFLAKE_USER",
                "SNOWFLAKE_WAREHOUSE",
                "SNOWFLAKE_DATABASE",
                "SNOWFLAKE_SCHEMA",
            )
        )
        and bool(
            os.getenv("SNOWFLAKE_PASSWORD")
            or os.getenv("SNOWFLAKE_PRIVATE_KEY_FILE")
            or os.getenv("SNOWFLAKE_AUTHENTICATOR", "snowflake") != "snowflake"
        ),
        "everos_credentials_present": bool(os.getenv("EVEROS_API_KEY")),
        "gemini_credentials_present": bool(os.getenv("GEMINI_API_KEY")),
        "openai_credentials_present": bool(os.getenv("OPENAI_API_KEY")),
    }
    try:
        import cv2

        checks["opencv"] = cv2.__version__
    except ImportError:
        checks["opencv"] = False
    try:
        import snowflake.connector  # noqa: F401

        checks["snowflake_connector"] = True
    except ImportError:
        checks["snowflake_connector"] = False
    _json(checks)
    return 0


def cmd_inspect_data(args: argparse.Namespace) -> int:
    root = Path(args.root).resolve()
    records = discover_clips(root)
    all_videos = [
        path
        for path in root.rglob("*")
        if path.is_file() and path.suffix.casefold() in {".mp4", ".avi", ".mov", ".mkv"}
    ]
    report = summarize(records)
    report["root"] = str(root)
    report["all_video_files"] = len(all_videos)
    report["unlabeled_video_files"] = len(all_videos) - len(records)
    if args.list_unlabeled:
        labeled = {row.path for row in records}
        report["unlabeled_paths"] = [str(path) for path in all_videos if path not in labeled]
    _json(report)
    return 0 if records else 2


def cmd_prepare_manifest(args: argparse.Namespace) -> int:
    records = discover_clips(Path(args.root))
    if not records:
        print("No labeled video clips found.", file=sys.stderr)
        return 2
    duplicate_pairs = []
    if args.duplicate_report:
        duplicate_pairs = json.loads(Path(args.duplicate_report).read_text(encoding="utf-8"))
        records = merge_duplicate_groups(
            records,
            duplicate_pairs,
            max_mean_hamming=args.duplicate_max_hamming,
            max_duration_delta_s=args.duplicate_max_duration,
        )
    forced_seed_ids: frozenset[str] = frozenset()
    if args.preserve_memory_experiences:
        forced_seed_ids = frozenset(
            str(row.get("payload", row).get("clip_id"))
            for row in (
                json.loads(line)
                for line in Path(args.preserve_memory_experiences)
                .read_text(encoding="utf-8")
                .splitlines()
                if line.strip()
            )
        )
    assigned = assign_grouped_splits(
        records,
        seed_count=args.seed_count,
        dev_count=args.dev_count,
        random_seed=args.random_seed,
        test_count=args.test_count,
        forced_seed_clip_ids=forced_seed_ids,
    )
    write_manifest(assigned, Path(args.output))
    report = summarize(assigned)
    report["manifest"] = str(Path(args.output).resolve())
    report["duplicate_report"] = (
        str(Path(args.duplicate_report).resolve()) if args.duplicate_report else None
    )
    report["duplicate_pairs_considered"] = len(duplicate_pairs)
    report["preserved_memory_clips"] = len(forced_seed_ids)
    _json(report)
    return 0


def _backend_factory(
    name: str, settings: Settings, query_tag_prefix: str = "mavis|seed"
):
    if name == "mock":
        return lambda: MockInferenceBackend(
            settings.cheap_model,
            settings.strong_model,
            seed=int(os.getenv("MAVIS_RUN_SEED", "42")),
        )
    if name == "snowflake":
        from .backends.snowflake import SnowflakeInferenceBackend

        return lambda: SnowflakeInferenceBackend(
            labels=settings.labels,
            cheap_model=settings.cheap_model,
            strong_model=settings.strong_model,
            query_tag_prefix=query_tag_prefix,
        )
    if name == "openai":
        from .backends.openai import TieredOpenAIInferenceBackend

        return lambda: TieredOpenAIInferenceBackend(
            labels=settings.labels,
            cheap_model=os.getenv("OPENAI_CHEAP_MODEL", "gpt-5.6-luna"),
            strong_model=os.getenv("OPENAI_STRONG_MODEL", "gpt-5.6-terra"),
        )
    raise ValueError(name)


def cmd_generate_seeds(args: argparse.Namespace) -> int:
    load_dotenv(Path(args.env))
    settings = _settings(args.config)
    clips = list(read_manifest(Path(args.manifest), split=args.split))
    limit = args.limit if args.limit is not None else settings.max_seed_clips
    clips = clips[:limit]
    store = RunStore(Path(args.database))
    pipeline = SeedPipeline(
        settings=settings,
        backend_factory=_backend_factory(args.backend, settings),
        store=store,
        work_dir=Path(args.work_dir),
        workers=args.workers or settings.max_workers,
    )

    def progress(done: int, total: int, status: str) -> None:
        print(f"[{done:>3}/{total}] {status}", file=sys.stderr)

    report = pipeline.run(clips, Path(args.output), progress=progress)
    payload = {
        "backend": args.backend,
        "created": report.created,
        "skipped": report.skipped,
        "failures": [failure.__dict__ for failure in report.failures],
        "store": store.summary(),
        "mock_results_are_not_benchmark_evidence": args.backend == "mock",
    }
    _json(payload)
    return 1 if report.failures else 0


def cmd_extract_candidates(args: argparse.Namespace) -> int:
    settings = _settings(args.config)
    clips = list(read_manifest(Path(args.manifest), split=args.split))
    if args.limit is not None:
        clips = clips[: args.limit]
    output_root = Path(args.work_dir).resolve() / "candidates"

    def run_one(clip):
        return extract_candidates(
            clip,
            output_root,
            analysis_fps=settings.analysis_fps,
            scan_width=settings.scan_width,
            jpeg_quality=settings.jpeg_quality,
            include_crop=settings.crop_strong,
        )

    completed = 0
    failures = []
    with ThreadPoolExecutor(max_workers=max(1, args.workers or settings.max_workers)) as executor:
        futures = {executor.submit(run_one, clip): clip for clip in clips}
        for future in as_completed(futures):
            clip = futures[future]
            try:
                future.result()
                completed += 1
            except Exception as exc:
                failures.append(
                    {"clip_id": clip.clip_id, "path": str(clip.path), "error": str(exc)}
                )
            print(f"[{completed + len(failures):>3}/{len(clips)}] {clip.clip_id}", file=sys.stderr)
    _json(
        {
            "actual_dataset_clips": len(clips),
            "candidates_ready": completed,
            "failures": failures,
            "output_root": str(output_root),
        }
    )
    return 1 if failures else 0


def cmd_plan_jobs(args: argparse.Namespace) -> int:
    settings = _settings(args.config)
    clips = list(read_manifest(Path(args.manifest), split=args.split))
    if args.limit is not None:
        clips = clips[: args.limit]
    _json(
        plan_seed_jobs(
            clips,
            settings,
            Path(args.work_dir),
            Path(args.output),
            run_id=args.run_id,
        )
    )
    return 0


def cmd_push_everos(args: argparse.Namespace) -> int:
    load_dotenv(Path(args.env))
    store = RunStore(Path(args.database))
    pending = store.pending_everos(limit=args.limit)
    if not pending:
        _json({"pushed": 0, "message": "No pending experiences."})
        return 0
    session_id = args.session_id
    pushed = 0
    with EverOSClient() as client:

        def push_one(row: dict) -> dict:
            clip_id = row["payload"]["clip_id"]
            row_session = f"{session_id}-{clip_id}-{row['experience_id'][:8]}"
            add_response = client.add_durable_experience(row, row_session)
            store.mark_everos(
                row["experience_id"],
                "queued",
                {"session_id": row_session, "add": add_response},
            )
            flush_response = client.flush(row_session)
            verification = client.get_session_episodes(row_session)
            returned_episodes = verification.get("episodes") or verification.get("data", {}).get(
                "episodes", []
            )
            episodes = [
                episode for episode in returned_episodes if episode.get("session_id") == row_session
            ]
            if not episodes and args.verify_wait > 0:
                time.sleep(args.verify_wait)
                verification = client.get_session_episodes(row_session)
                returned_episodes = verification.get("episodes") or verification.get(
                    "data", {}
                ).get("episodes", [])
                episodes = [
                    episode
                    for episode in returned_episodes
                    if episode.get("session_id") == row_session
                ]
            return {
                "experience_id": row["experience_id"],
                "session_id": row_session,
                "add": add_response,
                "flush": flush_response,
                "verification": verification,
                "verified": bool(episodes),
                "verified_episode_ids": [episode.get("id") for episode in episodes],
                "memory_type": "episode",
            }

        responses = []
        failures = []
        with ThreadPoolExecutor(max_workers=max(1, args.workers)) as executor:
            futures = {executor.submit(push_one, row): row for row in pending}
            for future in as_completed(futures):
                row = futures[future]
                try:
                    response = future.result()
                    status = "verified" if response["verified"] else "flushed_unverified"
                    store.mark_everos(row["experience_id"], status, response)
                    responses.append(response)
                    pushed += 1
                except Exception as exc:
                    failures.append(
                        {
                            "experience_id": row["experience_id"],
                            "error": f"{type(exc).__name__}: {exc}",
                        }
                    )
        _json({"pushed": pushed, "failures": failures, "sessions": responses})
        return 1 if failures else 0


def cmd_search_everos(args: argparse.Namespace) -> int:
    load_dotenv(Path(args.env))
    with EverOSClient() as client:
        _json(client.search(args.query, top_k=args.top_k, method=args.method, scope=args.scope))
    return 0


def cmd_get_everos(args: argparse.Namespace) -> int:
    load_dotenv(Path(args.env))
    with EverOSClient() as client:
        _json(client.get(args.memory_type, page=args.page, page_size=args.page_size))
    return 0


def cmd_summary(args: argparse.Namespace) -> int:
    _json(RunStore(Path(args.database)).summary())
    return 0


def cmd_audit_leakage(args: argparse.Namespace) -> int:
    clips = list(read_manifest(Path(args.manifest)))
    fingerprints, failures = fingerprint_many(clips, workers=args.workers)
    pairs = near_duplicate_pairs(
        fingerprints,
        max_mean_hamming=args.max_mean_hamming,
        max_duration_delta_s=args.max_duration_delta,
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(pairs, ensure_ascii=False, indent=2), encoding="utf-8")
    cross_split = [pair for pair in pairs if pair["cross_split"]]
    _json(
        {
            "clips_fingerprinted": len(fingerprints),
            "failures": failures,
            "near_duplicate_pairs": len(pairs),
            "cross_split_pairs": len(cross_split),
            "report": str(output.resolve()),
            "thresholds": {
                "max_mean_hamming": args.max_mean_hamming,
                "max_duration_delta_s": args.max_duration_delta,
            },
            "warning": "pHash candidates require visual review before regrouping",
        }
    )
    return 1 if failures else 0


def cmd_reconcile_snowflake(args: argparse.Namespace) -> int:
    load_dotenv(Path(args.env))
    settings = _settings(args.config)
    store = RunStore(Path(args.database))
    query_ids = store.query_ids_without_actual_cost()
    if not query_ids:
        _json({"pending_query_ids": 0, "updated": 0, "store": store.summary()})
        return 0
    from .backends.snowflake import SnowflakeInferenceBackend

    backend = SnowflakeInferenceBackend(
        labels=settings.labels,
        cheap_model=settings.cheap_model,
        strong_model=settings.strong_model,
    )
    try:
        costs = backend.reconcile_costs(query_ids, wait_seconds=args.wait_seconds)
    finally:
        backend.close()
    updated = store.update_actual_costs(costs)
    rebuilt = store.rebuild_reconciled(settings.labels)
    _json(
        {
            "pending_query_ids_before": len(query_ids),
            "usage_rows_found": len(costs),
            "action_rows_updated": updated,
            "experiences_finalized": rebuilt,
            "still_pending": len(store.query_ids_without_actual_cost()),
            "store": store.summary(),
        }
    )
    return 0


def cmd_benchmark(args: argparse.Namespace) -> int:
    load_dotenv(Path(args.env))
    settings = _settings(args.config)
    clips = list(read_manifest(Path(args.manifest), split=args.split))
    if args.limit is not None:
        clips = clips[: args.limit]
    index = PortableExperienceIndex.from_jsonl(
        Path(args.experiences), limit=args.memory_limit
    )

    def run(retriever) -> dict[str, object]:
        runner = RuntimeBenchmark(
            settings=settings,
            baseline_backend_factory=_backend_factory(
                args.backend, settings, f"mavis|baseline|{args.run_id}"
            ),
            optimized_backend_factory=_backend_factory(
                args.backend, settings, f"mavis|optimized|{args.run_id}"
            ),
            retriever=retriever,
            work_dir=Path(args.work_dir),
            run_id=args.run_id,
            memory_top_k=args.memory_top_k,
            workers=args.workers,
        )

        def progress(done: int, total: int, status: str) -> None:
            print(f"[{done:>3}/{total}] {status}", file=sys.stderr, flush=True)

        report = runner.run(clips, Path(args.output), progress=progress)
        return {
            "run_id": args.run_id,
            "backend": args.backend,
            "memory_source": args.memory_source,
            "memory_limit": len(index.payloads),
            "output": str(Path(args.output).resolve()),
            "metrics": report.metrics,
            "failures": [asdict(failure) for failure in report.failures],
            "mock_results_are_not_benchmark_evidence": args.backend == "mock",
        }

    if args.memory_source == "none":
        payload = run(NoMemoryRetriever())
    elif args.memory_source == "local":
        payload = run(LocalMemoryRetriever(index))
    elif args.memory_source == "visual":
        visual_index = VisualMemoryIndex.load(Path(args.visual_index))
        payload = run(VisualMemoryRetriever(index, visual_index))
    elif args.memory_source == "everos":
        with EverOSClient() as client:
            payload = run(EverOSMemoryRetriever(index, client))
    else:
        visual_index = VisualMemoryIndex.load(Path(args.visual_index))
        with EverOSClient() as client:
            payload = run(
                HybridMemoryRetriever(
                    VisualMemoryRetriever(index, visual_index),
                    EverOSMemoryRetriever(index, client),
                )
            )
    _json(payload)
    return 1 if payload["failures"] else 0


def cmd_reconcile_benchmark(args: argparse.Namespace) -> int:
    load_dotenv(Path(args.env))
    settings = _settings(args.config)
    path = Path(args.input)
    rows = load_benchmark_rows(path)
    query_ids = sorted(
        {
            result.query_id
            for row in rows
            for result in (row.baseline, row.cheap, row.optimized_final)
            if result.query_id and result.actual_credits is None
        }
    )
    if query_ids:
        from .backends.snowflake import SnowflakeInferenceBackend

        backend = SnowflakeInferenceBackend(
            labels=settings.labels,
            cheap_model=settings.cheap_model,
            strong_model=settings.strong_model,
            query_tag_prefix="mavis|reconcile",
        )
        try:
            costs = backend.reconcile_costs(query_ids, wait_seconds=args.wait_seconds)
        finally:
            backend.close()
        rows = attach_actual_costs(rows, costs)
        write_benchmark_rows(rows, path)
    else:
        costs = {}
    metrics = benchmark_metrics(rows, settings, "snowflake")
    _json(
        {
            "input": str(path.resolve()),
            "pending_query_ids_before": len(query_ids),
            "usage_rows_found": len(costs),
            "metrics": metrics,
        }
    )
    return 0


def cmd_build_visual_memory(args: argparse.Namespace) -> int:
    rows = [
        json.loads(line)
        for line in Path(args.experiences).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    experience_ids_by_clip = {
        str(row.get("payload", row)["clip_id"]): str(row["experience_id"]) for row in rows
    }
    clips = list(read_manifest(Path(args.manifest)))
    index, failures = build_visual_memory_index(
        clips,
        experience_ids_by_clip,
        workers=args.workers,
        proxy_dir=Path(args.proxy_dir) if args.proxy_dir else None,
    )
    index.save(Path(args.output))
    _json(
        {
            "entries": len(index.entries),
            "failures": failures,
            "output": str(Path(args.output).resolve()),
            "uses_real_dataset_clips": True,
        }
    )
    return 1 if failures else 0


def cmd_train_gemini(args: argparse.Namespace) -> int:
    load_dotenv(Path(args.env))
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("Missing required environment variable: GEMINI_API_KEY")
    clips = list(read_manifest(Path(args.manifest), split=args.split))
    if args.limit is not None:
        clips = clips[: args.limit]
    store = GeminiTeacherStore(Path(args.database))
    store.add_prior_spend(
        args.prior_spend_usd,
        {"reason": "credential_and_10fps_smoke", "included_in_hard_cap": True},
    )
    recovered_reservations = store.settle_stale_reservations()
    analyzer = GeminiAnalyzer(api_key)

    def progress(done: int, total: int, status: str) -> None:
        if done == 1 or done == total or done % 10 == 0 or "failed" in status:
            budget = store.budget()
            print(
                f"[{done:>3}/{total}] {status} spend=${budget['spent']:.4f} "
                f"reserved=${budget['reserved']:.4f}",
                file=sys.stderr,
                flush=True,
            )

    work_dir = Path(args.work_dir).resolve()
    proxies, proxy_failures = build_proxies(
        clips, work_dir / "dense10", args.fps, args.proxy_workers, progress
    )
    cheap = run_cheap_pass(
        clips,
        proxies,
        analyzer,
        store,
        args.fps,
        args.budget_usd,
        args.workers,
        progress,
    )
    strong = run_strong_counterfactuals(
        analyzer, store, work_dir, args.fps, args.budget_usd, progress
    )
    experience_count = export_experiences(store, Path(args.output))
    _json(
        {
            "actual_dataset_clips": len(clips),
            "conservatively_charged_interrupted_reservations": recovered_reservations,
            "proxy_failures": proxy_failures,
            "cheap_pass": cheap,
            "strong_counterfactual_pairs": strong,
            "experiences": experience_count,
            "experience_output": str(Path(args.output).resolve()),
            "database": str(Path(args.database).resolve()),
            "summary": store.summary(args.budget_usd),
            "cost_note": "Estimated public list price from API token usage; invoice can differ.",
        }
    )
    return 1 if proxy_failures else 0


def cmd_push_gemini_everos(args: argparse.Namespace) -> int:
    load_dotenv(Path(args.env))
    store = GeminiTeacherStore(Path(args.database))
    rows = [
        row
        for row in load_experiences(Path(args.input))
        if not store.everos_done(row["experience_id"])
    ]
    if args.limit is not None:
        rows = rows[: args.limit]
    responses: list[dict] = []
    failures: list[dict] = []

    if args.batch:
        session_id = args.session_id
        with EverOSClient() as client:
            added = client.add_experience_batch(rows, session_id)
            flushed = client.flush(session_id)
            if args.verify_wait:
                time.sleep(args.verify_wait)
            verification = client.get_session_cases(session_id)
        cases = verification.get("agent_cases") or verification.get("data", {}).get(
            "agent_cases", []
        )
        if len(cases) >= len(rows):
            status = "verified"
        elif cases:
            status = "partial"
        else:
            status = "flushed"
        response = {
            "session_id": session_id,
            "operation": "batch_tool_trajectory",
            "experience_count": len(rows),
            "add": added,
            "flush": flushed,
            "verification": verification,
            "case_count": len(cases),
        }
        for row in rows:
            store.mark_everos(row["experience_id"], status, response)
        _json(
            {
                "attempted": len(rows),
                "batch_session": session_id,
                "agent_cases": len(cases),
                "status": status,
            }
        )
        return 0

    def push_one(client: EverOSClient, row: dict) -> dict:
        experience_id = row["experience_id"]
        session_id = f"{args.session_id}-{experience_id}"
        added = client.add_durable_experience(row, session_id)
        operation = "verified_gt_evaluated_episode"
        flushed = client.flush(session_id)
        verification = client.get_session_episodes(session_id)
        returned_episodes = verification.get("episodes") or verification.get("data", {}).get(
            "episodes", []
        )
        episodes = [
            episode for episode in returned_episodes if episode.get("session_id") == session_id
        ]
        if not episodes and args.verify_wait:
            time.sleep(args.verify_wait)
            verification = client.get_session_episodes(session_id)
            returned_episodes = verification.get("episodes") or verification.get("data", {}).get(
                "episodes", []
            )
            episodes = [
                episode for episode in returned_episodes if episode.get("session_id") == session_id
            ]
        result = {
            "experience_id": experience_id,
            "session_id": session_id,
            "operation": operation,
            "add": added,
            "flush": flushed,
            "verification": verification,
            "verified": bool(episodes),
            "verified_episode_ids": [episode.get("id") for episode in episodes],
            "memory_type": "episode",
        }
        store.mark_everos(experience_id, "verified" if episodes else "flushed", result)
        return result

    with (
        EverOSClient() as client,
        ThreadPoolExecutor(max_workers=max(1, args.workers)) as executor,
    ):
        futures = {executor.submit(push_one, client, row): row for row in rows}
        for future in as_completed(futures):
            row = futures[future]
            try:
                response = future.result()
                responses.append(response)
            except Exception as exc:
                failure = {
                    "experience_id": row["experience_id"],
                    "error": f"{type(exc).__name__}: {exc}",
                }
                failures.append(failure)
                store.mark_everos(row["experience_id"], "failed", failure)
    _json(
        {
            "attempted": len(rows),
            "verified_episodes": sum(bool(row["verified"]) for row in responses),
            "flushed_unverified": sum(not row["verified"] for row in responses),
            "failures": failures,
        }
    )
    return 1 if failures else 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="mavis", description="MAVIS seed pipeline")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--env", default=".env")
    sub = parser.add_subparsers(dest="command", required=True)

    doctor = sub.add_parser("doctor", help="Check local dependencies and credential presence")
    doctor.set_defaults(func=cmd_doctor)

    inspect_data = sub.add_parser("inspect-data", help="Inspect dataset layout and labels")
    inspect_data.add_argument("--root", required=True)
    inspect_data.add_argument("--list-unlabeled", action="store_true")
    inspect_data.set_defaults(func=cmd_inspect_data)

    manifest = sub.add_parser("prepare-manifest", help="Build leakage-aware seed/dev/test JSONL")
    manifest.add_argument("--root", required=True)
    manifest.add_argument("--output", default="data/manifest.jsonl")
    manifest.add_argument("--seed-count", type=int, default=300)
    manifest.add_argument("--dev-count", type=int, default=91)
    manifest.add_argument(
        "--test-count",
        type=int,
        help="Keep this held-out test size; dev absorbs forced duplicate-group growth",
    )
    manifest.add_argument("--random-seed", type=int, default=42)
    manifest.add_argument(
        "--duplicate-report",
        help="Fingerprint-pair JSON from audit-leakage; merge matches before splitting",
    )
    manifest.add_argument("--duplicate-max-hamming", type=float, default=0.0)
    manifest.add_argument("--duplicate-max-duration", type=float, default=0.0)
    manifest.add_argument(
        "--preserve-memory-experiences",
        help="Portable Experience JSONL whose clip groups must remain in seed",
    )
    manifest.set_defaults(func=cmd_prepare_manifest)

    generate = sub.add_parser("generate-seeds", help="Extract frames and generate experiences")
    generate.add_argument("--manifest", default="data/manifest.jsonl")
    generate.add_argument("--split", default="seed")
    generate.add_argument(
        "--backend", choices=("mock", "openai", "snowflake"), default="mock"
    )
    generate.add_argument("--limit", type=int)
    generate.add_argument("--workers", type=int)
    generate.add_argument("--database", default="outputs/mavis.sqlite3")
    generate.add_argument("--work-dir", default="data/work")
    generate.add_argument("--output", default="outputs/experiences.jsonl")
    generate.set_defaults(func=cmd_generate_seeds)

    extract = sub.add_parser(
        "extract-candidates", help="Precompute real video candidates without API keys"
    )
    extract.add_argument("--manifest", default="data/manifest.jsonl")
    extract.add_argument("--split", default="seed")
    extract.add_argument("--limit", type=int)
    extract.add_argument("--workers", type=int)
    extract.add_argument("--work-dir", default="data/work")
    extract.set_defaults(func=cmd_extract_candidates)

    plan_jobs = sub.add_parser(
        "plan-jobs", help="Freeze deterministic Snowflake inference jobs without executing them"
    )
    plan_jobs.add_argument("--manifest", default="data/manifest.jsonl")
    plan_jobs.add_argument("--split", default="seed")
    plan_jobs.add_argument("--limit", type=int)
    plan_jobs.add_argument("--work-dir", default="data/work")
    plan_jobs.add_argument("--output", default="outputs/seed_jobs.jsonl")
    plan_jobs.add_argument("--run-id", default="seed-v1")
    plan_jobs.set_defaults(func=cmd_plan_jobs)

    push = sub.add_parser("push-everos", help="Push locally generated experiences to EverOS")
    push.add_argument("--database", default="outputs/mavis.sqlite3")
    push.add_argument("--session-id", default="mavis-seed")
    push.add_argument("--workers", type=int, default=6)
    push.add_argument("--limit", type=int)
    push.add_argument("--verify-wait", type=float, default=2.0)
    push.set_defaults(func=cmd_push_everos)

    search = sub.add_parser("search-everos", help="Inspect actual EverOS cases/skills/episodes")
    search.add_argument("query")
    search.add_argument("--top-k", type=int, default=10)
    search.add_argument(
        "--method", choices=("keyword", "vector", "hybrid", "agentic"), default="hybrid"
    )
    search.add_argument(
        "--scope",
        choices=("user", "agent"),
        default="user",
        help="Search durable episodes (user) or extracted cases/skills (agent)",
    )
    search.set_defaults(func=cmd_search_everos)

    get_memory = sub.add_parser(
        "get-everos", help="List actual extracted EverOS agent cases or skills"
    )
    get_memory.add_argument(
        "memory_type", choices=("episode", "profile", "agent_case", "agent_skill")
    )
    get_memory.add_argument("--page", type=int, default=1)
    get_memory.add_argument("--page-size", type=int, default=50)
    get_memory.set_defaults(func=cmd_get_everos)

    summary = sub.add_parser("summary", help="Show local seed and cost summary")
    summary.add_argument("--database", default="outputs/mavis.sqlite3")
    summary.set_defaults(func=cmd_summary)

    audit = sub.add_parser(
        "audit-leakage", help="Flag visually similar clips that cross frozen splits"
    )
    audit.add_argument("--manifest", default="data/manifest.jsonl")
    audit.add_argument("--workers", type=int, default=8)
    audit.add_argument("--max-mean-hamming", type=float, default=5.0)
    audit.add_argument("--max-duration-delta", type=float, default=0.75)
    audit.add_argument("--output", default="outputs/leakage_candidates.json")
    audit.set_defaults(func=cmd_audit_leakage)

    reconcile = sub.add_parser(
        "reconcile-snowflake", help="Attach billed Cortex credits and finalize trajectories"
    )
    reconcile.add_argument("--database", default="outputs/mavis.sqlite3")
    reconcile.add_argument("--wait-seconds", type=int, default=0)
    reconcile.set_defaults(func=cmd_reconcile_snowflake)

    benchmark = sub.add_parser(
        "benchmark", help="Run strong baseline and memory-aware MAVIS on the same held-out clips"
    )
    benchmark.add_argument("--manifest", default="data/manifest.jsonl")
    benchmark.add_argument("--split", default="dev")
    benchmark.add_argument(
        "--backend", choices=("mock", "openai", "snowflake"), default="mock"
    )
    benchmark.add_argument("--run-id", default="dev-v1")
    benchmark.add_argument("--limit", type=int)
    benchmark.add_argument("--workers", type=int, default=4)
    benchmark.add_argument("--work-dir", default="data/work/runtime")
    benchmark.add_argument("--output", default="outputs/benchmark.jsonl")
    benchmark.add_argument("--experiences", default="outputs/gemini_experiences.jsonl")
    benchmark.add_argument(
        "--memory-source",
        choices=("none", "local", "everos", "visual", "hybrid"),
        default="hybrid",
    )
    benchmark.add_argument("--memory-limit", type=int, default=300)
    benchmark.add_argument("--memory-top-k", type=int, default=15)
    benchmark.add_argument("--visual-index", default="outputs/visual_memory.json")
    benchmark.set_defaults(func=cmd_benchmark)

    reconcile_benchmark = sub.add_parser(
        "reconcile-benchmark", help="Attach billed Snowflake credits to a benchmark ledger"
    )
    reconcile_benchmark.add_argument("--input", default="outputs/benchmark.jsonl")
    reconcile_benchmark.add_argument("--wait-seconds", type=int, default=0)
    reconcile_benchmark.set_defaults(func=cmd_reconcile_benchmark)

    visual_memory = sub.add_parser(
        "build-visual-memory", help="Cache OpenCV scene signatures for the 300 EverOS episodes"
    )
    visual_memory.add_argument("--manifest", default="data/manifest_final.jsonl")
    visual_memory.add_argument("--experiences", default="outputs/gemini_experiences.jsonl")
    visual_memory.add_argument("--proxy-dir", default="data/work/gemini/dense10")
    visual_memory.add_argument("--workers", type=int, default=8)
    visual_memory.add_argument("--output", default="outputs/visual_memory.json")
    visual_memory.set_defaults(func=cmd_build_visual_memory)

    gemini = sub.add_parser(
        "train-gemini",
        help="Run real 10 fps Gemini teachers with a strict estimated list-price cap",
    )
    gemini.add_argument("--manifest", default="data/manifest.jsonl")
    gemini.add_argument("--split", default="seed")
    gemini.add_argument("--limit", type=int, default=300)
    gemini.add_argument("--fps", type=float, default=10.0)
    gemini.add_argument("--budget-usd", type=float, default=4.0)
    gemini.add_argument("--prior-spend-usd", type=float, default=0.0)
    gemini.add_argument("--workers", type=int, default=4)
    gemini.add_argument("--proxy-workers", type=int, default=8)
    gemini.add_argument("--database", default="outputs/gemini_teacher.sqlite3")
    gemini.add_argument("--work-dir", default="data/work/gemini")
    gemini.add_argument("--output", default="outputs/gemini_experiences.jsonl")
    gemini.set_defaults(func=cmd_train_gemini)

    push_gemini = sub.add_parser(
        "push-gemini-everos", help="Push validated Gemini counterfactual experiences to EverOS"
    )
    push_gemini.add_argument("--input", default="outputs/gemini_experiences.jsonl")
    push_gemini.add_argument("--database", default="outputs/gemini_teacher.sqlite3")
    push_gemini.add_argument("--session-id", default="mavis-gemini-seed-v2")
    push_gemini.add_argument("--workers", type=int, default=4)
    push_gemini.add_argument("--limit", type=int)
    push_gemini.add_argument("--verify-wait", type=float, default=2.0)
    push_gemini.add_argument(
        "--replay-full",
        action="store_true",
        help="Send a full tool-call trajectory after a prior text-only flush",
    )
    push_gemini.add_argument(
        "--batch",
        action="store_true",
        help="Send all experiences as one multi-task session for EverOS boundary detection",
    )
    push_gemini.set_defaults(func=cmd_push_gemini_everos)
    return parser


def main(argv: list[str] | None = None) -> int:
    if os.name == "nt" and hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="backslashreplace")
    args = build_parser().parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
