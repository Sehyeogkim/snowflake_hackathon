"""Command line entry points.

    mavis data                     inventory of downloaded clips
    mavis demo <clip>              run one clip with the decision overlay
    mavis bench                    baseline vs MAVIS over the eval split
    mavis smoke                    verify a live Snowflake connection

Every command defaults to ``--cortex mock`` so the pipeline runs on a bare
checkout. Nothing reported from a mock run is evidence of anything; ``bench``
prints a loud banner saying so.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from . import benchmark, cortex as cortex_mod, dataset, env, memory as memory_mod
from .config import DEFAULT, Config
from .metrics import render
from .overlay import OverlayWriter, annotate, console_line
from .runner import run_baseline, run_mavis

DEFAULT_CLIPS = "data/clips"


def _add_common(ap: argparse.ArgumentParser) -> None:
    ap.add_argument("--clips", default=DEFAULT_CLIPS, help="root of the labelled clip tree")
    ap.add_argument(
        "--cortex",
        default="mock",
        choices=list(cortex_mod.KINDS),
        help="inference backend (default: mock, which fabricates costs)",
    )
    ap.add_argument(
        "--memory",
        default="local",
        choices=["everos", "local", "none"],
        help="memory store; 'none' is the cold-start ablation",
    )
    ap.add_argument("--seed", type=int, default=DEFAULT.seed)


def _build(args) -> tuple[object, object, Config]:
    cfg = Config()
    cfg.seed = args.seed
    client = cortex_mod.build(args.cortex, seed=args.seed)
    store = memory_mod.build(args.memory)
    return client, store, cfg


# -- commands --------------------------------------------------------------


def cmd_data(args) -> int:
    clips = dataset.discover(args.clips)
    if not clips:
        print(
            f"no clips under {args.clips}.\n"
            "Run: uv run --with remotezip scripts/fetch_clips.py",
            file=sys.stderr,
        )
        return 1
    print(dataset.summarise(clips))
    return 0


def cmd_demo(args) -> int:
    clips = dataset.discover(args.clips)
    target = _resolve_clip(clips, args.clip)
    if target is None:
        print(f"clip not found: {args.clip}", file=sys.stderr)
        return 1

    client, store, cfg = _build(args)
    print(f"clip     : {target.clip_id}")
    print(f"truth    : {'HAZARD' if target.is_hazard else 'safe'} — {target.description}")
    print(f"cortex   : {client.name}{'  (SIMULATED COSTS)' if client.estimated else ''}")
    print(f"memory   : {store.name} ({len(store)} episodes)")
    print("-" * 70)

    spent = 0.0
    detected = False
    writer = OverlayWriter(args.out, show=args.show) if args.out else None

    def on_step(step, image):
        nonlocal spent, detected
        if step.cost:
            spent += step.cost.credits or step.cost.total_tokens / 1e6
        detected = detected or step.hazard_prob_after >= cfg.belief.detect_threshold
        print(console_line(step))
        if writer:
            writer.write(annotate(image, step, detected=detected, spent=spent))

    try:
        runner = run_baseline if args.policy == "baseline" else run_mavis
        trace = (
            run_baseline(target, client, cfg, on_step=on_step)
            if args.policy == "baseline"
            else run_mavis(target, client, store, cfg, learn=not args.no_learn, on_step=on_step)
        )
    finally:
        if writer:
            writer.close()
        client.close()

    print("-" * 70)
    verdict = "HAZARD DETECTED" if detected else "no hazard"
    print(
        f"{verdict}   peak p={trace.peak_hazard_prob():.2f}   "
        f"{trace.cortex_calls} cortex calls   {trace.total_tokens:,} tokens"
    )
    print(f"gate: {trace.frames_gated}/{trace.frames_decoded} frames passed")
    if writer:
        print(f"overlay written to {args.out}")
    return 0


def cmd_bench(args) -> int:
    eval_clips = dataset.discover(args.clips, split=args.eval_split)
    warm_clips = (
        dataset.discover(args.clips, split=args.warmup_split) if args.warmup_split else []
    )
    if args.limit:
        eval_clips = eval_clips[: args.limit]
    if args.warmup_limit:
        warm_clips = warm_clips[: args.warmup_limit]
    if not eval_clips:
        print(
            f"no clips in split {args.eval_split!r} under {args.clips}. "
            "The download may still be in flight — check data/logs/fetch_clips.log",
            file=sys.stderr,
        )
        return 1

    client, store, cfg = _build(args)
    print(
        f"eval: {len(eval_clips)} clips ({args.eval_split})   "
        f"warm-up: {len(warm_clips)} clips ({args.warmup_split or 'none'})   "
        f"cortex: {client.name}   memory: {store.name}"
    )
    try:
        result = benchmark.run(
            eval_clips,
            client,
            store,
            cfg,
            warmup_clips=warm_clips or None,
            baseline_gated=args.baseline_gated,
            shuffle_seed=args.seed,
        )
    finally:
        client.close()

    print()
    if result.calibrated_costs:
        rows = "  ".join(f"{k}={v:.7f}" for k, v in result.calibrated_costs.items() if v)
        print(f"measured action costs: {rows}")
        print(f"(calibration itself cost {result.calibration_cost:.6f}, excluded from both totals)\n")
    print(render(result.comparison))
    out = result.save(args.out)
    print(f"\nfull traces: {out}")
    return 0


def cmd_smoke(args) -> int:
    """Prove a live backend can actually serve image inference and report cost."""
    import cv2
    import numpy as np

    from .cortex.base import HAZARD_PROMPT
    from .dataset import CLASSIFY_CATEGORIES

    try:
        client = cortex_mod.build(args.cortex)
    except Exception as exc:  # noqa: BLE001 - surface the real cause verbatim
        print(f"cannot connect: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1

    # A flat grey rectangle tells us nothing about model quality, but it proves
    # the image path, the schema and the usage accounting all work.
    image = np.full((360, 640, 3), 90, dtype=np.uint8)
    cv2.rectangle(image, (80, 120), (260, 300), (60, 60, 200), -1)

    try:
        scene, c1 = client.classify(image, CLASSIFY_CATEGORIES)
        print(
            f"cheap  ok  model={c1.model} labels={scene.labels} risk={scene.risk:.2f} "
            f"tokens={c1.prompt_tokens}+{c1.completion_tokens} {c1.latency_s:.2f}s"
        )
        result, c2 = client.complete([image], HAZARD_PROMPT, strong=True)
        print(
            f"strong ok  model={c2.model} hazard_prob={result.hazard_prob:.2f} "
            f"tokens={c2.prompt_tokens}+{c2.completion_tokens} {c2.latency_s:.2f}s"
        )
        unit = getattr(client, "cost_unit", "credits")
        for label, c in (("cheap", c1), ("strong", c2)):
            shown = f"{c.credits:.8f} {unit}" if c.credits is not None else "pending reconciliation"
            print(f"  {label} cost: {shown}")
        if c2.total_tokens == 0:
            print(
                "warning: backend reported no token counts — cost tracking cannot "
                "work without them",
                file=sys.stderr,
            )
            return 1
        print(f"\nratio strong/cheap = {c2.credits / c1.credits:.1f}x" if c1.credits else "")
    except Exception as exc:  # noqa: BLE001
        print(f"inference failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    finally:
        client.close()
    return 0


def _resolve_clip(clips, wanted: str):
    path = Path(wanted)
    for clip in clips:
        if clip.clip_id == wanted or clip.path == path or clip.path.name == wanted:
            return clip
    return None


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="mavis", description="Memory-Aware Visual Inference Scheduler")
    sub = ap.add_subparsers(dest="command", required=True)

    p = sub.add_parser("data", help="inventory downloaded clips")
    p.add_argument("--clips", default=DEFAULT_CLIPS)
    p.set_defaults(func=cmd_data)

    p = sub.add_parser("demo", help="run one clip with the decision overlay")
    _add_common(p)
    p.add_argument("clip", help="clip id, path, or file name")
    p.add_argument("--policy", default="mavis", choices=["mavis", "baseline"])
    p.add_argument("--out", default="data/out/demo.mp4", help="overlay video path ('' to skip)")
    p.add_argument("--show", action="store_true", help="also open a live window (needs a display)")
    p.add_argument("--no-learn", action="store_true", help="do not write episodes to memory")
    p.set_defaults(func=cmd_demo)

    p = sub.add_parser("bench", help="baseline vs MAVIS")
    _add_common(p)
    p.add_argument("--eval-split", default="test")
    p.add_argument("--warmup-split", default="train", help="'' to skip warm-up")
    p.add_argument("--limit", type=int, default=0, help="cap eval clips (0 = all)")
    p.add_argument("--warmup-limit", type=int, default=0, help="cap warm-up clips (0 = all)")
    p.add_argument(
        "--baseline-gated",
        action="store_true",
        help="give the baseline the same OpenCV gate (harder, fairer comparison)",
    )
    p.add_argument("--out", default="data/out/benchmark.json")
    p.set_defaults(func=cmd_bench)

    p = sub.add_parser("smoke", help="verify a live backend's image inference")
    p.add_argument("--cortex", default="gemini", choices=list(cortex_mod.KINDS))
    p.set_defaults(func=cmd_smoke)

    # Credentials live in .env; existing environment variables still win.
    env.load()
    args = ap.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
