# MAVIS

**Memory-Aware Visual Inference Scheduler** for factory CCTV safety video.

MAVIS builds reusable experiences by testing visual-inference strategies on labeled clips,
measuring which strategy is the least expensive safe action, and storing the evaluated
routing lesson as a searchable EverOS episode. Agent Case extraction remains an optional
derived track; the verified episode is the durable cloud record.

The benchmark claim is intentionally strict: only `CREDITS` from Snowflake's
`CORTEX_AI_FUNCTIONS_USAGE_HISTORY` counts as actual Cortex cost. Local mock values are
pipeline fixtures and must never appear in the final percentage-reduction claim.

Gemini is used only to build real-video teacher experiences before Snowflake is available.
It does not replace the final held-out Snowflake cost benchmark.

## Architecture

```text
Seed: real MP4 -> 10 fps Gemini teacher -> GT-evaluated Experience
               -> EverOS episode + cached OpenCV scene signature

Runtime: new MP4 -> event gate -> cheap Snowflake observation
                 -> EverOS text retrieval + visual-signature retrieval
                 -> safety-gated MAVIS policy
                 -> ACCEPT_CHEAP / STRONG_SINGLE / STRONG_MULTI
                 -> prediction + query IDs + billed Cortex AI credits
```

The implementation is resumable through SQLite, uses a bounded worker pool, scans each
video at a configurable low FPS, and only decodes selected frames at full resolution.

## Quick start without API keys

PowerShell:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e '.[dev]'
.\.venv\Scripts\mavis.exe doctor

.\.venv\Scripts\mavis.exe inspect-data --root data\raw
.\.venv\Scripts\mavis.exe prepare-manifest --root data\raw --output data\manifest.jsonl
.\.venv\Scripts\mavis.exe extract-candidates --manifest data\manifest.jsonl --split seed --workers 8
.\.venv\Scripts\mavis.exe plan-jobs --manifest data\manifest.jsonl --split seed
.\.venv\Scripts\mavis.exe generate-seeds --backend mock --limit 3
.\.venv\Scripts\python.exe -m pytest
```

`mock` is deterministic and exists only to validate extraction, concurrency, resume,
schemas, and EverOS payload generation. It is not evidence of model quality or cost.

## Interactive demo

The `demo/` app starts empty, accepts a local CCTV video, extracts 16 real observations in
the browser, and selects three critical frames from measured pixel-motion peaks across the
beginning, middle, and end of the clip.

```powershell
cd demo
npm install
npm run dev
```

Choose `demo/public/demo-video.mp4` to reproduce the checked-in held-out walkway result.
The UI verifies both its filename and duration before displaying the saved VLM class and
confidence. Any other upload still receives real frame extraction and evidence selection,
but no safety class is fabricated without a live VLM credential.

## Checked-in evidence

- [`results/memory-build-summary.json`](results/memory-build-summary.json): 300 evaluated
  seed Experiences, 300 scene signatures, 32 counterfactual audits, and a duplicate-safe
  57-development / 300-test split.
- [`results/heldout-demo-analysis.json`](results/heldout-demo-analysis.json): one real
  held-out clip whose saved VLM prediction matches dataset ground truth.
- [`results/provider-pilot-8.json`](results/provider-pilot-8.json): an eight-clip provider
  pilot reduced measured tokens from 71,254 to 10,063 (85.877%), but failed the quality
  gate. This is useful engineering evidence, not the final Snowflake benchmark claim.

## Dense Gemini teacher seeding

The reusable teacher path keeps every sampled frame real: OpenCV converts each seed clip
to an audio-free 10 fps MP4 proxy, Gemini 3.1 Flash-Lite analyzes all 300 seed clips, and
Gemini 3.1 Pro handles safety-first hard cases. Each Pro teacher call is paired with a
second Pro call in which the proposed critical time window is replaced by neutral frames.
The ground-truth probability change validates or rejects the proposed window.

The number of Pro clips is not fixed. Before each call or pair, the SQLite budget ledger
reserves a conservative maximum based on video frame count and maximum output tokens. It
continues through shorter candidates when a longer one no longer fits and stops before the
configured public-list-price cap.

```powershell
.\.venv\Scripts\mavis.exe --env .env train-gemini `
  --manifest data\manifest.jsonl --split seed --limit 300 --fps 10 `
  --budget-usd 4 --workers 4 --proxy-workers 8

.\.venv\Scripts\mavis.exe --env .env push-gemini-everos `
  --input outputs\gemini_experiences.jsonl `
  --database outputs\gemini_teacher.sqlite3 `
  --session-id mavis-memory-v3 --workers 8 --verify-wait 1
```

`outputs/gemini_teacher.sqlite3` is the resumable execution/cost ledger;
`outputs/gemini_experiences.jsonl` is the portable long-term-memory payload. EverOS is not
SQL storage: the CLI sends each GT-evaluated lesson with `add(mode="chat")`, calls `flush`,
then verifies the returned episode by exact `session_id`. The local SQLite ledger only
records resumable execution, cost, and delivery status. The completed run produced 300
portable Experiences and 300 distinct verified EverOS episodes.

The completed August 7, 2026 run is documented in
[`docs/GEMINI_RUN.md`](docs/GEMINI_RUN.md). The `$4` value is an estimate from public model
rates and returned token usage, not a provider invoice.

The cheap teacher's dominant error was a confidently safe description of an actually unsafe
clip. Runtime therefore never treats confidence alone as permission to accept a safe result.
Similar GT false negatives force temporal escalation; verified safe memories can authorize
cheap acceptance; ambiguous safe results receive at least one strong-keyframe confirmation.

## Credentials

Copy `.env.example` to `.env` and fill it locally. Do not commit `.env`.

Snowflake does **not** require a separate Cortex/model-provider API key. It needs a
Snowflake account identifier, user authentication, role, warehouse, database, schema,
and image stage. Password auth is fastest for a workshop; key-pair/OAuth is better for a
deployed service. Run [`snowflake/01_setup.sql`](snowflake/01_setup.sql) once with an
administrator and grant the returned model application roles.

A paid account is not required for the hackathon. The official rules provide participants a
trial account with $400 in credit, and Snowflake trial documentation says an account without a
valid payment method can use roughly ten Cortex AI credits per day. Use the hackathon site's
dedicated sign-up link first; the general fallback is the
[30-day Snowflake trial](https://www.snowflake.com/en/snowflake-trial/).

EverOS Cloud needs `EVEROS_API_KEY`; use `EVER_OS_BASE_URL=https://api.evermind.ai`.
MAVIS uses the official `everos-cloud` SDK. Each clip gets a separate deterministic session,
the canonical Experience is retrieved with `get(episode, user_id=...)`, and the default
hybrid search uses `--scope user`. The optional `--batch` path submits Agent Memory
trajectories for server-side Case/Skill derivation, but a successful HTTP flush alone is
not counted as indexed memory.

### Optional Cortex Code CLI

The Snowflake-hosted command shared with this project installs **Cortex Code CLI**, a
developer assistant, not the Python connector used by the runtime. The installer targets
`%LOCALAPPDATA%\cortex`, adds a stable user-PATH shim, downloads releases from
`sfc-repo.snowflakecomputing.com`, and verifies the package SHA-256 from its manifest.
It can help with interactive Snowflake work but is not required to run MAVIS. Review the
[installer](https://ai.snowflake.com/static/cc-scripts/install.ps1) before executing the
usual `irm ... | iex` form; the stable release observed during setup was `1.1.60`.

Build the 300-entry visual-signature cache and validate the full scheduler without spending
Snowflake credits:

```powershell
.\.venv\Scripts\mavis.exe build-visual-memory `
  --manifest data\manifest_final.jsonl `
  --experiences outputs\gemini_experiences.jsonl `
  --output outputs\visual_memory.json

.\.venv\Scripts\mavis.exe benchmark `
  --manifest data\manifest_final.jsonl --split dev --backend mock --limit 3 `
  --memory-source hybrid --memory-limit 300 --run-id local-smoke-v1
```

The real Snowflake sequence is:

```powershell
.\.venv\Scripts\python.exe -m pip install -e '.[snowflake,dev]'
.\.venv\Scripts\mavis.exe doctor
.\.venv\Scripts\mavis.exe benchmark `
  --manifest data\manifest_final.jsonl --split dev --backend snowflake --limit 3 `
  --memory-source hybrid --run-id sf-smoke-v1 --output outputs\sf_smoke.jsonl

.\.venv\Scripts\mavis.exe reconcile-benchmark `
  --input outputs\sf_smoke.jsonl --wait-seconds 300
```

Use the three-record run as a credential/model smoke test. Then tune thresholds on all 57
development clips. Freeze the configuration before running the 300 held-out test clips once.

## Data split

The final manifest is `data/manifest_final.jsonl`. It contains 334 seed-group clips, 57
development clips, and 300 held-out test clips. The 34 extra seed clips are exact visual
duplicates connected to the original memory set; the actual teacher/EverOS memory remains
exactly 300 real clips.
Recognized labels are:

- `safe_walkway` / `safe_walkway_violation`
- `authorized_intervention` / `unauthorized_intervention`
- `closed_panel_cover` / `opened_panel_cover`
- `safe_carrying` / `carrying_overload_with_forklift`

The final split merges all pairs with identical three-frame dHash and identical duration before
assignment, forces every group containing one of the 300 stored memories into seed, and keeps
the test target at 300. This reduced exact cross-split duplicate pairs from 84 to zero. Broader
near-duplicate candidates remain an explicit limitation because public camera/time metadata is
not available and aggressive pHash grouping would merge visually similar but distinct events.

## Actual cost reduction

`mavis benchmark` tags baseline and optimized sessions with a stable benchmark ID, runs both on the same
held-out clips, wait for the usage view (normally about two minutes, up to five), then run
[`snowflake/02_actual_cost_reduction.sql`](snowflake/02_actual_cost_reduction.sql).
Per-query tags are `mavis|baseline|RUN_ID|CLIP_ID|ACTION` and
`mavis|optimized|RUN_ID|CLIP_ID|ACTION`; the SQL aggregates by prefix.

Report unsafe recall and macro-F1 beside the cost result. The intended guardrail is:

```text
minimize Cortex AI credits
subject to unsafe recall >= baseline unsafe recall - 2 percentage points
```

Warehouse/platform credits are separate units from AI credits and must not be added as
raw numbers. If both are reported in dollars, convert each with its contracted unit price.

## Output

- `data/manifest.jsonl`: reproducible split and labels
- `data/manifest_final.jsonl`: duplicate-safe 334/57/300 benchmark split
- `data/work/candidates/`: only selected JPEG observations
- `outputs/mavis.sqlite3`: resumable source of truth, query IDs, responses, credits
- `outputs/experiences.jsonl`: portable structured trajectories
- `data/work/gemini/dense10/`: 300 real-frame, 10 fps Gemini proxies
- `outputs/gemini_teacher.sqlite3`: Gemini tokens, estimated USD, responses, reservations
- `outputs/gemini_experiences.jsonl`: 300 GT-evaluated routing Experiences, including 32
  paired temporal-counterfactual audits
- `outputs/visual_memory.json`: 300 cached real-video scene signatures
- `outputs/benchmark*.jsonl`: per-clip baseline/optimized decisions and query IDs
