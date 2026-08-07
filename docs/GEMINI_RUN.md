# Gemini dense-teacher run — 2026-08-07

## Outcome

The frozen 300-clip seed split was analyzed from real video at 10 fps. No synthetic
training clips or generated frames were used. The run stopped at an estimated public list
price of **$3.977947**, leaving **$0.022053** under the hard $4 cap.

| Stage | Clips | Input tokens | Output/thinking tokens | Estimated USD |
|---|---:|---:|---:|---:|
| Gemini 3.1 Flash-Lite, low-resolution dense 10 fps | 300 | 1,732,716 | 141,601 | $0.645581 |
| Gemini 3.1 Pro full + temporal-occlusion calls | 65 calls | 1,496,182 | 28,248 | $3.331340 |
| Credential/10 fps smoke | 1 | 2,804 | 217 | $0.001027 |
| **Total** |  |  |  | **$3.977947** |

The Pro total includes two full calls that could not be paired, one conservatively charged
interrupted reservation, and all completed teacher/occlusion calls. Every unknown in-flight
amount was charged at its reserved ceiling rather than assumed free.

## Ground-truth results

| Stage | N | Accuracy | Macro-F1 | Unsafe recall |
|---|---:|---:|---:|---:|
| Flash-Lite dense | 300 | 16.67% | 15.03% | 0.00% (0/224) |
| Pro full, safety-first selected subset | 33 | 39.39% | 18.01% | 60.61% (20/33) |
| Pro with proposed window occluded | 32 | 40.63% | 18.75% | 56.25% (18/32) |

Flash-Lite never predicted an unsafe class on this dataset, so it cannot be used as the
final safety classifier. Its useful role is inexpensive hard-case discovery. Because the
objective prioritizes unsafe false negatives before class balancing, all 33 Pro selections
were unsafe examples: 10 walkway violations, 9 opened-panel cases, 8 forklift-overload
cases, and 5 unauthorized interventions among the 32 completed pairs.

## Experience validation

All 300 dense observations were exported as GT-evaluated Experiences. A wrong model answer
is never stored as a reusable answer; it becomes an explicit reject/escalate policy.

- 224 unsafe examples became unsafe-false-negative escalation rules. Thirty-two of these
  also received the Pro counterfactual audit.
- 26 other classification errors became abstain/escalate rules.
- 50 correct cheap observations became cheap-path candidates that still require the stated
  ground-truth guardrails.
- 32 Experiences include paired Pro full/occluded measurements.

Within the 32 paired counterfactual audits:

- 13 proposed time windows were validated by a ground-truth log-loss increase, prediction
  flip, or induced unsafe failure.
- 8 occlusions changed the predicted class.
- 5 occlusions induced an unsafe-to-safe failure.
- 19 windows were explicitly stored as rejected hypotheses. These are negative experiences
  and must not be reused as shortcuts.

Each Experience contains the available measured actions, ground-truth label, selected
routing policy, outcome, and safety guardrail. Paired records also contain the critical
interval and full/occluded GT probabilities. Raw video and API keys are never included.

## EverOS delivery status

EverOS Cloud now contains 300/300 verified Experience episodes under app `mavis`, project
`factory-safety-seed-v1`, and owner `mavis-hackathon`. Every Experience uses a deterministic
independent session prefixed `mavis-memory-v3`. Verification matched the requested
`session_id` exactly and found 300 distinct episode IDs; there were zero failed or
flushed-but-unverified records.

Retrieval was tested against three policy families. Hybrid search returned the unsafe
false-negative escalation lessons, while exact keyword search returned the stored
counterfactual fields (`counterfactual_tested`, `critical_window`, and
`window_validated`). Cheap-strategy lessons were independently retrievable as well.

The earlier experimental `mode="agent"` trajectories were accepted and flushed, but the
Cloud extractor produced zero `agent_case` records. They are not counted as indexed memory.
The canonical verified record is the searchable EverOS episode; Agent Case/Skill promotion
is an optional server-derived layer rather than a delivery dependency.

## Artifacts

- `data/manifest.jsonl` — frozen seed/dev/test split
- `data/work/gemini/dense10/` — 300 real-frame 10 fps proxies
- `outputs/gemini_teacher.sqlite3` — resumable runs, budget ledger, and exact delivery status
- `outputs/gemini_experiences.jsonl` — 300 portable GT-evaluated Experiences
