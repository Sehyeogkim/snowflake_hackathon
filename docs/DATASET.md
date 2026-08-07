# Dataset record

MAVIS uses **Video Dataset for Safe and Unsafe Behaviours**, Mendeley Data v1,
DOI `10.17632/xjmtb22pff.1`.

- Official record: <https://data.mendeley.com/datasets/xjmtb22pff/1>
- Stable ZIP endpoint: <https://data.mendeley.com/public-api/zip/xjmtb22pff/download/1>
- Data paper: <https://pmc.ncbi.nlm.nih.gov/articles/PMC11367630/>
- Dataset license: CC BY 4.0 (the paper text has a separate CC BY-NC 4.0 license)

## Verified local copy

`data/raw/safe_unsafe_behaviours/Safe and Unsafe Behaviours Dataset`

- 691 H.264 MP4 clips, 1920×1080, 24 FPS, roughly 1–20 seconds
- 9,999,951,582 bytes (9.313 GiB)
- All 691 files matched the official per-file SHA-256 metadata
- Train: 566 clips / 8,045,758,362 bytes
- Test: 125 clips / 1,954,193,220 bytes

| ID | Primary clip label | Safety | Train | Test | Total |
|---:|---|---|---:|---:|---:|
| 0 | Safe Walkway Violation | Unsafe | 178 | 32 | 210 |
| 1 | Unauthorized Intervention | Unsafe | 97 | 11 | 108 |
| 2 | Opened Panel Cover | Unsafe | 129 | 13 | 142 |
| 3 | Carrying Overload with Forklift | Unsafe | 48 | 8 | 56 |
| 4 | Safe Walkway | Safe | 50 | 25 | 75 |
| 5 | Authorized Intervention | Safe | 23 | 15 | 38 |
| 6 | Closed Panel Cover | Safe | 19 | 13 | 32 |
| 7 | Safe Carrying | Safe | 22 | 8 | 30 |

There is no CSV, bounding box, frame-level label, camera ID, or timestamp metadata.
Folder names are clip-level primary labels. The paper notes that some clips can contain
more than one behavior, so MAVIS does not describe the directory label as an exclusive
ground truth for every frame.

## Frozen MAVIS benchmark split

[`data/manifest_final.jsonl`](../data/manifest_final.jsonl) is deterministic
(`random_seed=42`) and contains 334 seed-group clips, 57 development clips, and 300
held-out test clips. Exactly 300 seed clips have Gemini/EverOS Experiences. The other 34
are exact-duplicate group members forced to seed so no stored-memory group can cross into
development or test. Paths are relative to the manifest for portability.

The split was rebuilt from `outputs/leakage_candidates.json`. Pairs whose three sampled dHashes
and durations were exactly equal were unioned before assignment. Exact cross-split matches fell
from 84 to zero, all 300 stored Experiences remain seed-only, and test remains 300 clips.
Broader pHash candidates are not automatically merged because public camera/time metadata is
absent and visually similar factory scenes are not necessarily duplicate recordings.

Runtime candidates are three chronological observations with a minimum temporal gap around
the event peak; adjacent nearly identical 2 Hz frames are no longer selected. The 300 memory
clips also have cached 20/50/80-percent dHash signatures in `outputs/visual_memory.json` for
auditable visual-memory retrieval.
