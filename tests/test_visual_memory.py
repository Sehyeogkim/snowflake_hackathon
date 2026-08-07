from mavis.fingerprints import VideoFingerprint
from mavis.visual_memory import VisualMemoryEntry, VisualMemoryIndex


def test_visual_index_ranks_matching_scene_signature_first() -> None:
    index = VisualMemoryIndex(
        [
            VisualMemoryEntry("a" * 24, "clip-a", "safe_walkway", 4.0, (1, 2, 3)),
            VisualMemoryEntry(
                "b" * 24,
                "clip-b",
                "safe_walkway_violation",
                4.0,
                (2**63, 2**62, 2**61),
            ),
        ]
    )
    query = VideoFingerprint("query", "", "dev", 4.0, (1, 2, 3))
    references = index.references(query, top_k=2)
    assert references[0][1] == "a" * 24
    assert references[0][2] > references[1][2]


def test_visual_index_filters_before_top_k_for_memory_ablation() -> None:
    index = VisualMemoryIndex(
        [
            VisualMemoryEntry("a" * 24, "clip-a", "safe_walkway", 4.0, (1, 2, 3)),
            VisualMemoryEntry("b" * 24, "clip-b", "safe_walkway", 4.0, (4, 5, 6)),
        ]
    )
    query = VideoFingerprint("query", "", "dev", 4.0, (1, 2, 3))
    references = index.references(query, top_k=1, allowed_experience_ids={"b" * 24})
    assert references[0][1] == "b" * 24
