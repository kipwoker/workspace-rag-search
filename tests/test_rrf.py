"""Tests for Reciprocal Rank Fusion (RRF) implementation.

Usage:
    python -m tests.test_rrf
"""

import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from rrf import ReciprocalRankFusion


def test_rrf_basic_fusion():
    """Test basic RRF fusion with two ranked lists."""
    rrf = ReciprocalRankFusion(k=60)

    # Simulate semantic results (doc_id, score) - ranked by semantic similarity
    semantic_results = [
        ("doc_a", 0.95),  # Rank 1
        ("doc_b", 0.85),  # Rank 2
        ("doc_c", 0.75),  # Rank 3
        ("doc_d", 0.65),  # Rank 4
    ]

    # Simulate BM25 results - different ranking
    bm25_results = [
        ("doc_b", 0.92),  # Rank 1
        ("doc_e", 0.88),  # Rank 2  <- doc_e not in semantic results!
        ("doc_a", 0.80),  # Rank 3
        ("doc_f", 0.70),  # Rank 4  <- doc_f not in semantic results!
    ]

    # Fuse the rankings
    fused = rrf.fuse([semantic_results, bm25_results], limit=5)

    # Check results
    assert len(fused) == 5, f"Expected 5 results, got {len(fused)}"

    # doc_b should be first (appears in both lists with good ranks)
    assert fused[0][0] == "doc_b", f"Expected doc_b first, got {fused[0][0]}"

    # doc_a should be second (appears in both lists)
    assert fused[1][0] == "doc_a", f"Expected doc_a second, got {fused[1][0]}"

    # doc_e and doc_c should be in results (from different sources)
    doc_ids = [doc_id for doc_id, _ in fused]
    assert "doc_e" in doc_ids, "doc_e should be in results (BM25 only)"
    assert "doc_c" in doc_ids, "doc_c should be in results (semantic only)"

    print("✓ test_rrf_basic_fusion passed")


def test_rrf_boosts_common_documents():
    """Test that documents appearing in multiple lists get boosted."""
    rrf = ReciprocalRankFusion(k=60)

    # Both lists have same document at different ranks
    list1 = [("shared", 0.9), ("unique1", 0.8)]  # shared at rank 1
    list2 = [("unique2", 0.9), ("shared", 0.8)]  # shared at rank 2

    fused = rrf.fuse([list1, list2], limit=3)

    # shared should be first due to appearing in both lists
    assert fused[0][0] == "shared", f"Expected 'shared' first, got {fused[0][0]}"

    # Calculate expected scores manually
    # shared: 1/(60+1) + 1/(60+2) = 0.01639 + 0.01613 = 0.03252
    # unique1: 1/(60+2) = 0.01613
    # unique2: 1/(60+1) = 0.01639
    shared_score = 1 / 61 + 1 / 62
    assert abs(fused[0][1] - shared_score) < 0.0001, "shared score mismatch"

    print("✓ test_rrf_boosts_common_documents passed")


def test_rrf_single_list():
    """Test RRF with a single ranked list."""
    rrf = ReciprocalRankFusion(k=60)

    single_list = [
        ("doc_a", 0.9),
        ("doc_b", 0.8),
        ("doc_c", 0.7),
    ]

    fused = rrf.fuse([single_list], limit=2)

    assert len(fused) == 2, f"Expected 2 results, got {len(fused)}"
    assert fused[0][0] == "doc_a", f"Expected doc_a first, got {fused[0][0]}"
    assert fused[1][0] == "doc_b", f"Expected doc_b second, got {fused[1][0]}"

    print("✓ test_rrf_single_list passed")


def test_rrf_empty_lists():
    """Test RRF with empty lists."""
    rrf = ReciprocalRankFusion(k=60)

    # Empty lists
    fused = rrf.fuse([[], []], limit=5)
    assert len(fused) == 0, f"Expected 0 results for empty lists, got {len(fused)}"

    # One empty, one with results
    fused = rrf.fuse([[], [("doc_a", 0.9)]], limit=5)
    assert len(fused) == 1, f"Expected 1 result, got {len(fused)}"
    assert fused[0][0] == "doc_a", f"Expected doc_a, got {fused[0][0]}"

    print("✓ test_rrf_empty_lists passed")


def test_rrf_different_k_values():
    """Test RRF with different k values."""
    # Higher k should make ranks less impactful
    rrf_low_k = ReciprocalRankFusion(k=10)
    rrf_high_k = ReciprocalRankFusion(k=100)

    list1 = [("doc_a", 0.9)]  # Rank 1
    list2 = [("doc_b", 0.9)]  # Rank 1

    fused_low = rrf_low_k.fuse([list1, list2], limit=2)
    fused_high = rrf_high_k.fuse([list1, list2], limit=2)

    # Both should have same documents
    assert [doc_id for doc_id, _ in fused_low] == [doc_id for doc_id, _ in fused_high]

    # Scores should be higher with lower k (1/(10+1) > 1/(100+1))
    assert fused_low[0][1] > fused_high[0][1], "Lower k should produce higher scores"

    print("✓ test_rrf_different_k_values passed")


def test_rrf_three_lists():
    """Test RRF fusion with three ranked lists."""
    rrf = ReciprocalRankFusion(k=60)

    list1 = [("doc_a", 0.9), ("doc_b", 0.8)]  # doc_a at rank 1
    list2 = [("doc_b", 0.9), ("doc_a", 0.8)]  # doc_b at rank 1
    list3 = [("doc_c", 0.9), ("doc_a", 0.8)]  # doc_c at rank 1, doc_a at rank 2

    fused = rrf.fuse([list1, list2, list3], limit=3)

    # doc_a appears in all 3 lists, should be first
    assert fused[0][0] == "doc_a", f"Expected doc_a first, got {fused[0][0]}"

    # Check doc_a score: 1/61 + 1/62 + 1/62 = ~0.0486
    expected_doc_a_score = 1/61 + 1/62 + 1/62
    doc_a_score = next(score for doc_id, score in fused if doc_id == "doc_a")
    assert abs(doc_a_score - expected_doc_a_score) < 0.0001, "doc_a score mismatch"

    print("✓ test_rrf_three_lists passed")


def test_rrf_limit():
    """Test that limit parameter works correctly."""
    rrf = ReciprocalRankFusion(k=60)

    list1 = [(f"doc_{i}", 0.9 - i * 0.01) for i in range(10)]

    # Test different limits
    for limit in [1, 3, 5, 10]:
        fused = rrf.fuse([list1], limit=limit)
        assert len(fused) == limit, f"Expected {limit} results, got {len(fused)}"

    # Limit larger than available results
    fused = rrf.fuse([list1], limit=20)
    assert len(fused) == 10, f"Expected 10 results (all), got {len(fused)}"

    print("✓ test_rrf_limit passed")


def print_test_summary():
    """Print a summary of RRF test results."""
    print("\n" + "=" * 60)
    print("RRF Fusion Test Summary")
    print("=" * 60)
    print()
    print("Example RRF fusion result:")
    print("-" * 60)

    rrf = ReciprocalRankFusion(k=60)

    semantic_results = [
        ("doc_a", 0.95),
        ("doc_b", 0.85),
        ("doc_c", 0.75),
        ("doc_d", 0.65),
    ]

    bm25_results = [
        ("doc_b", 0.92),
        ("doc_e", 0.88),
        ("doc_a", 0.80),
        ("doc_f", 0.70),
    ]

    fused = rrf.fuse([semantic_results, bm25_results], limit=5)

    print(f"{'Rank':<6} {'Doc ID':<10} {'RRF Score':<12} {'Source'}")
    print("-" * 50)

    semantic_ids = {d for d, _ in semantic_results}
    bm25_ids = {d for d, _ in bm25_results}

    for rank, (doc_id, score) in enumerate(fused, 1):
        in_semantic = doc_id in semantic_ids
        in_bm25 = doc_id in bm25_ids

        if in_semantic and in_bm25:
            source = "both (boosted!)"
        elif in_semantic:
            source = "semantic only"
        else:
            source = "BM25 only"

        print(f"{rank:<6} {doc_id:<10} {score:<12.4f} {source}")

    print()
    print("Key observations:")
    print("  - doc_b: rank #2 semantic, rank #1 BM25 -> highest RRF score")
    print("  - doc_a: appears in both lists -> second highest")
    print("  - doc_e, doc_f: appear in only one list -> lower scores")
    print("  - Documents in both lists get boosted (sum of contributions)")
    print()


def main():
    """Run all RRF tests."""
    print("\n" + "█" * 60)
    print("█" + " " * 58 + "█")
    print("█" + "  RECIPROCAL RANK FUSION (RRF) TESTS".center(58) + "█")
    print("█" + " " * 58 + "█")
    print("█" * 60)

    tests = [
        test_rrf_basic_fusion,
        test_rrf_boosts_common_documents,
        test_rrf_single_list,
        test_rrf_empty_lists,
        test_rrf_different_k_values,
        test_rrf_three_lists,
        test_rrf_limit,
    ]

    passed = 0
    failed = 0

    for test in tests:
        try:
            test()
            passed += 1
        except AssertionError as e:
            print(f"✗ {test.__name__} failed: {e}")
            failed += 1
        except Exception as e:
            print(f"✗ {test.__name__} error: {e}")
            failed += 1

    print_test_summary()

    print("=" * 60)
    print(f"Results: {passed} passed, {failed} failed")
    print("=" * 60)

    if failed == 0:
        print("\n✅ All RRF tests passed!")
    else:
        print(f"\n❌ {failed} test(s) failed")
        sys.exit(1)


if __name__ == "__main__":
    main()
