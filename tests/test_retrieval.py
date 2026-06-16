from src.retrieval import reciprocal_rank_fusion


def test_rrf_merges_rankings():
    dense = [(0, 0.9), (1, 0.8), (2, 0.7)]
    sparse = [(1, 5.0), (3, 4.0), (0, 3.0)]
    fused = reciprocal_rank_fusion([dense, sparse], k=60)
    ids = [idx for idx, _ in fused]
    assert 0 in ids
    assert 1 in ids
    assert fused[0][1] >= fused[-1][1]
