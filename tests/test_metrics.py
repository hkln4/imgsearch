import pytest

from metrics import overlap_at_k, percentile, precision_at_k, r_precision


def test_precision_at_k_counts_relevant_in_top_k():
    retrieved = [1, 2, 3, 4, 5]
    relevant = {1, 3, 9}
    assert precision_at_k(retrieved, relevant, 1) == 1.0
    assert precision_at_k(retrieved, relevant, 2) == 0.5
    assert precision_at_k(retrieved, relevant, 5) == 0.4


def test_precision_at_k_short_list_counts_missing_as_misses():
    # fewer results than k must not inflate precision
    assert precision_at_k([1], {1, 2}, 4) == 0.25


def test_precision_at_k_rejects_non_positive_k():
    with pytest.raises(ValueError):
        precision_at_k([1], {1}, 0)


def test_r_precision_uses_number_of_relevant_items():
    relevant = {1, 2, 3}
    assert r_precision([1, 2, 3, 4], relevant) == 1.0
    assert r_precision([1, 4, 2, 3], relevant) == pytest.approx(2 / 3)


def test_r_precision_rejects_empty_relevant_set():
    with pytest.raises(ValueError):
        r_precision([1, 2], set())


def test_overlap_at_k_ignores_order_within_top_k():
    exact = [1, 2, 3, 4]
    assert overlap_at_k([3, 2, 1, 9], exact, 3) == 1.0
    assert overlap_at_k([1, 9, 8, 2], exact, 2) == 0.5


def test_percentile_matches_linear_interpolation():
    values = [10, 20, 30, 40, 50]
    assert percentile(values, 50) == 30
    assert percentile(values, 0) == 10
    assert percentile(values, 100) == 50
    assert percentile(values, 95) == pytest.approx(48.0)
