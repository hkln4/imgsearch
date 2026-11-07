import math


def precision_at_k(retrieved, relevant, k):
    """Fraction of the top-k retrieved ids that are relevant."""
    if k <= 0:
        raise ValueError("k must be positive")
    return sum(1 for r in retrieved[:k] if r in relevant) / k


def r_precision(retrieved, relevant):
    """Precision at R, where R is the number of relevant items.

    At R, precision equals recall, so a perfect ranking scores 1.0
    regardless of how many relevant items a class has.
    """
    if not relevant:
        raise ValueError("relevant must not be empty")
    return precision_at_k(retrieved, relevant, len(relevant))


def overlap_at_k(approx, exact, k):
    """ANN recall: fraction of the exact top-k that the approximate search also returned."""
    if k <= 0:
        raise ValueError("k must be positive")
    return len(set(approx[:k]) & set(exact[:k])) / k


def percentile(values, p):
    """p-th percentile with linear interpolation (same as numpy's default)."""
    if not values:
        raise ValueError("values must not be empty")
    s = sorted(values)
    pos = (len(s) - 1) * p / 100
    lo, hi = math.floor(pos), math.ceil(pos)
    return s[lo] + (s[hi] - s[lo]) * (pos - lo)
