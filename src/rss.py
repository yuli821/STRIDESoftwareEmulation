"""RSS indirection table (bucket -> queue), one per domain."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np


# Default skew weights used by ``init="skewed"``. Two "hot" queues hold
# the majority of the bucket-to-queue mapping, the rest of the queues
# hold a small (but non-zero) minority. Motivated by adversarial /
# pathological RSS hash behaviour documented in Woo et al. (RSS++, NSDI
# 2019) and Liu et al. (SIGCOMM 2022), where Toeplitz hash collisions
# and skewed 5-tuple distributions cause a small number of RSS queues
# to absorb a disproportionate share of flows. This is precisely the
# regime where an adaptive scheduler should be able to outperform a
# static modulo-initialised mapping, since the scheduler can migrate
# buckets off the hot queues to restore balance.
#
# Weights: Q0 50%, Q1 30%, Q2 10%, Q3 10%, remainder split evenly.
# For num_queues != 4 the weights are generalised in _skewed_weights.
_SKEWED_DEFAULT_HEAD: tuple[float, ...] = (0.50, 0.30)


def _skewed_weights(num_queues: int) -> np.ndarray:
    """Return a length-``num_queues`` non-negative weight vector that
    sums to 1.0, with the two "hot" queues carrying 0.5 and 0.3 of the
    mass and the remaining queues sharing 0.2 evenly.

    Degenerate cases:

    * ``num_queues == 1``: weights = [1.0] (nothing to skew).
    * ``num_queues == 2``: weights = [0.7, 0.3] (keep a clear hot/cold
      split instead of collapsing to 50/50).
    """
    if num_queues <= 0:
        raise ValueError("num_queues must be positive")
    if num_queues == 1:
        return np.array([1.0], dtype=np.float64)
    if num_queues == 2:
        return np.array([0.70, 0.30], dtype=np.float64)
    w = np.zeros(num_queues, dtype=np.float64)
    w[0], w[1] = _SKEWED_DEFAULT_HEAD
    tail_share = 1.0 - (w[0] + w[1])
    w[2:] = tail_share / (num_queues - 2)
    return w


def _skewed_table(num_buckets: int, num_queues: int) -> np.ndarray:
    """Build a bucket->queue table whose queue-occupancy histogram
    matches ``_skewed_weights(num_queues) * num_buckets`` (rounded to
    integers) and whose buckets are placed as contiguous runs per
    queue. Layout: [Q0 * c0, Q1 * c1, Q2 * c2, ...].
    """
    weights = _skewed_weights(num_queues)
    counts = np.round(weights * num_buckets).astype(int)
    # Fix rounding so counts.sum() == num_buckets exactly.
    diff = int(num_buckets - counts.sum())
    if diff > 0:
        # Dump extras into the hottest (largest count) queue.
        counts[int(np.argmax(counts))] += diff
    elif diff < 0:
        # Steal from hottest queue until balanced.
        for _ in range(-diff):
            counts[int(np.argmax(counts))] -= 1
    if counts.min() < 0:
        raise ValueError(
            f"skewed RSS: derived non-positive count {counts}")
    table = np.concatenate([
        np.full(int(counts[q]), q, dtype=np.int64)
        for q in range(num_queues)
    ])
    assert table.size == num_buckets, (
        f"skewed RSS table size {table.size} != num_buckets {num_buckets}")
    return table


@dataclass
class RSSIndirectionTable:
    num_buckets: int
    num_queues: int
    init: str = "modulo"
    rng: Optional[np.random.Generator] = None

    def __post_init__(self) -> None:
        if self.init == "modulo":
            self.table = (np.arange(self.num_buckets) % self.num_queues).astype(np.int64)
        elif self.init == "random":
            rng = self.rng or np.random.default_rng()
            self.table = rng.integers(0, self.num_queues, size=self.num_buckets, dtype=np.int64)
        elif self.init == "skewed":
            # Deterministic: two hot queues absorb the majority of
            # buckets; remaining queues share 20 percent evenly. See
            # ``_skewed_weights`` for the parameterisation and
            # justification.
            self.table = _skewed_table(self.num_buckets, self.num_queues)
        else:
            raise ValueError(f"unknown RSS init: {self.init}")

    def lookup(self, bucket_ids: np.ndarray) -> np.ndarray:
        return self.table[bucket_ids]

    def set(self, bucket: int, queue: int) -> None:
        self.table[bucket] = queue

    def snapshot(self) -> np.ndarray:
        return self.table.copy()
