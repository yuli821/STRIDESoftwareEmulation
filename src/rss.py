"""RSS indirection table (bucket -> queue), one per domain."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np


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
        else:
            raise ValueError(f"unknown RSS init: {self.init}")

    def lookup(self, bucket_ids: np.ndarray) -> np.ndarray:
        return self.table[bucket_ids]

    def set(self, bucket: int, queue: int) -> None:
        self.table[bucket] = queue

    def snapshot(self) -> np.ndarray:
        return self.table.copy()
