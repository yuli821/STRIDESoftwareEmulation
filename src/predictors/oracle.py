"""Oracle predictor (peeks at ground-truth next-epoch P_q); upper bound."""
from __future__ import annotations

import numpy as np

from .base import BasePredictor


class OraclePredictor(BasePredictor):
    def __init__(self, num_queues: int, window: int) -> None:
        super().__init__(num_queues, window)
        self._next_P: np.ndarray | None = None

    def set_truth(self, next_P: np.ndarray) -> None:
        self._next_P = next_P

    def predict(self) -> np.ndarray:
        if self._next_P is None:
            return np.zeros(self.num_queues, dtype=np.float64)
        return np.clip(self._next_P, 0.0, 1.0)
