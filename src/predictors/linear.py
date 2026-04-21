"""Least-squares linear extrapolation of P_q."""
from __future__ import annotations

import numpy as np

from .base import BasePredictor


class LinearPredictor(BasePredictor):
    def __init__(self, num_queues: int, window: int, lookback: int = 4) -> None:
        super().__init__(num_queues, window)
        self.lookback = min(lookback, window)

    def predict(self) -> np.ndarray:
        if len(self._history) == 0:
            return np.zeros(self.num_queues, dtype=np.float64)
        hist = np.stack(list(self._history)[-self.lookback:], axis=0)
        P = hist[:, :, 3]
        T = P.shape[0]
        if T < 2:
            return np.clip(P[-1], 0.0, 1.0)
        t = np.arange(T, dtype=np.float64)
        t_bar = t.mean()
        P_bar = P.mean(axis=0)
        num = ((t[:, None] - t_bar) * (P - P_bar)).sum(axis=0)
        den = ((t - t_bar) ** 2).sum()
        slope = num / max(den, 1e-9)
        intercept = P_bar - slope * t_bar
        return np.clip(slope * T + intercept, 0.0, 1.0)
