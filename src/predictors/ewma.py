"""EWMA queue-hotspot risk predictor."""
from __future__ import annotations

import numpy as np

from .base import BasePredictor


class EWMAPredictor(BasePredictor):
    def __init__(self, num_queues: int, window: int, alpha: float = 0.3) -> None:
        super().__init__(num_queues, window)
        self.alpha = float(alpha)
        self.ewma_feat: np.ndarray | None = None

    def observe(self, feat: np.ndarray) -> None:
        super().observe(feat)
        if self.ewma_feat is None:
            self.ewma_feat = feat.copy().astype(np.float64)
            return
        self.ewma_feat = self.alpha * feat + (1.0 - self.alpha) * self.ewma_feat

    def predict(self) -> np.ndarray:
        if self.ewma_feat is None:
            return np.zeros(self.num_queues, dtype=np.float64)
        P = self.ewma_feat[:, 3]
        R_peak = self.ewma_feat[:, 1]
        ewma_B = self.ewma_feat[:, 0]
        cur_B = self._history[-1][:, 0] if len(self._history) > 0 else ewma_B
        trend = (cur_B - ewma_B) / np.maximum(1.0, ewma_B)
        R_norm = np.clip(R_peak / 100e9, 0.0, 1.0)
        x = 0.6 * P + 0.25 * np.clip(trend, -1.0, 1.0) + 0.15 * R_norm
        return np.clip(x, 0.0, 1.0)
