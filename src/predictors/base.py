"""Base class for queue-level hotspot predictors."""
from __future__ import annotations

from abc import ABC, abstractmethod
from collections import deque
from typing import Deque

import numpy as np


class BasePredictor(ABC):
    """Called at each epoch boundary with per-queue feature vector
    psi_q = [B_q_gen, R_q_peak, K_q, P_q].
    """

    def __init__(self, num_queues: int, window: int) -> None:
        self.num_queues = num_queues
        self.window = window
        self._history: Deque[np.ndarray] = deque(maxlen=window)

    def observe(self, feat: np.ndarray) -> None:
        self._history.append(feat.copy())

    @abstractmethod
    def predict(self) -> np.ndarray: ...

    def reset(self) -> None:
        self._history.clear()
