"""Base classes for domain schedulers."""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict

import numpy as np


class BaseStatelessScheduler(ABC):
    name: str = "base"

    @abstractmethod
    def step(self, epoch: int, telem: Dict[str, np.ndarray],
             current_table: np.ndarray,
             pred_risk: np.ndarray) -> np.ndarray: ...

    def reset(self) -> None:
        pass


class BaseStatefulScheduler(ABC):
    name: str = "base"

    @abstractmethod
    def step(self, epoch: int, telem: Dict[str, np.ndarray],
             current_table: np.ndarray) -> np.ndarray: ...

    @abstractmethod
    def tick_handoffs(self, epoch: int) -> None: ...

    def reset(self) -> None:
        pass
