"""Predictor factory."""
from __future__ import annotations

from ..config import PredictorConfig
from .base import BasePredictor
from .ewma import EWMAPredictor
from .linear import LinearPredictor
from .oracle import OraclePredictor
from .tcn import TCNPredictor


def make_predictor(num_queues: int, cfg: PredictorConfig) -> BasePredictor:
    t = cfg.predictor_type
    if t == "ewma":
        return EWMAPredictor(num_queues, cfg.W_window_epochs, cfg.ewma_alpha)
    if t == "linear":
        return LinearPredictor(num_queues, cfg.W_window_epochs, cfg.linear_lookback)
    if t == "oracle":
        return OraclePredictor(num_queues, cfg.W_window_epochs)
    if t == "tcn":
        return TCNPredictor(num_queues, cfg.W_window_epochs,
                            cfg.tcn_channels, cfg.tcn_kernel, cfg.tcn_layers)
    if t == "none":
        class _Zero(BasePredictor):
            def predict(self):
                import numpy as _np
                return _np.zeros(self.num_queues, dtype=_np.float64)
        return _Zero(num_queues, cfg.W_window_epochs)
    raise ValueError(f"unknown predictor_type: {t}")


__all__ = ["BasePredictor", "make_predictor"]
