"""Scheduler factory."""
from __future__ import annotations

import numpy as np

from ..config import StatelessSchedulerConfig, StatefulSchedulerConfig
from ..handoff import HandoffManager

from .base import BaseStatelessScheduler, BaseStatefulScheduler
from .static_sched import StaticStatelessScheduler, StaticStatefulScheduler
from .stateless_proposed import (GreedyStatelessScheduler,
                                 ReactiveNoPredStatelessScheduler)
from .stateful_proposed import ReactiveStatefulScheduler


def make_stateless_scheduler(cfg: StatelessSchedulerConfig
                             ) -> BaseStatelessScheduler:
    t = cfg.scheduler_type
    # Backward-compatible aliases for older experiment names.
    alias = {
        "proposed": "ewma_greedy",
        "current_only": "reactive_greedy",
        "reactive_no_pred": "reactive_oneshot",
    }
    t = alias.get(t, t)
    if t == "static":
        return StaticStatelessScheduler()
    if t in ("ewma_greedy", "reactive_greedy"):
        return GreedyStatelessScheduler(cfg, mode=t)
    if t == "reactive_oneshot":
        return ReactiveNoPredStatelessScheduler(cfg)
    raise ValueError(f"unknown stateless scheduler_type: {t}")


def make_stateful_scheduler(cfg: StatefulSchedulerConfig,
                            handoff_mgr: HandoffManager,
                            rng: np.random.Generator,
                            epoch_ns: float) -> BaseStatefulScheduler:
    t = cfg.scheduler_type
    if t == "static":
        return StaticStatefulScheduler()
    if t == "proposed":
        return ReactiveStatefulScheduler(cfg, handoff_mgr, rng, epoch_ns)
    raise ValueError(f"unknown stateful scheduler_type: {t}")


__all__ = ["BaseStatelessScheduler", "BaseStatefulScheduler",
           "make_stateless_scheduler", "make_stateful_scheduler"]
