"""Scheduler factory."""
from __future__ import annotations

import numpy as np

from ..config import StatelessSchedulerConfig, StatefulSchedulerConfig
from ..handoff import HandoffManager

from .base import BaseStatelessScheduler, BaseStatefulScheduler
from .static_sched import StaticStatelessScheduler, StaticStatefulScheduler
from .stateless_proposed import (GreedyStatelessScheduler,
                                 OneshotStatelessScheduler,
                                 canonicalize, parse_scheduler_type)
from .stateful_proposed import ReactiveStatefulScheduler


def make_stateless_scheduler(cfg: StatelessSchedulerConfig
                             ) -> BaseStatelessScheduler:
    t = canonicalize(cfg.scheduler_type)
    if t == "static":
        return StaticStatelessScheduler()
    signal, policy = parse_scheduler_type(t)
    if policy == "greedy":
        return GreedyStatelessScheduler(cfg, signal=signal)
    if policy == "oneshot":
        return OneshotStatelessScheduler(cfg, signal=signal)
    raise ValueError(f"unsupported scheduler_type: {cfg.scheduler_type!r}")


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
