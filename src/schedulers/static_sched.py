"""Static baseline: RSS table never changes."""
from __future__ import annotations

from .base import BaseStatelessScheduler, BaseStatefulScheduler


class StaticStatelessScheduler(BaseStatelessScheduler):
    name = "static"

    def step(self, epoch, telem, current_table, pred_risk):
        return current_table.copy()


class StaticStatefulScheduler(BaseStatefulScheduler):
    name = "static"

    def step(self, epoch, telem, current_table):
        return current_table.copy()

    def tick_handoffs(self, epoch):
        return
