"""Stateless bucket-reassignment schedulers.

Canonical names:
``ewma_greedy``      : Algorithm 2. H_q = alpha*P_q + (1-alpha)*R_hat.
                       Greedy move with fit-condition gate + iterative H refresh.
``reactive_greedy``  : alpha = 1 (uses P_q only). Same greedy loop as above.
``reactive_oneshot`` : simpler threshold-only reactive baseline -- one pass,
                       move heaviest bucket of each hot queue to the
                       currently coolest queue, no fit-condition check.

Legacy aliases (accepted for backward compatibility):
``proposed`` -> ``ewma_greedy``
``current_only`` -> ``reactive_greedy``
``reactive_no_pred`` -> ``reactive_oneshot``
"""
from __future__ import annotations

from typing import Dict
import numpy as np

from .base import BaseStatelessScheduler
from ..config import StatelessSchedulerConfig


class GreedyStatelessScheduler(BaseStatelessScheduler):
    """Realises ``ewma_greedy`` and ``reactive_greedy``."""

    def __init__(self, cfg: StatelessSchedulerConfig, mode: str = "ewma_greedy") -> None:
        assert mode in ("ewma_greedy", "reactive_greedy")
        self.cfg = cfg
        self.mode = mode
        self.name = mode
        self.moves_this_epoch = 0

    def step(self, epoch, telem, current_table, pred_risk):
        P = telem["P_q"]
        B_q_gen = telem["B_q_gen"].copy()
        B_b = telem["B_b"].copy()

        if self.mode == "reactive_greedy":
            H = P.copy()
        else:
            a = self.cfg.alpha_blend
            H = a * P + (1.0 - a) * pred_risk

        new_table = current_table.copy()
        avg_B = B_q_gen.mean()
        # Fit-condition tolerance (Algorithm 2): the destination queue's
        # post-move byte count must stay within (1+tolerance) * mean.
        fit_slack = self.cfg.fit_condition_tolerance * (avg_B + 1.0)

        hot = np.where(H > self.cfg.tau_hot_s)[0]
        cold = np.where(H < self.cfg.tau_cold_s)[0]
        self.moves_this_epoch = 0

        while len(hot) > 0 and len(cold) > 0 and \
                self.moves_this_epoch < self.cfg.max_moves_per_epoch:
            q_src = int(hot[np.argmax(H[hot])])
            q_dst = int(cold[np.argmin(H[cold])])
            bucket_ids = np.where(new_table == q_src)[0]
            if bucket_ids.size == 0:
                hot = hot[hot != q_src]
                continue
            order = bucket_ids[np.argsort(-B_b[bucket_ids])]
            moved = False
            for b in order:
                b = int(b)
                new_B_src = B_q_gen[q_src] - B_b[b]
                new_B_dst = B_q_gen[q_dst] + B_b[b]
                if new_B_dst <= avg_B + fit_slack:
                    new_table[b] = q_dst
                    B_q_gen[q_src] = new_B_src
                    B_q_gen[q_dst] = new_B_dst
                    self.moves_this_epoch += 1
                    denom = avg_B + 1e-9
                    H[q_src] = max(0.0, H[q_src] - 0.5 * B_b[b] / denom)
                    H[q_dst] = min(1.0, H[q_dst] + 0.5 * B_b[b] / denom)
                    moved = True
                    break
            if not moved:
                cold = np.setdiff1d(cold, [q_dst])
            hot = np.where(H > self.cfg.tau_hot_s)[0]
            if moved:
                cold = np.where(H < self.cfg.tau_cold_s)[0]
        return new_table


class ReactiveNoPredStatelessScheduler(BaseStatelessScheduler):
    """One pass: for each hot queue, move its heaviest bucket to the
    currently coldest free queue. No fit check, no iteration."""

    name = "reactive_oneshot"

    def __init__(self, cfg: StatelessSchedulerConfig) -> None:
        self.cfg = cfg
        self.moves_this_epoch = 0

    def step(self, epoch, telem, current_table, pred_risk):
        P = telem["P_q"]
        B_b = telem["B_b"]
        new_table = current_table.copy()
        self.moves_this_epoch = 0
        hot = np.where(P > self.cfg.tau_hot_s)[0]
        used: set[int] = set()
        for q_src in [int(hot[i]) for i in np.argsort(-P[hot])]:
            if self.moves_this_epoch >= self.cfg.max_moves_per_epoch:
                break
            bucket_ids = np.where(new_table == q_src)[0]
            if bucket_ids.size == 0:
                continue
            b = int(bucket_ids[np.argmax(B_b[bucket_ids])])
            q_dst = None
            for q in np.argsort(P):
                q = int(q)
                if q == q_src or q in used:
                    continue
                q_dst = q
                break
            if q_dst is None:
                break
            new_table[b] = q_dst
            used.add(q_dst)
            self.moves_this_epoch += 1
        return new_table
