"""Reactive stateful bucket-reassignment scheduler with software handoff
(Algorithm 3)."""
from __future__ import annotations

from typing import Dict
import numpy as np

from .base import BaseStatefulScheduler
from ..config import StatefulSchedulerConfig
from ..handoff import HandoffManager


class ReactiveStatefulScheduler(BaseStatefulScheduler):
    name = "stateful_proposed"

    def __init__(self, cfg: StatefulSchedulerConfig,
                 handoff_mgr: HandoffManager,
                 rng: np.random.Generator,
                 epoch_ns: float) -> None:
        self.cfg = cfg
        self.handoff = handoff_mgr
        self.rng = rng
        self.epoch_ns = epoch_ns
        self.issued_this_epoch = 0
        self.committed_this_epoch = 0
        self._finished: list = []

    def tick_handoffs(self, epoch: int) -> None:
        self._finished = self.handoff.advance_time(self.epoch_ns, epoch)
        self.committed_this_epoch = len(self._finished)

    def step(self, epoch, telem, current_table):
        new_table = current_table.copy()
        for h in self._finished:
            new_table[h.bucket] = h.q_dst

        P = telem["P_q"]
        B_q_gen = telem["B_q_gen"].copy()
        B_b = telem["B_b"]
        eps = 1e-9
        B_sigma = float(B_q_gen.sum())
        share_q = B_q_gen / (B_sigma + eps)
        S = self.cfg.eta1 * P + self.cfg.eta2 * share_q

        hot = np.where(P > self.cfg.tau_hot_t)[0]
        cold_mask = ((P < self.cfg.tau_cold_t)
                     & (self.handoff.R_q == 1)
                     & (self.handoff.H_pend == 0))
        cold = np.where(cold_mask)[0]

        cand_buckets = np.where(np.isin(new_table, hot))[0]
        cand_buckets = cand_buckets[np.argsort(-B_b[cand_buckets])]
        self.issued_this_epoch = 0

        for b in cand_buckets:
            if self.issued_this_epoch >= self.cfg.max_concurrent_handoffs:
                break
            b = int(b)
            q_src = int(new_table[b])
            if self.handoff.H_pend[q_src]:
                continue
            best_gain = -np.inf
            best_dst = None
            for q_dst in cold:
                q_dst = int(q_dst)
                if q_dst == q_src:
                    continue
                if not self.handoff.can_issue(q_src, q_dst):
                    continue
                nB_src = B_q_gen[q_src] - B_b[b]
                nB_dst = B_q_gen[q_dst] + B_b[b]
                S_src_p = self.cfg.eta1 * P[q_src] + self.cfg.eta2 * (nB_src / (B_sigma + eps))
                S_dst_p = self.cfg.eta1 * P[q_dst] + self.cfg.eta2 * (nB_dst / (B_sigma + eps))
                benefit = max(S[q_src], S[q_dst]) - max(S_src_p, S_dst_p)
                penalty = self.handoff.T_hand_ewma[q_dst] / self.epoch_ns
                gain = benefit - self.cfg.lambda_t * penalty
                if gain > best_gain:
                    best_gain = gain
                    best_dst = q_dst
            if best_dst is not None and best_gain > self.cfg.epsilon_t:
                self.handoff.issue(b, q_src, best_dst, epoch, self.rng)
                self.issued_this_epoch += 1
                B_q_gen[q_src] -= B_b[b]
                B_q_gen[best_dst] += B_b[b]
                S[q_src] = self.cfg.eta1 * P[q_src] + self.cfg.eta2 * (B_q_gen[q_src] / (B_sigma + eps))
                S[best_dst] = self.cfg.eta1 * P[best_dst] + self.cfg.eta2 * (B_q_gen[best_dst] / (B_sigma + eps))
                cold = cold[cold != best_dst]
        return new_table
