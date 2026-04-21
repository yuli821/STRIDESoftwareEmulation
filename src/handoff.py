"""Software handoff state machine (stateful domain).

Algorithm 3 migration protocol, per the design document:

  1. FPGA proposes a move (bucket, q_src, q_dst) and notifies the source core.
  2. Source core marks the flow state "in transition" and enters a **drain**
     phase, finishing any in-flight packets of the bucket that are already
     in the source queue.
  3. Meanwhile the FPGA buffers newly-generated packets of the bucket (no
     delivery to the source queue). Overflow of this buffer == drop.
  4. Once the drain completes, the kernel **migrates** per-flow metadata
     (TCB, timers), the application-thread ownership, and TX-side ownership
     to the destination partition (a.k.a. destination core).
  5. Destination core ACKs the FPGA.
  6. FPGA commits the new RSS indirection entry and **releases** the
     buffered packets to the destination queue.

The three software phases (drain, migration, ack) are modeled as independent
Gaussian samples whose sum is the total handoff latency. The running EWMA of
realised totals feeds the scheduler's penalty term.

Host-visible flags per stateful queue (exposed to the hardware scheduler):
  R_q      : destination-ready to accept flow state.
  H_pend   : handoff in progress involving this queue.
  T_hand   : EWMA of realised total handoff latencies.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np


@dataclass
class Handoff:
    bucket: int
    q_src: int
    q_dst: int
    start_epoch: int
    latency_ns: float                 # total (drain + migration + ack)
    ns_remaining: float
    t_drain_ns: float = 0.0
    t_migration_ns: float = 0.0
    t_ack_ns: float = 0.0
    buffered_bytes: int = 0
    buffered_pkts: int = 0
    buffer_overflow_bytes: int = 0
    buffer_overflow_pkts: int = 0
    completed: bool = False


@dataclass
class HandoffManager:
    n_queues: int
    drain_mean_ns: float
    drain_std_ns: float
    migration_mean_ns: float
    migration_std_ns: float
    ack_mean_ns: float
    ack_std_ns: float
    single_phase_mean_ns: Optional[float]   # legacy override
    single_phase_std_ns: Optional[float]
    ewma_alpha: float
    max_concurrent: int
    buffer_capacity_bytes: int

    def __post_init__(self) -> None:
        self.pending: Dict[int, Handoff] = {}
        self.R_q = np.ones(self.n_queues, dtype=np.int8)
        self.H_pend = np.zeros(self.n_queues, dtype=np.int8)
        default_total = (self.drain_mean_ns + self.migration_mean_ns
                         + self.ack_mean_ns)
        init_val = (self.single_phase_mean_ns
                    if self.single_phase_mean_ns is not None
                    else default_total)
        self.T_hand_ewma = np.full(self.n_queues, float(init_val),
                                   dtype=np.float64)
        self.completed_history: List[Handoff] = []

    # ------------------------------------------------------------------
    def _sample_latency(self, rng: np.random.Generator
                        ) -> tuple[float, float, float, float]:
        if self.single_phase_mean_ns is not None:
            total = max(1.0, float(rng.normal(self.single_phase_mean_ns,
                                              self.single_phase_std_ns or 0.0)))
            return total, total, 0.0, 0.0
        td = max(0.0, float(rng.normal(self.drain_mean_ns, self.drain_std_ns)))
        tm = max(0.0, float(rng.normal(self.migration_mean_ns,
                                       self.migration_std_ns)))
        ta = max(0.0, float(rng.normal(self.ack_mean_ns, self.ack_std_ns)))
        total = max(1.0, td + tm + ta)
        return total, td, tm, ta

    # ------------------------------------------------------------------
    def can_issue(self, q_src: int, q_dst: int) -> bool:
        if len(self.pending) >= self.max_concurrent:
            return False
        if self.H_pend[q_src] or self.H_pend[q_dst]:
            return False
        if not self.R_q[q_dst]:
            return False
        return True

    def issue(self, bucket: int, q_src: int, q_dst: int, epoch: int,
              rng: np.random.Generator) -> Handoff:
        if bucket in self.pending:
            raise RuntimeError(f"bucket {bucket} already in handoff")
        total, td, tm, ta = self._sample_latency(rng)
        h = Handoff(bucket=bucket, q_src=q_src, q_dst=q_dst,
                    start_epoch=epoch, latency_ns=total, ns_remaining=total,
                    t_drain_ns=td, t_migration_ns=tm, t_ack_ns=ta)
        self.pending[bucket] = h
        self.H_pend[q_src] = 1
        self.H_pend[q_dst] = 1
        return h

    # ------------------------------------------------------------------
    def absorb_bucket_traffic(self, bucket: int, bytes_: float, pkts: int
                              ) -> tuple[int, int]:
        h = self.pending.get(bucket)
        if h is None:
            return 0, 0
        room = self.buffer_capacity_bytes - h.buffered_bytes
        if bytes_ <= room:
            h.buffered_bytes += int(bytes_)
            h.buffered_pkts += int(pkts)
            return 0, 0
        added_bytes = max(0, room)
        frac = added_bytes / max(1.0, bytes_)
        added_pkts = int(pkts * frac)
        h.buffered_bytes += int(added_bytes)
        h.buffered_pkts += added_pkts
        ovf_b = int(bytes_) - int(added_bytes)
        ovf_p = int(pkts) - added_pkts
        h.buffer_overflow_bytes += ovf_b
        h.buffer_overflow_pkts += ovf_p
        return ovf_b, ovf_p

    def affected_buckets(self) -> set[int]:
        return set(self.pending.keys())

    # ------------------------------------------------------------------
    def advance_time(self, dt_ns: float, epoch: int) -> List[Handoff]:
        finished: List[Handoff] = []
        for b, h in list(self.pending.items()):
            h.ns_remaining -= dt_ns
            if h.ns_remaining <= 0:
                h.completed = True
                self.H_pend[h.q_src] = 0
                self.H_pend[h.q_dst] = 0
                a = self.ewma_alpha
                self.T_hand_ewma[h.q_dst] = (
                    a * h.latency_ns + (1.0 - a) * self.T_hand_ewma[h.q_dst]
                )
                finished.append(h)
                del self.pending[b]
                self.completed_history.append(h)
        return finished
