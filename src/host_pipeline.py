"""Event-driven host-side pipeline model.

Faithfully models the full QDMA descriptor path from FPGA packet generation
to descriptor writeback, per queue. This replaces the old bin-averaged
``queue_model.py`` + ``host_model.py`` abstraction, which was only a
steady-state rate approximation and did not capture the packet-level timing
the paper's scheduler depends on (P_q is driven by *descriptor occupancy*,
which is a temporal phenomenon).

Packet life, per queue q (descriptor ring depth D_q):

    T0                      : FPGA generates packet. Checks V_q (free descriptors
                              at FPGA). If V_q == 0 -> drop at FPGA.
    T0 + T_dma(size)        : packet arrives in host C2H ring. Occupies one
                              descriptor slot.
    t_start = max(T_ring_arrival, core_next_free_time[q]) : core picks up.
    t_start + T_app(size)   : core finishes processing (DPDK poller for the
                              stateless domain, TCP RX stack for the stateful
                              domain). Descriptor is freed at host.
    + T_wb                  : descriptor writeback TLP reaches FPGA. FPGA
                              observes credit return (V_q += 1).

Invariants:

* #in-flight descriptors at time t = D_q - V_q(t)
  (= packets currently somewhere in DMA + ring + core + writeback pipeline)
* U_q(t) = D_q - V_q(t)  matches the paper's occupancy definition.
* A drop happens ONLY when FPGA has no free descriptor at the moment of
  packet generation. Once admitted, a packet always completes processing.

Uncongested round trip T_RTT(size) = T_dma(size) + T_app(size) + T_wb.
We calibrate T_dma(size) from a (size, RTT_ns) table measured on the real
hardware; T_app and T_wb are parameters.

The pipeline is driven per packet (not per bin). SimPy is not needed because
each packet's fate is computable analytically from the queue's state
(core_next_free_time + scheduled credit returns).
"""
from __future__ import annotations

import heapq
from dataclasses import dataclass, field
from typing import List, Sequence, Tuple

import numpy as np


# ----------------------------------------------------------------------
# RTT calibration
# ----------------------------------------------------------------------
def build_t_dma_one_way_fn(rtt_table: Sequence[Sequence[float]],
                           t_app_base_ns: float,
                           t_wb_ns: float):
    """Return a function ``f(size_bytes) -> T_dma_ns`` from an uncongested
    measured RTT table. T_dma(s) = RTT(s) - T_app_base - T_wb.

    The baseline T_app_base corresponds to per-packet processing under ZERO
    congestion (i.e. the constant term of the measured RTT, not the
    congested / shared-core effective T_app the pipeline uses at runtime).
    """
    xs = np.array([row[0] for row in rtt_table], dtype=np.float64)
    ys = np.array([row[1] for row in rtt_table], dtype=np.float64)

    def f(size_bytes: float) -> float:
        rtt = float(np.interp(size_bytes, xs, ys))
        return max(100.0, rtt - t_app_base_ns - t_wb_ns)

    return f


# ----------------------------------------------------------------------
# Per-queue pipeline
# ----------------------------------------------------------------------
@dataclass
class QueuePipelineStats:
    """Resettable per-bin counters. Also tracks occupancy min/max within the
    bin so telemetry can compute U_q growth properly.

    Drops are split by root cause so we can attribute pressure to the right
    layer:
      * credit_drop_* : FPGA had no descriptor credit (ring full at host).
      * pcie_drop_*   : FPGA egress FIFO full (PCIe link saturated).
    The ``drop_pkts`` / ``drop_bytes`` aggregates both so that telemetry's
    P_q computation (via L_q = drops / gen) sees total loss.

    ``latency_samples_ns`` is the per-bin list of end-to-end packet
    latencies (in nanoseconds) for packets that were admitted in this
    bin. Latency is measured from the moment the FPGA generates the
    packet (``t_gen``) to the moment the host core finishes processing
    it (``t_service_end``). The sim aggregates these per epoch and
    computes p50/p95/p99/p99.9/max/mean for tail-latency reporting.
    """
    adm_pkts: int = 0
    adm_bytes: float = 0.0
    drop_pkts: int = 0
    drop_bytes: float = 0.0
    credit_drop_pkts: int = 0
    credit_drop_bytes: float = 0.0
    pcie_drop_pkts: int = 0
    pcie_drop_bytes: float = 0.0
    credits_returned: int = 0
    u_sum: float = 0.0       # for time-average occupancy if desired
    u_max: int = 0
    latency_samples_ns: list = field(default_factory=list)

    def reset(self) -> None:
        self.adm_pkts = 0
        self.adm_bytes = 0.0
        self.drop_pkts = 0
        self.drop_bytes = 0.0
        self.credit_drop_pkts = 0
        self.credit_drop_bytes = 0.0
        self.pcie_drop_pkts = 0
        self.pcie_drop_bytes = 0.0
        self.credits_returned = 0
        self.u_sum = 0.0
        self.u_max = 0
        self.latency_samples_ns = []


@dataclass
class QueuePipeline:
    queue_id: int
    D_q: int

    t_app_mean_ns: float
    t_app_jitter_ns: float
    t_wb_ns: float
    per_conn_lookup_ns: float
    core_share_factor: float  # how much the core is shared; T_app *= this

    t_dma_fn: callable
    rng: np.random.Generator

    def __post_init__(self) -> None:
        self.credits_at_fpga: int = self.D_q
        self.next_service_free_ns: float = 0.0
        self.credit_return_heap: List[Tuple[float, int]] = []
        self.stats = QueuePipelineStats()
        # Track the current epoch's starting occupancy and per-bin end occupancy
        self._last_advance_ns: float = 0.0

    # ------------------------------------------------------------------
    # Internal timing helpers
    # ------------------------------------------------------------------
    def _t_app(self, size_bytes: int) -> float:
        t = self.t_app_mean_ns
        if self.t_app_jitter_ns > 0:
            t = t + float(self.rng.normal(0.0, self.t_app_jitter_ns))
        t = max(self.t_app_mean_ns * 0.1, t) * self.core_share_factor
        return t + self.per_conn_lookup_ns

    def _t_dma(self, size_bytes: int) -> float:
        return self.t_dma_fn(float(size_bytes))

    # ------------------------------------------------------------------
    # Pipeline advance: process credit returns due by time t
    # ------------------------------------------------------------------
    def advance_to(self, t_ns: float) -> None:
        heap = self.credit_return_heap
        while heap and heap[0][0] <= t_ns:
            _, n = heapq.heappop(heap)
            new_credits = min(self.D_q, self.credits_at_fpga + n)
            ret = new_credits - self.credits_at_fpga
            self.credits_at_fpga = new_credits
            self.stats.credits_returned += ret
        self._last_advance_ns = t_ns

    # ------------------------------------------------------------------
    # FPGA-side attempt to admit a generated packet
    #
    # If ``t_ring_arrive_ns`` is provided, the caller (sim driver) has
    # already run the packet through the shared PCIe link and determined
    # when the packet will arrive at the host ring. In that case T_dma
    # calibration is bypassed.
    #
    # If ``t_ring_arrive_ns`` is None, the legacy uncongested T_dma(size)
    # from the RTT table is used (no bandwidth contention).
    # ------------------------------------------------------------------
    def try_admit(self, t_gen_ns: float, size_bytes: int,
                  t_ring_arrive_ns: float | None = None) -> bool:
        self.advance_to(t_gen_ns)
        if self.credits_at_fpga <= 0:
            self.stats.drop_pkts += 1
            self.stats.drop_bytes += float(size_bytes)
            self.stats.credit_drop_pkts += 1
            self.stats.credit_drop_bytes += float(size_bytes)
            return False

        self.credits_at_fpga -= 1
        U = self.D_q - self.credits_at_fpga
        if U > self.stats.u_max:
            self.stats.u_max = U

        if t_ring_arrive_ns is None:
            t_ring_arrive = t_gen_ns + self._t_dma(size_bytes)
        else:
            t_ring_arrive = float(t_ring_arrive_ns)

        t_start = max(t_ring_arrive, self.next_service_free_ns)
        t_service_end = t_start + self._t_app(size_bytes)
        t_credit_back = t_service_end + self.t_wb_ns

        self.next_service_free_ns = t_service_end
        heapq.heappush(self.credit_return_heap, (t_credit_back, 1))

        self.stats.adm_pkts += 1
        self.stats.adm_bytes += float(size_bytes)
        # End-to-end per-packet latency: from FPGA generation to core
        # finishing processing. Includes PCIe serialization, ring wait,
        # and core service (which inflates under contention -> the
        # canonical tail-latency signal).
        self.stats.latency_samples_ns.append(float(t_service_end) - float(t_gen_ns))
        return True

    def record_pcie_drop(self, size_bytes: int) -> None:
        """Record a drop that happened at the PCIe egress FIFO (no credit
        consumed at the queue, the packet never reached the descriptor
        ring). Aggregates into the per-queue total drop counter so L_q /
        P_q reflect the loss."""
        self.stats.drop_pkts += 1
        self.stats.drop_bytes += float(size_bytes)
        self.stats.pcie_drop_pkts += 1
        self.stats.pcie_drop_bytes += float(size_bytes)

    # ------------------------------------------------------------------
    # Read-only views
    # ------------------------------------------------------------------
    @property
    def occupancy(self) -> int:
        return self.D_q - self.credits_at_fpga

    def snapshot_V(self) -> int:
        return int(self.credits_at_fpga)


# ----------------------------------------------------------------------
# Host domain: multiple queues + core sharing
# ----------------------------------------------------------------------
@dataclass
class HostDomainPipeline:
    queues: List[QueuePipeline]
    queue_to_core: np.ndarray

    @property
    def n_queues(self) -> int:
        return len(self.queues)

    def advance_all_to(self, t_ns: float) -> None:
        for q in self.queues:
            q.advance_to(t_ns)

    def reset_bin_stats(self) -> None:
        for q in self.queues:
            q.stats.reset()

    def snapshot_V(self) -> np.ndarray:
        return np.array([q.snapshot_V() for q in self.queues], dtype=np.int64)

    def snapshot_occupancy(self) -> np.ndarray:
        return np.array([q.occupancy for q in self.queues], dtype=np.int64)


def build_queue_to_core(n_queues: int, n_cores: int, policy: str) -> np.ndarray:
    q2c = np.zeros(n_queues, dtype=np.int64)
    if policy in ("one_to_one", "round_robin"):
        for q in range(n_queues):
            q2c[q] = q % n_cores
    elif policy == "block":
        per = max(1, int(np.ceil(n_queues / n_cores)))
        for q in range(n_queues):
            q2c[q] = min(n_cores - 1, q // per)
    else:
        raise ValueError(f"unknown queue_to_core policy: {policy}")
    return q2c


def build_domain_pipeline(
    n_queues: int,
    n_cores: int,
    D_q: int,
    t_app_mean_ns: float,
    t_app_jitter_ns: float,
    t_wb_ns: float,
    per_conn_lookup_ns: float,
    rtt_table: Sequence[Sequence[float]],
    t_app_base_ns: float,
    policy: str,
    rng: np.random.Generator,
) -> HostDomainPipeline:
    q2c = build_queue_to_core(n_queues, n_cores, policy)
    core_load = np.bincount(q2c, minlength=n_cores)
    t_dma_fn = build_t_dma_one_way_fn(rtt_table, t_app_base_ns, t_wb_ns)

    queues = []
    for q in range(n_queues):
        share = float(core_load[q2c[q]])  # # queues on my core
        queues.append(QueuePipeline(
            queue_id=q,
            D_q=D_q,
            t_app_mean_ns=t_app_mean_ns,
            t_app_jitter_ns=t_app_jitter_ns,
            t_wb_ns=t_wb_ns,
            per_conn_lookup_ns=per_conn_lookup_ns,
            core_share_factor=share,
            t_dma_fn=t_dma_fn,
            rng=rng,
        ))
    return HostDomainPipeline(queues=queues, queue_to_core=q2c)
