"""Shared PCIe link model.

A single PCIe link (default 64 Gbps effective) is the data path shared by
the stateless and stateful domains. Packets from both domains contend for
this link; when the link cannot keep up, packets back up in the FPGA's
egress staging FIFO, and once the FIFO fills the FPGA drops incoming
packets ("PCIe drop").

Three-layer architecture:

  FPGA layer : packet generator + descriptor-credit check
               - if V_q == 0         -> credit drop
               - else enqueue into   v
  PCIe layer : serializes packets at B Gbps
               - FPGA egress FIFO of capacity fifo_bytes
               - if FIFO would overflow -> PCIe drop
               - otherwise link busy for size*8/B seconds
               - packet arrives at host ring at t_complete
  Host layer : QDMA ring (depth D_q) -> core -> writeback
               - credit returns to FPGA after T_wb

This module implements the middle (PCIe) layer as a shared resource. One
instance serves both domains. Accounting is per domain so telemetry can
report PCIe drops split by domain.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional


@dataclass
class PCIeLinkStats:
    """Resettable per-bin counters."""
    accepted_pkts: int = 0
    accepted_bytes: float = 0.0
    dropped_pkts: int = 0
    dropped_bytes: float = 0.0

    # Per-domain PCIe drops (indexed by domain name).
    dropped_pkts_by_domain: Dict[str, int] = field(default_factory=dict)
    dropped_bytes_by_domain: Dict[str, float] = field(default_factory=dict)

    # Peak FIFO occupancy (bytes) seen during this bin.
    fifo_peak_bytes: float = 0.0

    def reset(self) -> None:
        self.accepted_pkts = 0
        self.accepted_bytes = 0.0
        self.dropped_pkts = 0
        self.dropped_bytes = 0.0
        self.dropped_pkts_by_domain.clear()
        self.dropped_bytes_by_domain.clear()
        self.fifo_peak_bytes = 0.0


@dataclass
class PCIeLink:
    """Single shared PCIe link between FPGA and host.

    Parameters:
      bandwidth_gbps : effective throughput of the link (default 64 for
                       VCK190 PCIe Gen4 x8 effective after overhead).
      fifo_bytes     : capacity of the FPGA egress staging FIFO. When
                       modeled pending bytes exceed this, a packet is
                       dropped.
      setup_ns       : fixed per-packet DMA-initiation cost added on top of
                       serialization (TLP header formation, descriptor
                       fetch, bridge traversal, etc.). If 0, the link
                       behaves as a pure rate limiter.
    """
    bandwidth_gbps: float
    fifo_bytes: int
    setup_ns: float = 0.0

    def __post_init__(self) -> None:
        self.bandwidth_bps: float = self.bandwidth_gbps * 1e9
        self.busy_until_ns: float = 0.0
        self.stats = PCIeLinkStats()

    # ------------------------------------------------------------------
    def pending_bytes(self, t_ns: float) -> float:
        """Bytes currently in-flight on the link at time ``t_ns`` (i.e.
        bytes that still need to finish transmitting). Equals the outstanding
        serialization work divided out to bytes."""
        if self.busy_until_ns <= t_ns:
            return 0.0
        return (self.busy_until_ns - t_ns) * self.bandwidth_bps / 8e9

    def _ns_per_byte(self) -> float:
        return 8.0 * 1e9 / self.bandwidth_bps

    # ------------------------------------------------------------------
    def try_transmit(self, t_ready_ns: float, size_bytes: int,
                     domain: str = "unknown") -> Optional[float]:
        """Attempt to transmit a packet through the link.

        Returns ``t_complete_ns`` (when the packet fully arrives at the
        host ring) on success, or ``None`` if dropped because the FPGA
        egress FIFO is full.
        """
        pending = self.pending_bytes(t_ready_ns)
        if pending + float(size_bytes) > float(self.fifo_bytes):
            self.stats.dropped_pkts += 1
            self.stats.dropped_bytes += float(size_bytes)
            self.stats.dropped_pkts_by_domain[domain] = (
                self.stats.dropped_pkts_by_domain.get(domain, 0) + 1
            )
            self.stats.dropped_bytes_by_domain[domain] = (
                self.stats.dropped_bytes_by_domain.get(domain, 0.0)
                + float(size_bytes)
            )
            return None

        # Serialize: link becomes busy for size*8/B.
        t_start = max(t_ready_ns + self.setup_ns, self.busy_until_ns)
        t_complete = t_start + float(size_bytes) * self._ns_per_byte()
        self.busy_until_ns = t_complete

        self.stats.accepted_pkts += 1
        self.stats.accepted_bytes += float(size_bytes)

        current_pending = self.pending_bytes(t_ready_ns)
        if current_pending > self.stats.fifo_peak_bytes:
            self.stats.fifo_peak_bytes = current_pending

        return t_complete

    # ------------------------------------------------------------------
    def advance_to(self, t_ns: float) -> None:
        """Advance the virtual clock; the link becomes free at ``busy_until_ns``.
        Nothing to do unless you want to clamp idle times, but exposed for
        symmetry with QueuePipeline.
        """
        if self.busy_until_ns < t_ns:
            self.busy_until_ns = t_ns

    def reset_bin_stats(self) -> None:
        self.stats.reset()
