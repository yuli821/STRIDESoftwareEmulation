"""Telemetry accumulators + epoch-level pressure computation.

Per bin we capture:
  * B_b[e,k], N_b[e,k]       : generated bucket bytes / packets this bin
                               (from TraceSet, before admission filtering)
  * B_q_gen, N_q_gen         : bucket -> queue folding of the above
  * B_q_adm, N_q_adm         : admitted bytes / packets this bin (from pipeline)
  * B_q_drop, N_q_drop       : dropped bytes / packets this bin (from pipeline)
  * K_q                      : credits returned to FPGA this bin (from pipeline)
  * V_q at end of bin        : available credits at FPGA
  * latency_samples          : per-packet end-to-end latencies (ns) for
                               admitted packets this bin; used at epoch
                               end to compute tail quantiles
                               (p50/p95/p99/p99.9/max).
At epoch end we compute O_q, G_q+, L_q, P_q per the paper, plus the
tail-latency summary over the epoch window.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List

import numpy as np


@dataclass
class DomainTelemetry:
    H: int
    num_buckets: int
    num_queues: int
    D_q: int
    w1: float
    w2: float
    w3: float
    eps: float

    def __post_init__(self) -> None:
        self.reset_epoch()

    def reset_epoch(self) -> None:
        H, B, Q = self.H, self.num_buckets, self.num_queues
        self.B_b = np.zeros((H, B), dtype=np.float64)
        self.N_b = np.zeros((H, B), dtype=np.int64)
        self.B_q_gen = np.zeros((H, Q), dtype=np.float64)
        self.N_q_gen = np.zeros((H, Q), dtype=np.int64)
        self.B_q_adm = np.zeros((H, Q), dtype=np.float64)
        self.N_q_adm = np.zeros((H, Q), dtype=np.int64)
        self.B_q_drop = np.zeros((H, Q), dtype=np.float64)
        self.N_q_drop = np.zeros((H, Q), dtype=np.int64)
        self.K_q = np.zeros((H, Q), dtype=np.int64)
        self.V_q = np.zeros((H + 1, Q), dtype=np.int64)
        # Per-epoch latency sample buffer (flat list of per-packet
        # end-to-end latencies in ns, across all queues/bins). Reset
        # each epoch.
        self.latency_samples_ns: List[float] = []

    def set_V_start(self, V_start: np.ndarray) -> None:
        self.V_q[0] = V_start

    def record_bin(self,
                   k: int,
                   bucket_bytes: np.ndarray,
                   bucket_pkts: np.ndarray,
                   rss_table: np.ndarray,
                   adm_pkts: np.ndarray,
                   adm_bytes: np.ndarray,
                   drop_pkts: np.ndarray,
                   drop_bytes: np.ndarray,
                   K_q_this_bin: np.ndarray,
                   V_q_end: np.ndarray,
                   latency_samples_ns: List[float] = None) -> None:
        Q = self.num_queues
        self.B_b[k] = bucket_bytes
        self.N_b[k] = bucket_pkts
        self.B_q_gen[k] = np.bincount(rss_table, weights=bucket_bytes,
                                      minlength=Q)[:Q]
        self.N_q_gen[k] = np.bincount(rss_table, weights=bucket_pkts,
                                      minlength=Q)[:Q].astype(np.int64)
        self.B_q_adm[k] = adm_bytes
        self.N_q_adm[k] = adm_pkts
        self.B_q_drop[k] = drop_bytes
        self.N_q_drop[k] = drop_pkts
        self.K_q[k] = K_q_this_bin
        self.V_q[k + 1] = V_q_end
        if latency_samples_ns:
            self.latency_samples_ns.extend(latency_samples_ns)

    def finalize_epoch(self, delta_bin_ns: float) -> Dict[str, np.ndarray]:
        H = self.H
        B_b = self.B_b.sum(axis=0)
        N_b = self.N_b.sum(axis=0)
        R_b_peak = self.B_b.max(axis=0) / (delta_bin_ns * 1e-9)

        B_q_gen = self.B_q_gen.sum(axis=0)
        N_q_gen = self.N_q_gen.sum(axis=0)
        B_q_adm = self.B_q_adm.sum(axis=0)
        N_q_adm = self.N_q_adm.sum(axis=0)
        B_q_drop = self.B_q_drop.sum(axis=0)
        N_q_drop = self.N_q_drop.sum(axis=0)
        K_q = self.K_q.sum(axis=0)
        R_q_peak = self.B_q_gen.max(axis=0) / (delta_bin_ns * 1e-9)

        D = float(self.D_q)
        U_q = D - self.V_q
        U_start = U_q[0]
        U_end = U_q[H]
        U_max = U_q.max(axis=0)

        G_q = U_end - U_start
        O_q = U_end / D
        G_q_plus = np.maximum(0.0, G_q) / D
        L_q = B_q_drop / (B_q_gen + self.eps)
        P_q = np.clip(self.w1 * O_q + self.w2 * G_q_plus + self.w3 * L_q,
                      0.0, 1.0)

        lat_stats = self._latency_stats()

        return dict(
            B_b=B_b, N_b=N_b, R_b_peak=R_b_peak,
            B_q_gen=B_q_gen, N_q_gen=N_q_gen,
            B_q_adm=B_q_adm, N_q_adm=N_q_adm,
            B_q_drop=B_q_drop, N_q_drop=N_q_drop,
            K_q=K_q, R_q_peak=R_q_peak,
            U_start=U_start, U_end=U_end, U_max=U_max,
            G_q=G_q, O_q=O_q, G_q_plus=G_q_plus, L_q=L_q, P_q=P_q,
            **lat_stats,
        )

    def _latency_stats(self) -> Dict[str, float]:
        """Compute tail-latency summary (mean, p50, p95, p99, p99.9, max)
        from this epoch's per-packet samples. All values are in
        nanoseconds; returns zero-filled stats when no packets were
        admitted this epoch."""
        samples = self.latency_samples_ns
        if not samples:
            return dict(
                lat_n=0, lat_mean_ns=0.0,
                lat_p50_ns=0.0, lat_p95_ns=0.0,
                lat_p99_ns=0.0, lat_p999_ns=0.0,
                lat_max_ns=0.0,
            )
        arr = np.asarray(samples, dtype=np.float64)
        qs = np.quantile(arr, [0.50, 0.95, 0.99, 0.999])
        return dict(
            lat_n=int(arr.size),
            lat_mean_ns=float(arr.mean()),
            lat_p50_ns=float(qs[0]),
            lat_p95_ns=float(qs[1]),
            lat_p99_ns=float(qs[2]),
            lat_p999_ns=float(qs[3]),
            lat_max_ns=float(arr.max()),
        )
