"""Per-flow packet-level traffic traces.

Everything upstream of the host pipeline is expressed as a ``TraceSet``: a
collection of ``FlowTrace`` objects, each with a 5-tuple, an RSS bucket, and
pre-computed ``(timestamps_ns, sizes_bytes)`` arrays spanning the full
simulation horizon.

Sources (selected by ``WorkloadConfig.source``):

* ``trace_mix``        : compose several synthetic kinds (web / cache /
                         hadoop / synthetic_rates) in one TraceSet.
* ``trace_csv``        : load real per-packet traces from CSV.
* ``synthetic_rates``  : one homogeneous TraceSet controlled by
                         num_flows / target_gbps / rate-dist / pkt-size dist /
                         burstiness.

CSV format for ``trace_csv``::

    flow_id,timestamp_ns,size_bytes[,src_ip,dst_ip,src_port,dst_port,proto]

Synthetic workload kinds approximate the per-flow character of the three
classes in Zhang et al., "High-resolution measurement of data center
microbursts", IMC 2017 (Meta web / cache / hadoop).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple
import csv

import numpy as np

from .hashing import Toeplitz


# ----------------------------------------------------------------------
# FlowTrace / TraceSet
# ----------------------------------------------------------------------
@dataclass
class FlowTrace:
    flow_id: int
    src_ip: int
    dst_ip: int
    src_port: int
    dst_port: int
    proto: int
    bucket_id: int
    timestamps_ns: np.ndarray   # int64, sorted
    sizes_bytes: np.ndarray     # int32
    _cursor: int = 0

    @property
    def n_packets(self) -> int:
        return int(self.timestamps_ns.size)


@dataclass
class TraceSet:
    flows: List[FlowTrace]
    horizon_ns: int
    num_buckets: int

    def reset_cursors(self) -> None:
        for f in self.flows:
            f._cursor = 0

    def generate_bin(self, t_start_ns: int, t_end_ns: int
                     ) -> Tuple[np.ndarray, np.ndarray]:
        """Per-bin aggregate (bucket_bytes, bucket_pkts) for telemetry."""
        bb = np.zeros(self.num_buckets, dtype=np.float64)
        bp = np.zeros(self.num_buckets, dtype=np.int64)
        for f in self.flows:
            if f._cursor >= f.n_packets:
                continue
            ts_remaining = f.timestamps_ns[f._cursor:]
            hi = int(np.searchsorted(ts_remaining, t_end_ns, side="left"))
            if hi == 0:
                continue
            sizes = f.sizes_bytes[f._cursor:f._cursor + hi]
            bb[f.bucket_id] += float(sizes.sum())
            bp[f.bucket_id] += hi
            f._cursor += hi
        return bb, bp

    def generate_bin_packets(self, t_start_ns: int, t_end_ns: int
                             ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Per-packet stream for the host pipeline: returns
        (timestamps_ns, sizes_bytes, bucket_ids) sorted by timestamp."""
        all_ts: List[np.ndarray] = []
        all_sz: List[np.ndarray] = []
        all_bk: List[np.ndarray] = []
        for f in self.flows:
            if f._cursor >= f.n_packets:
                continue
            ts_remaining = f.timestamps_ns[f._cursor:]
            hi = int(np.searchsorted(ts_remaining, t_end_ns, side="left"))
            if hi == 0:
                continue
            all_ts.append(f.timestamps_ns[f._cursor:f._cursor + hi])
            all_sz.append(f.sizes_bytes[f._cursor:f._cursor + hi])
            all_bk.append(np.full(hi, f.bucket_id, dtype=np.int64))
            f._cursor += hi
        if not all_ts:
            return (np.empty(0, dtype=np.int64),
                    np.empty(0, dtype=np.int32),
                    np.empty(0, dtype=np.int64))
        ts = np.concatenate(all_ts)
        sz = np.concatenate(all_sz)
        bk = np.concatenate(all_bk)
        order = np.argsort(ts, kind="stable")
        return ts[order], sz[order], bk[order]

    def rebucket(self, hasher: Toeplitz) -> None:
        """Recompute every flow's bucket_id from its 5-tuple (call after a
        pattern shift that mutates src_port)."""
        if not self.flows:
            return
        sip = np.array([f.src_ip for f in self.flows], dtype=np.uint32)
        dip = np.array([f.dst_ip for f in self.flows], dtype=np.uint32)
        sp = np.array([f.src_port for f in self.flows], dtype=np.uint16)
        dp = np.array([f.dst_port for f in self.flows], dtype=np.uint16)
        pr = np.array([f.proto for f in self.flows], dtype=np.uint8)
        b = hasher.bucket_of(sip, dip, sp, dp, pr, num_buckets=self.num_buckets)
        for f, bi in zip(self.flows, b):
            f.bucket_id = int(bi)


# ----------------------------------------------------------------------
# Per-flow timeline primitives
# ----------------------------------------------------------------------
def _cbr_timestamps(target_bps: float, avg_pkt_size: float,
                    t_start_ns: int, t_end_ns: int,
                    rng: np.random.Generator,
                    jitter_frac: float = 0.05) -> np.ndarray:
    if target_bps <= 0 or avg_pkt_size <= 0 or t_end_ns <= t_start_ns:
        return np.empty(0, dtype=np.int64)
    inter_ns = (avg_pkt_size * 8.0 / target_bps) * 1e9
    span = t_end_ns - t_start_ns
    n = max(0, int(span / inter_ns))
    if n == 0:
        return np.empty(0, dtype=np.int64)
    base = t_start_ns + (np.arange(n, dtype=np.float64) + 0.5) * inter_ns
    if jitter_frac > 0:
        base = base + rng.normal(0.0, jitter_frac * inter_ns, size=n)
    return np.sort(np.clip(base, t_start_ns, t_end_ns - 1)).astype(np.int64)


def _onoff_timestamps(target_bps: float, avg_pkt_size: float,
                      t_start_ns: int, t_end_ns: int,
                      mean_on_ns: float, mean_off_ns: float,
                      rng: np.random.Generator) -> np.ndarray:
    if target_bps <= 0 or avg_pkt_size <= 0 or t_end_ns <= t_start_ns:
        return np.empty(0, dtype=np.int64)
    on_frac = mean_on_ns / (mean_on_ns + mean_off_ns)
    peak_bps = target_bps / max(on_frac, 1e-3)
    chunks: List[np.ndarray] = []
    t = float(t_start_ns)
    state = 1
    while t < t_end_ns:
        if state == 1:
            dur = rng.exponential(mean_on_ns)
            end = min(t_end_ns, t + dur)
            if end > t:
                chunk = _cbr_timestamps(peak_bps, avg_pkt_size,
                                        int(t), int(end), rng,
                                        jitter_frac=0.02)
                if chunk.size > 0:
                    chunks.append(chunk)
            t = end
        else:
            dur = rng.exponential(mean_off_ns)
            t = t + dur
        state ^= 1
    if not chunks:
        return np.empty(0, dtype=np.int64)
    ts = np.concatenate(chunks)
    ts.sort()
    return ts


def _assign_sizes(n: int, pkt_size_distribution: str,
                  fixed_bytes: int,
                  imix_profile: Sequence[Sequence[float]],
                  rng: np.random.Generator) -> np.ndarray:
    if n == 0:
        return np.empty(0, dtype=np.int32)
    if pkt_size_distribution == "fixed":
        return np.full(n, int(fixed_bytes), dtype=np.int32)
    if pkt_size_distribution == "imix":
        sizes = np.array([row[0] for row in imix_profile], dtype=np.int32)
        probs = np.array([row[1] for row in imix_profile], dtype=np.float64)
        probs = probs / probs.sum()
        return rng.choice(sizes, size=n, p=probs).astype(np.int32)
    raise ValueError(f"unknown packet_size_distribution: {pkt_size_distribution}")


def _sample_per_flow_bps(n: int, total_bps: float, distribution: str,
                         zipf_s: float, hh_frac: float, hh_mult: float,
                         rng: np.random.Generator) -> np.ndarray:
    if n == 0:
        return np.empty(0, dtype=np.float64)
    if distribution == "uniform":
        w = np.ones(n, dtype=np.float64)
    elif distribution == "zipf":
        ranks = np.arange(1, n + 1, dtype=np.float64)
        w = 1.0 / (ranks ** zipf_s)
    elif distribution == "heavy_hitter":
        w = np.ones(n, dtype=np.float64)
        n_hot = max(1, int(hh_frac * n))
        idx = rng.choice(n, size=n_hot, replace=False)
        w[idx] = hh_mult
    else:
        raise ValueError(f"unknown flow_rate_distribution: {distribution}")
    return w / w.sum() * total_bps


def _assign_five_tuple(flow_id: int, rng: np.random.Generator,
                       proto: int) -> Tuple[int, int, int, int, int]:
    src_ip = 0x0A000000 | int(rng.integers(0, 0x00FFFFFF))
    dst_ip = 0xC0A80000 | int(rng.integers(0, 0x0000FFFF))
    src_port = int(1024 + (flow_id * 2654435761) % (65535 - 1024))
    dst_port = int(rng.integers(1024, 65535))
    return src_ip, dst_ip, src_port, dst_port, int(proto)


# ----------------------------------------------------------------------
# Synthetic workload kinds
# ----------------------------------------------------------------------
def _gen_synthetic_rates(n_flows: int, total_gbps: float, horizon_ns: int,
                         rng: np.random.Generator, wc,
                         ) -> List[Tuple[np.ndarray, np.ndarray]]:
    """General zipf/uniform/heavy-hitter + cbr/onoff + fixed/imix generator."""
    bps = _sample_per_flow_bps(
        n_flows, total_gbps * 1e9,
        wc.flow_rate_distribution, wc.zipf_s,
        wc.heavy_hitter_fraction, wc.heavy_hitter_multiplier, rng,
    )
    # Average pkt size used for inter-arrival timing; individual packets
    # are drawn from the configured size dist.
    if wc.packet_size_distribution == "fixed":
        avg_sz = float(wc.fixed_packet_bytes)
    else:
        sizes = np.array([row[0] for row in wc.imix_profile], dtype=np.float64)
        probs = np.array([row[1] for row in wc.imix_profile], dtype=np.float64)
        probs = probs / probs.sum()
        avg_sz = float((sizes * probs).sum())

    result: List[Tuple[np.ndarray, np.ndarray]] = []
    for i in range(n_flows):
        if wc.burstiness_model == "onoff":
            ts = _onoff_timestamps(bps[i], avg_sz, 0, horizon_ns,
                                   wc.onoff_mean_on_ns, wc.onoff_mean_off_ns,
                                   rng)
        else:
            ts = _cbr_timestamps(bps[i], avg_sz, 0, horizon_ns, rng)
        sz = _assign_sizes(ts.size, wc.packet_size_distribution,
                           wc.fixed_packet_bytes, wc.imix_profile, rng)
        result.append((ts, sz))
    return result


def _gen_meta_web(n_flows: int, total_gbps: float, horizon_ns: int,
                  rng: np.random.Generator
                  ) -> List[Tuple[np.ndarray, np.ndarray]]:
    """Web-RPC: short bursts of small-ish packets with idle periods
    between them. Long-run per-flow rate equals ``total_gbps / n_flows``.

    We model each flow as an on/off source with short on periods
    (typical RPC active window) and larger off periods (think time
    between requests). Packets within an on period use a lognormal
    size distribution around ~500 B.
    """
    per_flow_bps = (total_gbps * 1e9) / max(1, n_flows)
    avg_size = 500.0
    result: List[Tuple[np.ndarray, np.ndarray]] = []
    for _ in range(n_flows):
        mean_on = rng.uniform(50_000.0, 250_000.0)    # 50-250 us active
        mean_off = rng.uniform(200_000.0, 800_000.0)  # 0.2-0.8 ms idle
        ts = _onoff_timestamps(per_flow_bps, avg_size, 0, horizon_ns,
                               mean_on, mean_off, rng)
        sz = np.clip(rng.lognormal(6.0, 0.5, size=ts.size), 64, 1500
                     ).astype(np.int32)
        result.append((ts, sz))
    return result


def _gen_meta_cache(n_flows: int, total_gbps: float, horizon_ns: int,
                    rng: np.random.Generator
                    ) -> List[Tuple[np.ndarray, np.ndarray]]:
    """Cache (memcached-like): tiny packets, high-rate microbursts."""
    per_flow_bps = (total_gbps * 1e9) / max(1, n_flows)
    result: List[Tuple[np.ndarray, np.ndarray]] = []
    for _ in range(n_flows):
        ts = _onoff_timestamps(per_flow_bps, 96.0, 0, horizon_ns,
                               rng.uniform(5_000, 30_000),
                               rng.uniform(10_000, 100_000), rng)
        sz = rng.choice([64, 96, 128, 256], size=ts.size,
                        p=[0.55, 0.25, 0.15, 0.05]).astype(np.int32)
        result.append((ts, sz))
    return result


def _gen_meta_hadoop(n_flows: int, total_gbps: float, horizon_ns: int,
                     rng: np.random.Generator
                     ) -> List[Tuple[np.ndarray, np.ndarray]]:
    """Hadoop / bulk: long-lived, MTU-sized, sustained."""
    per_flow_bps = (total_gbps * 1e9) / max(1, n_flows)
    result: List[Tuple[np.ndarray, np.ndarray]] = []
    for _ in range(n_flows):
        ts = _cbr_timestamps(per_flow_bps, 1500.0, 0, horizon_ns, rng,
                             jitter_frac=0.02)
        sz = np.where(rng.random(ts.size) < 0.9, 1500, 128).astype(np.int32)
        result.append((ts, sz))
    return result


# ----------------------------------------------------------------------
# Public constructors
# ----------------------------------------------------------------------
def _materialise_flows(per_flow_pairs: List[Tuple[np.ndarray, np.ndarray]],
                       num_buckets: int, hasher: Toeplitz,
                       rng: np.random.Generator, proto: int,
                       starting_flow_id: int = 0) -> List[FlowTrace]:
    flows: List[FlowTrace] = []
    for i, (ts, sz) in enumerate(per_flow_pairs):
        fid = starting_flow_id + i
        sip, dip, sp, dp, pr = _assign_five_tuple(fid, rng, proto)
        bucket = int(hasher.bucket_of(
            np.array([sip], np.uint32),
            np.array([dip], np.uint32),
            np.array([sp], np.uint16),
            np.array([dp], np.uint16),
            np.array([pr], np.uint8),
            num_buckets=num_buckets,
        )[0])
        flows.append(FlowTrace(flow_id=fid, src_ip=sip, dst_ip=dip,
                               src_port=sp, dst_port=dp, proto=pr,
                               bucket_id=bucket,
                               timestamps_ns=ts, sizes_bytes=sz))
    return flows


def _dispatch_kind(kind: str, n_flows: int, total_gbps: float,
                   horizon_ns: int, rng: np.random.Generator,
                   wc) -> List[Tuple[np.ndarray, np.ndarray]]:
    if kind == "web":
        return _gen_meta_web(n_flows, total_gbps, horizon_ns, rng)
    if kind == "cache":
        return _gen_meta_cache(n_flows, total_gbps, horizon_ns, rng)
    if kind == "hadoop":
        return _gen_meta_hadoop(n_flows, total_gbps, horizon_ns, rng)
    if kind == "synthetic_rates":
        return _gen_synthetic_rates(n_flows, total_gbps, horizon_ns, rng, wc)
    raise ValueError(f"unknown workload kind: {kind}")


def build_traceset_from_workload(wc, horizon_ns: int, num_buckets: int,
                                  hasher: Toeplitz, rng: np.random.Generator,
                                  domain: str) -> TraceSet:
    """Build the TraceSet for one domain (`stateless` | `stateful`) based
    on the full WorkloadConfig."""
    proto = 6 if domain == "stateful" else 17

    source = wc.source
    if source == "trace_csv":
        path = (wc.trace_file_stateful if domain == "stateful"
                else wc.trace_file_stateless)
        if not path:
            raise ValueError(f"workload.source=trace_csv but no trace_file_"
                             f"{domain} provided")
        return load_trace_csv(path, horizon_ns, num_buckets, hasher, rng,
                              default_proto=proto)

    if source == "trace_mix":
        mix = (wc.trace_mix_stateful if domain == "stateful"
               else wc.trace_mix_stateless)
        flows: List[FlowTrace] = []
        fid = 0
        for spec in mix:
            pairs = _dispatch_kind(
                kind=spec["kind"],
                n_flows=int(spec["n_flows"]),
                total_gbps=float(spec["gbps"]),
                horizon_ns=horizon_ns, rng=rng, wc=wc,
            )
            flows.extend(_materialise_flows(pairs, num_buckets, hasher, rng,
                                            proto, starting_flow_id=fid))
            fid += len(pairs)
        return TraceSet(flows=flows, horizon_ns=horizon_ns,
                        num_buckets=num_buckets)

    if source == "synthetic_rates":
        n = wc.num_flows_stateful if domain == "stateful" else wc.num_flows_stateless
        g = wc.stateful_target_gbps if domain == "stateful" else wc.stateless_target_gbps
        pairs = _gen_synthetic_rates(n, g, horizon_ns, rng, wc)
        flows = _materialise_flows(pairs, num_buckets, hasher, rng, proto)
        return TraceSet(flows=flows, horizon_ns=horizon_ns,
                        num_buckets=num_buckets)

    raise ValueError(f"unknown workload.source: {source}")


def load_trace_csv(path: str, horizon_ns: int, num_buckets: int,
                   hasher: Toeplitz, rng: np.random.Generator,
                   default_proto: int = 17) -> TraceSet:
    per_flow: Dict[int, List[Tuple[int, int]]] = {}
    meta: Dict[int, Tuple[int, int, int, int, int]] = {}
    with open(path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            fid = int(row["flow_id"])
            ts_ns = int(row["timestamp_ns"])
            if ts_ns >= horizon_ns:
                continue
            sz = int(row["size_bytes"])
            per_flow.setdefault(fid, []).append((ts_ns, sz))
            if fid not in meta:
                if row.get("src_ip"):
                    meta[fid] = (int(row["src_ip"]), int(row["dst_ip"]),
                                 int(row["src_port"]), int(row["dst_port"]),
                                 int(row.get("proto") or default_proto))
                else:
                    meta[fid] = _assign_five_tuple(fid, rng, default_proto)

    flows: List[FlowTrace] = []
    for fid, pairs in per_flow.items():
        pairs.sort()
        ts = np.array([p[0] for p in pairs], dtype=np.int64)
        sz = np.array([p[1] for p in pairs], dtype=np.int32)
        sip, dip, sp, dp, pr = meta[fid]
        bucket = int(hasher.bucket_of(
            np.array([sip], np.uint32),
            np.array([dip], np.uint32),
            np.array([sp], np.uint16),
            np.array([dp], np.uint16),
            np.array([pr], np.uint8),
            num_buckets=num_buckets,
        )[0])
        flows.append(FlowTrace(flow_id=fid, src_ip=sip, dst_ip=dip,
                               src_port=sp, dst_port=dp, proto=pr,
                               bucket_id=bucket, timestamps_ns=ts,
                               sizes_bytes=sz))
    return TraceSet(flows=flows, horizon_ns=horizon_ns, num_buckets=num_buckets)
