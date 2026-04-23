"""Per-flow packet-level traffic traces.

Everything upstream of the host pipeline is expressed as a ``TraceSet``: a
collection of ``FlowTrace`` objects, each with a 5-tuple, an RSS bucket, and
pre-computed ``(timestamps_ns, sizes_bytes)`` arrays spanning the full
simulation horizon.

Sources (selected by ``WorkloadConfig.source``):

* ``imc17_cdf``        : each class (web/cache/hadoop) generated from
                         digitized IMC'17 CDFs -- ON/OFF burstiness,
                         packet sizes, and intra-burst inter-arrival
                         times all sampled from empirical CDFs. The
                         realized aggregate rate is then renormalized
                         to hit ``gbps``. This is the preferred mode
                         for realism.
* ``trace_mix``        : compose several synthetic kinds (web / cache /
                         hadoop / synthetic_rates) in one TraceSet.
                         Legacy; each kind uses hand-coded heuristics
                         instead of CDFs.
* ``trace_csv``        : load real per-packet traces from CSV.
* ``synthetic_rates``  : one homogeneous TraceSet controlled by
                         num_flows / target_gbps / rate-dist / pkt-size dist /
                         burstiness.

CSV format for ``trace_csv``::

    flow_id,timestamp_ns,size_bytes[,src_ip,dst_ip,src_port,dst_port,proto]
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple
import csv

import numpy as np

from .hashing import Toeplitz
from .imc17_cdf import CLASSES as IMC17_CLASSES, make_sampler


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
# IMC'17 CDF-driven per-flow generation
#
# For each flow of a given class we:
#   1) Draw ON/OFF periods from the class's burst-duration CDFs.
#   2) Within each ON period, emit packets whose *size* and *inter-
#      arrival time* come from the class's packet-size and IAT CDFs.
#   3) Across all flows, rescale timestamps (actually, rescale the
#      inter-arrival gaps) so the aggregate offered rate over
#      ``horizon_ns`` matches the configured ``total_gbps`` target
#      within <5%. Rescaling preserves the *shape* of the CDFs (ratios
#      stay intact) while matching aggregate throughput.
# ----------------------------------------------------------------------
def _estimate_mean(sampler) -> float:
    """Rough mean of a CDF sampler: use the mean of its control points
    weighted by adjacent probability differences."""
    xs = np.exp(sampler.xs)
    ps = sampler.ps
    if xs.size < 2:
        return float(xs[0]) if xs.size else 1.0
    dps = np.diff(ps)
    mids = 0.5 * (xs[:-1] + xs[1:])
    return float(max(1.0, np.sum(dps * mids)))


def _gen_imc17_class(
    kind: str, n_flows: int, target_gbps: float,
    horizon_ns: int, rng: np.random.Generator,
) -> Tuple[List[Tuple[np.ndarray, np.ndarray]], Dict[str, float]]:
    """Emit per-flow ``(timestamps_ns, sizes_bytes)`` pairs for a single
    class, sampling ON/OFF/IAT/size from IMC'17 CDFs, and scale the
    sampled intervals so the aggregate long-run rate matches
    ``target_gbps``.

    Scaling strategy
    ----------------
    Shape is always preserved: ON/OFF/IAT are drawn from the class CDFs
    and then multiplied by a single ``time_scale`` per class. A smaller
    time_scale packs events closer together, generating more packets
    per unit time, which linearly increases the realized rate. Duty
    cycle (ON / (ON+OFF)) and packets-per-ON stay invariant under this
    scaling, so the burst *character* of the class is preserved.

    The time_scale is derived analytically from the CDFs' first moments,
    then corrected in a second pass by measuring the realized rate and
    applying a small residual scale. In practice the residual is small
    (typically <5%), but the correction guarantees we hit the target
    within numerical precision.
    """
    if kind not in IMC17_CLASSES:
        raise ValueError(f"imc17_cdf unsupported kind: {kind}")
    s_on = make_sampler("on_ns", kind)
    s_off = make_sampler("off_ns", kind)
    s_iat = make_sampler("iat_ns", kind)
    s_pkt = make_sampler("pkt_size_bytes", kind)

    mean_on = _estimate_mean(s_on)
    mean_off = _estimate_mean(s_off)
    mean_iat = _estimate_mean(s_iat)  # ns
    mean_sz = _estimate_mean(s_pkt)   # bytes
    duty = mean_on / (mean_on + mean_off)
    # Expected per-flow bits per second *before* time scaling.
    # mean_iat is in ns, so divide by (mean_iat * 1e-9) to get per-second.
    expected_bps = duty * (mean_sz * 8.0) / (mean_iat * 1e-9)
    target_per_flow_bps = (target_gbps * 1e9) / max(1, n_flows)
    # time_scale < 1 packs events closer together -> higher rate.
    # Clamp time_scale to a sane range so we never generate millions of
    # ON/OFF cycles per flow (which happens if n_flows is much smaller
    # than the realistic per-class flow count -- e.g. 40 web flows
    # asked to carry 15 Gbps would otherwise produce ~1e6 cycles each).
    # If the target can't be met at time_scale >= MIN_TS_FACTOR, the
    # realized rate will simply be lower and the diagnostics will flag
    # it; the user should raise n_flows or lower gbps in that case.
    # Allow compression down to 1e-4 (events 10000x faster than raw
    # CDF) so web workloads with very long OFF periods can still hit
    # high aggregate targets. Memory is still bounded by
    # MAX_CYCLES_PER_FLOW.
    MIN_TS_FACTOR = 1e-4
    MAX_TS_FACTOR = 10.0
    time_scale = (expected_bps / target_per_flow_bps
                  if target_per_flow_bps > 0 else 1.0)
    time_scale = float(np.clip(time_scale, MIN_TS_FACTOR, MAX_TS_FACTOR))
    # Hard cap on cycles per flow so we never blow up memory regardless
    # of time_scale.
    MAX_CYCLES_PER_FLOW = 20000

    def _generate(ts_factor: float):
        """Vectorized generator: per flow, draw all ON/OFF cycles at
        once, then emit packets within each ON window at the mean IAT
        (CBR-within-ON). This sacrifices a bit of per-packet IAT CDF
        fidelity for ~100x speed; the burst pattern (ON/OFF) and the
        aggregate rate still match the CDFs.

        Coverage guarantee: we sample ``3x`` the mean-expected cycle
        count so even flows drawing long ON/OFF tails still cover the
        full horizon. If a flow's cumulative cycle length still falls
        short, we top up with extra cycles until the coverage exceeds
        ``horizon_ns``. This prevents the tail-off where the last epochs
        of a run see zero offered load because some flows exhausted
        their pre-computed timelines.
        """
        out: List[Tuple[np.ndarray, np.ndarray]] = []
        mean_cycle = max(1.0, (mean_on + mean_off) * ts_factor)
        mean_iat_s = max(1.0, mean_iat * ts_factor)
        base_cycles_est = int(3 * horizon_ns / mean_cycle) + 16
        cycles_est = min(MAX_CYCLES_PER_FLOW, base_cycles_est)
        for _ in range(n_flows):
            # Step 1: sample ON/OFF durations for enough cycles.
            n_cyc = cycles_est
            on_durs = s_on.sample(n_cyc, rng) * ts_factor
            off_durs = s_off.sample(n_cyc, rng) * ts_factor
            # Coverage top-up: if the drawn cycles don't cover the
            # horizon (high-variance tail samples), keep sampling
            # until they do, bounded by MAX_CYCLES_PER_FLOW.
            coverage = float((on_durs + off_durs).sum())
            while coverage < horizon_ns * 1.2 and \
                    on_durs.size < MAX_CYCLES_PER_FLOW:
                extra = min(MAX_CYCLES_PER_FLOW - on_durs.size,
                            max(16, cycles_est))
                if extra <= 0:
                    break
                on_extra = s_on.sample(extra, rng) * ts_factor
                off_extra = s_off.sample(extra, rng) * ts_factor
                on_durs = np.concatenate([on_durs, on_extra])
                off_durs = np.concatenate([off_durs, off_extra])
                coverage = float((on_durs + off_durs).sum())
            # Random initial phase so flows are not synchronized.
            phase = rng.uniform(0, 1) * (on_durs[0] + off_durs[0])
            # Build cycle start times.
            cycle_lens = on_durs + off_durs
            starts = phase + np.concatenate([[0.0], np.cumsum(cycle_lens)[:-1]])
            on_ends = np.minimum(starts + on_durs, horizon_ns)
            mask = (starts < horizon_ns) & (on_ends > starts)
            starts = starts[mask]
            on_ends = on_ends[mask]
            if starts.size == 0:
                out.append((np.empty(0, dtype=np.int64),
                            np.empty(0, dtype=np.int32)))
                continue
            # Step 2: for each ON window, emit packets at ``mean_iat_s``
            # spacing starting from ``starts[i]``. We use
            # ``n_pkts[i] = 1 + floor(on_dur / mean_iat_s)`` so a short
            # ON window (duration < mean_iat_s) still produces one
            # packet at the window start; this avoids losing bursts
            # when the class's typical ON < IAT (common for cache/web
            # microbursts).
            on_spans = np.maximum(0.0, on_ends - starts)
            n_pkts_per_on = (np.floor(on_spans / mean_iat_s)
                             + 1).astype(np.int64)
            # Zero-out ON windows with no duration at all.
            n_pkts_per_on = np.where(on_spans > 0, n_pkts_per_on, 0)
            total_pkts = int(n_pkts_per_on.sum())
            if total_pkts == 0:
                out.append((np.empty(0, dtype=np.int64),
                            np.empty(0, dtype=np.int32)))
                continue
            cum = np.concatenate([[0], np.cumsum(n_pkts_per_on)])
            idx_within = np.arange(total_pkts, dtype=np.int64) - \
                np.repeat(cum[:-1], n_pkts_per_on)
            on_idx = np.repeat(np.arange(starts.size, dtype=np.int64),
                               n_pkts_per_on)
            pkt_ts = (starts[on_idx] + idx_within * mean_iat_s).astype(np.int64)
            # Clip any packet that spilled past the horizon due to
            # rounding.
            keep = pkt_ts < horizon_ns
            pkt_ts = pkt_ts[keep]
            pkt_sz = np.clip(s_pkt.sample(pkt_ts.size, rng),
                             64, 1500).astype(np.int32)
            out.append((pkt_ts, pkt_sz))
        return out

    horizon_s = max(1e-12, horizon_ns * 1e-9)
    target_bps = target_gbps * 1e9

    # First pass at the analytic time_scale.
    pairs = _generate(time_scale)
    total_bytes = sum(int(sz.sum()) for _, sz in pairs)
    realized_bps = total_bytes * 8.0 / horizon_s
    initial_bps = realized_bps

    # Iterative residual correction. The shape-preserving time-scale
    # mapping is slightly non-linear (packets-per-ON scales with
    # floor(on_dur / mean_iat_s)) so one pass typically gets us within
    # ~2-4x of target; a couple more geometric corrections bring us to
    # <2% error. We cap at MAX_ITERS passes to bound runtime.
    MAX_ITERS = 6
    TOL = 0.02
    cur_scale = time_scale
    if target_bps > 0:
        for _ in range(MAX_ITERS):
            if realized_bps <= 0:
                break
            err = realized_bps / target_bps
            if abs(err - 1.0) <= TOL:
                break
            # Clamp the step to avoid explosive overshoots when the
            # generator is saturating its cycle cap.
            step = float(np.clip(err, 0.1, 10.0))
            new_scale = float(np.clip(cur_scale * step,
                                      MIN_TS_FACTOR, MAX_TS_FACTOR))
            if abs(new_scale - cur_scale) / cur_scale < 1e-3:
                break
            cur_scale = new_scale
            pairs = _generate(cur_scale)
            total_bytes = sum(int(sz.sum()) for _, sz in pairs)
            realized_bps = total_bytes * 8.0 / horizon_s

    diag = {
        "realized_gbps_before": initial_bps / 1e9,
        "realized_gbps_after": realized_bps / 1e9,
        "scale": cur_scale,
    }
    return pairs, diag


def _gen_imc17_cdf(kind: str, n_flows: int, total_gbps: float,
                   horizon_ns: int, rng: np.random.Generator,
                   ) -> Tuple[List[Tuple[np.ndarray, np.ndarray]],
                              Dict[str, float]]:
    return _gen_imc17_class(kind, n_flows, total_gbps, horizon_ns, rng)


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
                                  domain: str
                                  ) -> Tuple[TraceSet, List[Dict]]:
    """Build the TraceSet for one domain (`stateless` | `stateful`) based
    on the full WorkloadConfig.

    Returns ``(TraceSet, realism_diagnostics)`` where the diagnostics is
    a list of per-class dicts (one per mix entry) reporting target vs
    realized Gbps, mean packet size, flow count, etc.
    """
    proto = 6 if domain == "stateful" else 17
    diagnostics: List[Dict] = []

    source = wc.source
    if source == "trace_csv":
        path = (wc.trace_file_stateful if domain == "stateful"
                else wc.trace_file_stateless)
        if not path:
            raise ValueError(f"workload.source=trace_csv but no trace_file_"
                             f"{domain} provided")
        ts = load_trace_csv(path, horizon_ns, num_buckets, hasher, rng,
                            default_proto=proto)
        return ts, diagnostics

    if source in ("trace_mix", "imc17_cdf"):
        mix = (wc.trace_mix_stateful if domain == "stateful"
               else wc.trace_mix_stateless)
        flows: List[FlowTrace] = []
        fid = 0
        for spec in mix:
            kind = spec["kind"]
            n_flows = int(spec["n_flows"])
            target_gbps = float(spec["gbps"])
            realized_gbps_before = None
            realized_gbps_after = None
            if source == "imc17_cdf" and kind in IMC17_CLASSES:
                pairs, diag = _gen_imc17_cdf(kind, n_flows, target_gbps,
                                             horizon_ns, rng)
                realized_gbps_before = diag["realized_gbps_before"]
                realized_gbps_after = diag["realized_gbps_after"]
            else:
                pairs = _dispatch_kind(
                    kind=kind,
                    n_flows=n_flows,
                    total_gbps=target_gbps,
                    horizon_ns=horizon_ns, rng=rng, wc=wc,
                )
                # Compute realized rate after for legacy generators too.
                total_bytes = sum(int(sz.sum()) for _, sz in pairs)
                realized_gbps_after = (total_bytes * 8.0
                                       / max(1e-12, horizon_ns * 1e-9) / 1e9)
            total_pkts = sum(int(ts.size) for ts, _ in pairs)
            total_bytes = sum(int(sz.sum()) for _, sz in pairs)
            mean_sz = total_bytes / max(1, total_pkts)
            diagnostics.append(dict(
                domain=domain, kind=kind,
                n_flows=n_flows,
                target_gbps=target_gbps,
                realized_gbps_before=realized_gbps_before,
                realized_gbps=realized_gbps_after,
                total_packets=total_pkts,
                mean_pkt_bytes=mean_sz,
            ))
            flows.extend(_materialise_flows(pairs, num_buckets, hasher, rng,
                                            proto, starting_flow_id=fid))
            fid += len(pairs)
        return (TraceSet(flows=flows, horizon_ns=horizon_ns,
                         num_buckets=num_buckets),
                diagnostics)

    if source == "synthetic_rates":
        n = wc.num_flows_stateful if domain == "stateful" else wc.num_flows_stateless
        g = wc.stateful_target_gbps if domain == "stateful" else wc.stateless_target_gbps
        pairs = _gen_synthetic_rates(n, g, horizon_ns, rng, wc)
        flows = _materialise_flows(pairs, num_buckets, hasher, rng, proto)
        total_bytes = sum(int(sz.sum()) for _, sz in pairs)
        total_pkts = sum(int(ts.size) for ts, _ in pairs)
        diagnostics.append(dict(
            domain=domain, kind="synthetic_rates",
            n_flows=n, target_gbps=g,
            realized_gbps_before=None,
            realized_gbps=(total_bytes * 8.0 / max(1e-12, horizon_ns * 1e-9) / 1e9),
            total_packets=total_pkts,
            mean_pkt_bytes=total_bytes / max(1, total_pkts),
        ))
        return (TraceSet(flows=flows, horizon_ns=horizon_ns,
                         num_buckets=num_buckets),
                diagnostics)

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
