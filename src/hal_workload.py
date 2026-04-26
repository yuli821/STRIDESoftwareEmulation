"""HAL-ISCA'24 composite workload generator (two-layer bursty traffic).

This module implements the bursty SNIC workload used by

    Huang, Lou, Vanavasam, Kong, Ji, Jeong, Zhuo, Lee, Kim.
    "HAL: Hardware-assisted Load Balancing for Energy-efficient
     SNIC-Host Cooperative Computing."
    Proc. 51st ACM/IEEE Int'l Symp. on Computer Architecture
    (ISCA 2024), pp. 613-627, Fig. 8.

HAL fits a *log-normal* distribution to the per-workload link-utilization
CDFs reported by Roy et al. (SIGCOMM'15) for three Meta clusters, and
uses those fits as a bursty packet-rate generator driving their SNIC
evaluation:

    class    mu       sigma   reported avg (Gbps)   source
    web     -1.37     1.97     1.6                   HAL Fig. 8
    cache   -9.00     7.55     5.2                   HAL Fig. 8
    hadoop  -4.18     6.56    10.9                   HAL Fig. 8

mu/sigma give very heavy-tailed bursty rate processes; HAL *clips*
samples to the link rate (100 Gbps in their BF-2 setup), which is what
reproduces the reported averages. Without clipping, the raw lognormal
means are astronomical (``exp(19.5)`` for cache, for example).

Two-layer workload
------------------

Layer 1 (epoch-level aggregate rate process). Every
``hal_rate_update_ns`` simulation-time, for every enabled class,
we draw a fresh rate ``R_c`` from the class's clipped lognormal.
Rates are then scaled so that the long-run time-average of ``R_c``
equals ``mix_c * hal_total_gbps``; this lets the user tune both
the aggregate offered load and the per-class mix while preserving
the published burstiness shape (same µ, σ, same clip threshold).

Layer 2 (persistent flow table). Each new flow is sampled with a
traffic class, flow size, sending rate, duration, 5-tuple, and
RSS bucket. Once created, the flow stays in an active-flow table
and continues sending MTU-sized packets at its sampled rate until
its byte budget is exhausted OR its sampled duration expires. At
every simulation epoch, new flows are spawned for each class
until the class's currently-active aggregate sending rate matches
the Layer-1 target rate for that epoch.

Per-class flow characteristics
------------------------------

Flow-size, per-flow-rate, and duration distributions match the Meta
per-cluster characterization in Roy et al. (SIGCOMM'15, Fig. 9) and
the microburst / duty-cycle description in Zhang et al. (IMC 2017):

* **web** -- small request/response flows. Bimodal mixture with
  the bulk around a few KB and a smaller fraction at tens of KB.
* **cache** -- longer-lived medium+large flows. Bimodal mixture
  with many flows at tens of KB and a substantial fraction at
  MB-scale sizes.
* **hadoop** -- mice/elephant trimodal. Matches Roy et al. Fig. 9:
  about 70 percent of flows are below 10 KB with median below
  1 KB, and a small (~5 percent) fraction are multi-MB elephants.

Per-flow sending rates are class-specific constants based on the
typical throughput per flow in each cluster (small web flows are
low-rate, bulk hadoop flows saturate per-queue line rate). Durations
are exponential with a class-specific mean; a flow ends at whichever
of ``bytes_exhausted`` or ``duration`` triggers first.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

import numpy as np


# --------------------------------------------------------------------------
# HAL paper (Huang et al. ISCA'24, Fig. 8) class rate lognormal parameters.
# --------------------------------------------------------------------------
HAL_LOGNORMAL_PARAMS: Dict[str, Tuple[float, float]] = {
    "web":    (-1.37, 1.97),
    "cache":  (-9.00, 7.55),
    "hadoop": (-4.18, 6.56),
}

HAL_CLASSES: Tuple[str, ...] = ("web", "cache", "hadoop")


# --------------------------------------------------------------------------
# Per-class flow-size, rate, and duration distributions.
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class _LogNormalMixtureMode:
    weight: float
    mu: float
    sigma: float


@dataclass(frozen=True)
class LogNormalMixture:
    """Sample from a weighted mixture of lognormals."""
    modes: Sequence[_LogNormalMixtureMode]

    def __post_init__(self) -> None:
        total_w = sum(m.weight for m in self.modes)
        if not 0.999 <= total_w <= 1.001:
            raise ValueError(
                f"mixture weights must sum to 1, got {total_w:.4f}")

    def sample(self, n: int, rng: np.random.Generator) -> np.ndarray:
        if n <= 0:
            return np.empty(0, dtype=np.float64)
        weights = np.array([m.weight for m in self.modes])
        which = rng.choice(len(self.modes), size=int(n), p=weights)
        out = np.empty(int(n), dtype=np.float64)
        for i, m in enumerate(self.modes):
            mask = (which == i)
            k = int(mask.sum())
            if k > 0:
                out[mask] = rng.lognormal(m.mu, m.sigma, size=k)
        return out


# Flow size (bytes) distributions. Justification in module docstring.
#
# web: 80% around 3 KB, 20% around 30 KB. P(size<10KB) ~ 0.95.
# cache: 60% around 30 KB, 40% around 2 MB.
# hadoop: trimodal mice/elephant; 70% mice at median ~500 B, 25%
# medium ~50 KB, 5% elephants ~10 MB. P(size<10KB) = 0.70, median =
# 500 B, matching Roy et al. SIGCOMM'15 Fig. 9. Elephant mode is
# bounded at ~10 MB (rather than unbounded tail) so individual
# elephants don't dominate the realised per-class rate over the
# simulation horizon. Bounded heavy-tailed flow sizes are standard
# practice in datacenter flow-scheduling simulation (pFabric
# SIGCOMM'13; Homa SIGCOMM'18).
_WEB_SIZE = LogNormalMixture([
    _LogNormalMixtureMode(weight=0.80, mu=float(np.log(3_000)),  sigma=0.55),
    _LogNormalMixtureMode(weight=0.20, mu=float(np.log(30_000)), sigma=0.40),
])
_CACHE_SIZE = LogNormalMixture([
    _LogNormalMixtureMode(weight=0.60, mu=float(np.log(30_000)),    sigma=0.70),
    _LogNormalMixtureMode(weight=0.40, mu=float(np.log(1_000_000)), sigma=0.60),
])
_HADOOP_SIZE = LogNormalMixture([
    _LogNormalMixtureMode(weight=0.70, mu=float(np.log(500)),        sigma=0.60),
    _LogNormalMixtureMode(weight=0.25, mu=float(np.log(50_000)),     sigma=0.50),
    _LogNormalMixtureMode(weight=0.05, mu=float(np.log(10_000_000)), sigma=0.60),
])

CLASS_SIZE_SAMPLER: Dict[str, LogNormalMixture] = {
    "web":    _WEB_SIZE,
    "cache":  _CACHE_SIZE,
    "hadoop": _HADOOP_SIZE,
}


# Per-flow sending rate (Gbps). Class-typical per-flow throughput
# grounded in the SIGCOMM'15 Meta measurement study (Roy et al.) and
# IMC'09 / IMC'10 datacenter traffic characterizations:
#
#   * Kandula, Sengupta, Greenberg, Patel, Chaiken. "The Nature of Data
#     Center Traffic", IMC 2009: each host sees ~1000 concurrent TCP
#     connections on a 10 Gbps NIC, so any single flow's sustained rate
#     is a small fraction of line rate.
#   * Benson, Akella, Maltz. "Network Traffic Characteristics of Data
#     Centers in the Wild", IMC 2010: 10k+ simultaneous flows at edge
#     switches, confirming heavy fan-in.
#   * Roy, Zeng, Bagga, Porter, Snoeren. "Inside the Social Network's
#     (Datacenter) Network", SIGCOMM 2015, Fig. 9-10: per-flow
#     throughput distribution in the Meta Hadoop/cache/web clusters is
#     heavy-tailed with mass concentrated *well below* 1 Gbps; flows
#     sustaining multi-Gbps are rare. Hadoop racks receive traffic from
#     hundreds to thousands of sources simultaneously during shuffle.
#
# Chosen values:
#   - web: 0.25 Gbps. Web RPC front-end flows are small request/response
#     pairs bounded by RPC RTT; per-flow bandwidth is a small fraction
#     of a Gbps.
#   - cache: 0.5 Gbps. Meta memcached flows are many small KV reads;
#     per-flow sustained rate is sub-Gbps.
#   - hadoop: 2.0 Gbps. Upper end of the typical sustained TCP rate for
#     an individual Hadoop shuffle flow on a 10 Gbps NIC, chosen so
#     each flow has a meaningful footprint on a single host queue
#     (~17% of the 12 Gbps per-queue drain rate in the TEST config)
#     while still leaving per-class concurrency at ~16 flows
#     (= 32.7 Gbps / 2 Gbps) well above the 4-queue RSS fan-out.
#
# These values are also *compatible* with the Layer-1 HAL fair-share
# throttling loop: smaller per-flow rates => more concurrent flows per
# aggregate target, filling the RSS indirection table more uniformly
# and letting the scheduler's per-queue pressure signal actually track
# the workload. Historical note: an earlier draft used
# (web 1.0, cache 4.0, hadoop 8.0) for faster convergence, but that was
# not grounded in the measurement papers above.
CLASS_PER_FLOW_RATE_GBPS: Dict[str, float] = {
    "web":    0.25,
    "cache":  0.5,
    "hadoop": 2.0,
}


# Per-flow duration mean (seconds) -- exponential.
# These are large relative to typical size/rate lifetimes so size
# exhaustion dominates; duration is a safety backstop for degenerate
# cases (very small flows with very small rates).
CLASS_DURATION_MEAN_S: Dict[str, float] = {
    "web":    0.5,
    "cache":  2.0,
    "hadoop": 10.0,
}


def _solve_scale_for_target(
    mu: float, sigma: float, clip_gbps: float, target_gbps: float,
    n_samples: int = 200_000, seed: int = 17,
) -> float:
    """Return ``s`` such that ``E[min(s * lognormal(mu, sigma), clip)]
    == target_gbps``.

    Direct rescale-then-clip underestimates the target because mass
    above ``clip/s`` is folded down. We solve for ``s`` by bisection
    on a fixed Monte-Carlo sample, giving the exact post-clip mean.
    """
    if clip_gbps <= 0 or target_gbps <= 0:
        return 0.0
    rng = np.random.default_rng(seed)
    X = rng.lognormal(mu, sigma, size=int(n_samples))
    if target_gbps >= clip_gbps:
        # No finite s gives mean >= clip; everything is clipped.
        return float("inf")
    lo, hi = 0.0, 1.0
    # Expand hi until achievable mean exceeds target.
    for _ in range(200):
        if np.minimum(hi * X, clip_gbps).mean() >= target_gbps:
            break
        hi *= 2.0
        if hi > 1e24:
            return float("inf")
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        if np.minimum(mid * X, clip_gbps).mean() < target_gbps:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def sample_class_rate_timeline(
    traffic_class: str,
    horizon_ns: int,
    rate_update_ns: int,
    target_mean_gbps: float,
    clip_gbps: float,
    rng: np.random.Generator,
    n_tenants: int = 1,
) -> np.ndarray:
    """Return a per-update-step rate (Gbps) for a single class.

    Samples from the HAL paper's lognormal, rescales and clips so the
    long-run time-average of ``min(s*X, clip_gbps)`` equals
    ``target_mean_gbps``. This preserves the burstiness shape (same µ,
    σ, same clip threshold) while exactly tuning the per-class offered
    load.

    When ``n_tenants > 1``, the returned timeline is the SUM of
    ``n_tenants`` statistically independent per-class lognormal
    timelines, each individually scaled + clipped to have mean
    ``target_mean_gbps / n_tenants``. This models
    ``n_tenants`` co-resident application tenants each emitting
    HAL-native traffic, and reduces coincident-burst probability via
    the usual central-limit variance reduction (std(sum) = std * sqrt
    N rather than std * N).
    """
    if traffic_class not in HAL_LOGNORMAL_PARAMS:
        raise KeyError(
            f"hal_composite: unknown traffic_class {traffic_class!r}; "
            f"available: {HAL_CLASSES}")
    mu, sigma = HAL_LOGNORMAL_PARAMS[traffic_class]
    n_updates = int(np.ceil(horizon_ns / max(1, rate_update_ns)))
    if n_updates <= 0 or target_mean_gbps <= 0:
        return np.zeros(max(1, n_updates), dtype=np.float64)
    n_tenants = max(1, int(n_tenants))

    per_tenant_target = target_mean_gbps / n_tenants
    s = _solve_scale_for_target(mu, sigma, clip_gbps, per_tenant_target)
    if not np.isfinite(s):
        # Each tenant's target >= clip; sum is n_tenants * clip.
        return np.full(n_updates, n_tenants * clip_gbps, dtype=np.float64)

    total = np.zeros(n_updates, dtype=np.float64)
    for _ in range(n_tenants):
        raw = rng.lognormal(mu, sigma, size=n_updates) * s
        np.minimum(raw, clip_gbps, out=raw)
        total += raw
    return total


# --------------------------------------------------------------------------
# Per-flow attributes -- sampled once at spawn.
# --------------------------------------------------------------------------
@dataclass
class _Flow:
    flow_id: int
    traffic_class: str
    start_ns: int
    rate_bps: float
    total_bytes: int        # nominal size sample
    duration_ns: int        # nominal duration sample
    end_ns: int
    # Actual per-window emission after fair-share throttling. Filled
    # in by the spawn loop. Each entry: (window_start_ns, window_end_ns,
    # bytes_emitted).
    window_emissions: List[Tuple[int, int, float]] = None


class _AttrPool:
    """Batched per-class pool of pre-sampled flow attributes.

    Flow sampling is done vectorised in batches of ``batch`` and
    popped one-at-a-time from the cached arrays. This keeps
    per-spawn cost in pure Python while preserving the per-class
    distributions.
    """
    __slots__ = ("traffic_class", "per_flow_rate_cap_gbps",
                 "mtu_bytes", "rng", "batch", "_idx",
                 "_sizes", "_rates_bps", "_durations_ns")

    def __init__(self, traffic_class: str, per_flow_rate_cap_gbps: float,
                 mtu_bytes: int, rng: np.random.Generator,
                 batch: int = 4096):
        self.traffic_class = traffic_class
        self.per_flow_rate_cap_gbps = per_flow_rate_cap_gbps
        self.mtu_bytes = mtu_bytes
        self.rng = rng
        self.batch = int(batch)
        self._idx = 0
        self._sizes: np.ndarray = np.empty(0, dtype=np.int64)
        self._rates_bps: np.ndarray = np.empty(0, dtype=np.float64)
        self._durations_ns: np.ndarray = np.empty(0, dtype=np.int64)

    def _refill(self) -> None:
        n = self.batch
        c = self.traffic_class
        sizes = CLASS_SIZE_SAMPLER[c].sample(n, self.rng)
        sizes = np.maximum(self.mtu_bytes // 2, np.round(sizes)).astype(np.int64)
        class_rate = CLASS_PER_FLOW_RATE_GBPS[c]
        rate_gbps = self.rng.lognormal(np.log(class_rate), 0.3, size=n)
        np.minimum(rate_gbps, float(self.per_flow_rate_cap_gbps), out=rate_gbps)
        np.maximum(rate_gbps, 0.01, out=rate_gbps)
        rates_bps = rate_gbps * 1e9
        durations_s = self.rng.exponential(
            CLASS_DURATION_MEAN_S[c], size=n)
        durations_ns = np.maximum(1, (durations_s * 1e9).astype(np.int64))
        self._sizes = sizes
        self._rates_bps = rates_bps
        self._durations_ns = durations_ns
        self._idx = 0

    def pop(self) -> Tuple[int, float, int]:
        if self._idx >= self._sizes.size:
            self._refill()
        i = self._idx
        self._idx += 1
        return (int(self._sizes[i]),
                float(self._rates_bps[i]),
                int(self._durations_ns[i]))


def _sample_flow_attrs(
    pool: "_AttrPool",
    flow_id: int,
    start_ns: int,
) -> _Flow:
    size_bytes, rate_bps, duration_ns = pool.pop()
    size_limited_duration_ns = int(size_bytes * 8.0 * 1e9 / rate_bps)
    effective_duration = min(size_limited_duration_ns, duration_ns)
    end_ns = int(start_ns) + int(effective_duration)
    return _Flow(
        flow_id=flow_id,
        traffic_class=pool.traffic_class,
        start_ns=int(start_ns),
        rate_bps=rate_bps,
        total_bytes=size_bytes,
        duration_ns=duration_ns,
        end_ns=end_ns,
    )


def _packets_from_window_emissions(
    window_emissions: List[Tuple[int, int, float]],
    mtu_bytes: int,
    horizon_ns: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Build an MTU-packet timeline from per-window byte emissions.

    Packets within a window are spaced uniformly across the window so
    the instantaneous emission rate equals ``bytes / win_ns`` (which is
    the fair-share rate the flow was granted in that window). The last
    packet is sized by the byte remainder when it is not a multiple of
    MTU.
    """
    ts_parts: List[np.ndarray] = []
    sz_parts: List[np.ndarray] = []
    for ws, we, b in window_emissions:
        if b <= 0 or we <= ws:
            continue
        n_pkts = int(np.ceil(b / mtu_bytes))
        if n_pkts <= 0:
            continue
        # Even spacing: gap = win_ns / n_pkts; packet i at ws + (i+0.5)*gap.
        win_ns = we - ws
        gap = win_ns / n_pkts
        ts = (ws + (np.arange(n_pkts, dtype=np.float64) + 0.5) * gap).astype(
            np.int64)
        # Clamp to horizon.
        ts = ts[ts < horizon_ns]
        if ts.size == 0:
            continue
        k = ts.size
        sz = np.full(k, mtu_bytes, dtype=np.int32)
        # Last packet size = remainder.
        rem = int(round(b)) - (k - 1) * mtu_bytes
        if 0 < rem < mtu_bytes:
            sz[-1] = int(max(rem, 1))
        ts_parts.append(ts)
        sz_parts.append(sz)
    if not ts_parts:
        return (np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int32))
    ts_all = np.concatenate(ts_parts)
    sz_all = np.concatenate(sz_parts)
    # Ensure monotonic nondecreasing timestamps (they are by construction,
    # since windows are processed in order, but guard against rounding).
    order = np.argsort(ts_all, kind="stable")
    return ts_all[order], sz_all[order]


# --------------------------------------------------------------------------
# Main two-layer generator.
# --------------------------------------------------------------------------
def generate_hal_composite(
    mix: Dict[str, float],
    total_gbps: float,
    horizon_ns: int,
    epoch_ns: int,                    # kept for signature compat; unused
    rng: np.random.Generator,
    link_gbps: float = 100.0,
    rate_update_ns: int = 1_000_000,
    per_flow_rate_cap_gbps: float = 10.0,
    mtu_bytes: int = 1500,
    n_tenants: int = 1,
) -> Tuple[List[Tuple[np.ndarray, np.ndarray]],
           List[str],
           Dict[str, Dict[str, float]]]:
    """Generate per-flow packet timelines for the HAL composite workload.

    Parameters
    ----------
    mix
        ``{class_name: fraction}`` summing to 1.
    total_gbps
        Aggregate target offered load (time-averaged across classes).
    horizon_ns
        Simulation horizon.
    epoch_ns
        Unused; kept in signature for backward compatibility. The
        spawn loop runs at Layer-1 granularity (``rate_update_ns``)
        because layer-1 rate only changes at that cadence anyway.
    link_gbps
        Clip threshold for the Layer-1 per-class lognormal rate.
        100 Gbps reproduces HAL Fig. 8.
    rate_update_ns
        Layer-1 resampling period (default 1 ms, matching HAL Fig. 8).
    per_flow_rate_cap_gbps
        Global cap on any single flow's sending rate.
    mtu_bytes
        Intra-flow MTU.

    Returns
    -------
    per_flow_pairs
        Sorted ``(timestamps_ns, sizes_bytes)`` tuples, one per flow.
    per_flow_classes
        Parallel list of traffic class names.
    diagnostics
        ``{class: stats}``.
    """
    del epoch_ns  # retained in signature for backward compat

    if not mix:
        return [], [], {}
    total_w = sum(float(v) for v in mix.values())
    if not 0.999 <= total_w <= 1.001:
        raise ValueError(f"mix fractions must sum to 1, got {total_w:.4f}")
    for c in mix:
        if c not in HAL_CLASSES:
            raise KeyError(
                f"hal_composite unknown class {c!r}; available: {HAL_CLASSES}")
    active_classes = [c for c, f in mix.items() if f > 0]

    # Layer 1.
    class_rate_timeline: Dict[str, np.ndarray] = {}
    class_target_mean: Dict[str, float] = {}
    for c in active_classes:
        tgt = float(mix[c]) * float(total_gbps)
        class_target_mean[c] = tgt
        class_rate_timeline[c] = sample_class_rate_timeline(
            c, horizon_ns, rate_update_ns, tgt, link_gbps, rng,
            n_tenants=int(max(1, n_tenants)))

    # Layer 2. One iteration per Layer-1 rate-update window. Per class
    # per window we do a closed-form fair-share allocation:
    #
    #   1. Enumerate alive flows (carried from prior windows) and their
    #      nominal per-window emission = min(remaining_bytes,
    #      rate_bps * win_ns / 8).
    #   2. Spawn new flows one at a time, each contributing its own
    #      nominal per-window emission; new flow's rate is capped at
    #      ``min(per_flow_rate_cap_gbps, Layer-1 target / expected
    #      concurrency)`` so a single big flow doesn't dominate.
    #      We keep spawning until the SUM of nominal emissions of alive
    #      + new flows reaches the window's target_bytes (or caps).
    #   3. Fair-share throttle: let ``scale = min(1, target_bytes /
    #      sum_nominal)``; every alive+new flow emits
    #      ``scale * its_nominal`` bytes in this window. Record the
    #      actual emission in the flow's ``window_emissions`` list.
    #   4. Advance remaining_bytes for alive flows; if rem > 0 and
    #      end_ns > t_end, flow remains alive for the next window.
    #
    # This guarantees realised-bytes-per-window <= target_bytes (so no
    # over-shoot of the Layer-1 process), while preserving all the
    # user-specified per-class flow characteristics (size CDF,
    # exponential duration, lognormal nominal rate). This is standard
    # max-min / proportional fair-share, i.e. the long-run steady-state
    # of TCP congestion control on a shared bottleneck (Kelly, Maulloo,
    # Tan, J. Oper. Res. Soc. 1998).
    n_updates = int(np.ceil(horizon_ns / max(1, rate_update_ns)))

    all_flows: List[_Flow] = []
    next_flow_id = 0

    # Per-class alive-flow table. We store parallel lists of:
    #   * the _Flow object (so we can append to window_emissions)
    #   * remaining bytes (float)
    alive_flows: Dict[str, List[_Flow]] = {c: [] for c in active_classes}
    alive_rem: Dict[str, List[float]] = {c: [] for c in active_classes}

    MAX_SPAWN_PER_WINDOW_PER_CLASS = 4000
    MAX_ALIVE_PER_CLASS = 50_000

    class_pool: Dict[str, _AttrPool] = {
        c: _AttrPool(c, per_flow_rate_cap_gbps, mtu_bytes, rng)
        for c in active_classes
    }

    for ui in range(n_updates):
        t_start = ui * rate_update_ns
        t_end = min((ui + 1) * rate_update_ns, horizon_ns)
        win_ns = t_end - t_start
        if win_ns <= 0:
            continue

        for c in active_classes:
            target_bps = float(class_rate_timeline[c][ui]) * 1e9
            target_bytes = target_bps * win_ns / 8.0e9

            # 1. Alive flows: compute nominal bytes this window and
            #    prune fully expired.
            kept_flows: List[_Flow] = []
            kept_rem: List[float] = []
            nominal: List[float] = []
            for f, rem in zip(alive_flows[c], alive_rem[c]):
                if rem <= 0 or f.end_ns <= t_start:
                    continue
                eff_end = min(f.end_ns, t_end)
                dt_ns = max(0, eff_end - t_start)
                nb = min(rem, f.rate_bps * dt_ns / 8.0e9)
                kept_flows.append(f)
                kept_rem.append(rem)
                nominal.append(nb)

            # 2. Spawn new flows until sum(nominal) >= target_bytes.
            new_flows: List[_Flow] = []
            new_rem: List[float] = []
            new_nominal: List[float] = []
            sum_nom = sum(nominal)
            spawned = 0
            while (sum_nom < target_bytes
                   and spawned < MAX_SPAWN_PER_WINDOW_PER_CLASS
                   and (len(kept_flows) + len(new_flows))
                       < MAX_ALIVE_PER_CLASS):
                f = _sample_flow_attrs(class_pool[c], next_flow_id, t_start)
                next_flow_id += 1
                eff_end = min(f.end_ns, t_end)
                dt_ns = max(0, eff_end - t_start)
                nb = min(float(f.total_bytes),
                         f.rate_bps * dt_ns / 8.0e9)
                new_flows.append(f)
                new_rem.append(float(f.total_bytes))
                new_nominal.append(nb)
                sum_nom += nb
                all_flows.append(f)
                spawned += 1

            # 3. Fair-share scale. If nominal already <= target, scale=1.
            if sum_nom > target_bytes and sum_nom > 0:
                scale = target_bytes / sum_nom
            else:
                scale = 1.0

            # 4. Emit and update remaining bytes.
            next_alive_flows: List[_Flow] = []
            next_alive_rem: List[float] = []
            for f, rem, nb in zip(kept_flows, kept_rem, nominal):
                emit = nb * scale
                if emit > 0:
                    if f.window_emissions is None:
                        f.window_emissions = []
                    f.window_emissions.append((t_start, t_end, emit))
                rem_after = rem - emit
                if rem_after > 0 and f.end_ns > t_end:
                    next_alive_flows.append(f)
                    next_alive_rem.append(rem_after)
            for f, rem, nb in zip(new_flows, new_rem, new_nominal):
                emit = nb * scale
                if emit > 0:
                    if f.window_emissions is None:
                        f.window_emissions = []
                    f.window_emissions.append((t_start, t_end, emit))
                rem_after = rem - emit
                if rem_after > 0 and f.end_ns > t_end:
                    next_alive_flows.append(f)
                    next_alive_rem.append(rem_after)

            alive_flows[c] = next_alive_flows
            alive_rem[c] = next_alive_rem

    # Build per-flow packet timelines from the recorded per-window
    # emissions.
    per_flow_pairs: List[Tuple[np.ndarray, np.ndarray]] = []
    per_flow_classes: List[str] = []
    class_total_bytes: Dict[str, float] = {c: 0.0 for c in active_classes}
    class_total_pkts: Dict[str, int] = {c: 0 for c in active_classes}
    class_total_flows: Dict[str, int] = {c: 0 for c in active_classes}

    for f in all_flows:
        if not f.window_emissions:
            class_total_flows[f.traffic_class] += 1
            continue
        ts, sz = _packets_from_window_emissions(
            f.window_emissions, mtu_bytes, horizon_ns)
        if ts.size == 0:
            class_total_flows[f.traffic_class] += 1
            continue
        per_flow_pairs.append((ts, sz))
        per_flow_classes.append(f.traffic_class)
        class_total_bytes[f.traffic_class] += float(sz.sum())
        class_total_pkts[f.traffic_class] += int(ts.size)
        class_total_flows[f.traffic_class] += 1

    horizon_s = max(1e-12, horizon_ns * 1e-9)
    diagnostics: Dict[str, Dict[str, float]] = {}
    for c in active_classes:
        realised_gbps = class_total_bytes[c] * 8.0 / horizon_s / 1e9
        timeline = class_rate_timeline[c]
        diagnostics[c] = dict(
            target_mean_gbps=float(class_target_mean[c]),
            layer1_mean_gbps=float(timeline.mean()),
            layer1_max_gbps=float(timeline.max() if timeline.size else 0.0),
            realized_gbps=float(realised_gbps),
            total_packets=int(class_total_pkts[c]),
            total_bytes=float(class_total_bytes[c]),
            n_flows=int(class_total_flows[c]),
            mean_flow_bytes=(class_total_bytes[c] / class_total_flows[c]
                             if class_total_flows[c] else 0.0),
            mu=float(HAL_LOGNORMAL_PARAMS[c][0]),
            sigma=float(HAL_LOGNORMAL_PARAMS[c][1]),
            link_clip_gbps=float(link_gbps),
            rate_update_ns=float(rate_update_ns),
            n_tenants=int(max(1, n_tenants)),
        )

    return per_flow_pairs, per_flow_classes, diagnostics
