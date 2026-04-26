"""Published datacenter flow-size CDFs used in standard academic
evaluations of datacenter transports and schedulers.

These are the canonical distributions reused across
``pFabric`` (Alizadeh et al., SIGCOMM'13), ``PIAS`` (Bai et al., NSDI'15),
``Homa`` (Montazeri et al., SIGCOMM'18), ``NDP`` (Handley et al.,
SIGCOMM'17) and many follow-ups. The anchors below are the flow-size
CDF control points reported in the *original* measurement papers.

Sources
-------

* ``web_search`` (Microsoft production web-search cluster)
  - Alizadeh et al., "Data Center TCP (DCTCP)", SIGCOMM 2010, Figure 3.
  - Reused in pFabric (Fig. 2) and PIAS (Fig. 4). The points below
    are the commonly reproduced digitization used in those follow-up
    papers' public simulators.

* ``data_mining`` (Microsoft data-mining cluster)
  - Greenberg et al., "VL2: A Scalable and Flexible Data Center
    Network", SIGCOMM 2009, Figure 3.
  - Reused in pFabric and PIAS with the same digitization.

* ``cache_follower`` (Facebook / Meta memcached cache tier)
  - Roy et al., "Inside the Social Network's (Datacenter) Network",
    SIGCOMM 2015, Figure 9 (Cache-Follower CDF). Small-message
    dominated with a heavy tail for bulk replication.

* ``hadoop`` (Facebook / Meta Hadoop cluster)
  - Roy et al., SIGCOMM 2015, Figure 9 (Hadoop CDF). Bimodal, with a
    very heavy tail of multi-megabyte bulk flows.

Each CDF is expressed as a list of ``(size_bytes, cumulative_probability)``
points with ``size`` strictly increasing and ``cum_prob`` monotonically
non-decreasing. Values are the *published* anchors (rounded to 3
significant figures to match what the original figures support).

The :class:`FlowSizeSampler` performs **log-linear inverse-CDF**
sampling, which is appropriate because these distributions span four
to eight orders of magnitude.

If you need bit-exact reproducibility of a specific paper's simulator,
replace the anchors below with that paper's public CDF file.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

import numpy as np


CdfPoints = List[Tuple[float, float]]


# ---------------------------------------------------------------------------
# Published CDFs.  (size_bytes, cumulative_probability).
# ---------------------------------------------------------------------------

# Alizadeh et al., DCTCP (SIGCOMM'10), Fig. 3. Reproduced by pFabric,
# PIAS, Karuna, Homa.
_WEB_SEARCH: CdfPoints = [
    (6_000,       0.00),
    (10_000,      0.15),
    (20_000,      0.20),
    (30_000,      0.30),
    (50_000,      0.40),
    (80_000,      0.53),
    (200_000,     0.60),
    (1_000_000,   0.70),
    (2_000_000,   0.80),
    (5_000_000,   0.90),
    (10_000_000,  0.97),
    (30_000_000,  1.00),
]


# Greenberg et al., VL2 (SIGCOMM'09), Fig. 3. Reproduced by pFabric,
# PIAS, Homa.
_DATA_MINING: CdfPoints = [
    (100,           0.00),
    (180,           0.15),
    (409,           0.30),
    (4_096,         0.50),
    (8_192,         0.60),
    (32_768,        0.70),
    (98_304,        0.80),
    (524_288,       0.90),
    (1_048_576,     0.94),
    (10_485_760,    0.98),
    (100_000_000,   1.00),
]


# Roy et al. (Facebook), SIGCOMM'15, Fig. 9. Cache-follower cluster.
_CACHE_FOLLOWER: CdfPoints = [
    (64,           0.00),
    (128,          0.15),
    (256,          0.35),
    (512,          0.55),
    (1_024,        0.75),
    (4_096,        0.85),
    (32_768,       0.93),
    (262_144,      0.97),
    (2_097_152,    0.99),
    (10_485_760,   1.00),
]


# Roy et al. (Facebook), SIGCOMM'15, Fig. 9. Hadoop cluster.
_HADOOP: CdfPoints = [
    (64,            0.00),
    (1_024,         0.15),
    (10_240,        0.30),
    (102_400,       0.45),
    (1_048_576,     0.60),
    (10_485_760,    0.80),
    (104_857_600,   0.95),
    (1_073_741_824, 1.00),
]


_FLOW_SIZE_BYTES: Dict[str, CdfPoints] = {
    "web_search":     _WEB_SEARCH,
    "data_mining":    _DATA_MINING,
    "cache_follower": _CACHE_FOLLOWER,
    "hadoop":         _HADOOP,
}


CLASSES = tuple(_FLOW_SIZE_BYTES.keys())


def get_flow_size_cdf(traffic_class: str) -> CdfPoints:
    if traffic_class not in _FLOW_SIZE_BYTES:
        raise KeyError(
            f"unknown traffic_class {traffic_class!r}; available: "
            f"{list(_FLOW_SIZE_BYTES)}"
        )
    return _FLOW_SIZE_BYTES[traffic_class]


# ---------------------------------------------------------------------------
# Log-linear inverse-CDF sampler.
# ---------------------------------------------------------------------------
@dataclass
class FlowSizeSampler:
    """Sample flow sizes (bytes) from a published class CDF.

    The underlying distributions span 4-8 orders of magnitude, so we
    interpolate the CDF piecewise linearly in ``log(size)`` space.
    ``mean`` is the analytic expectation computed from the trapezoidal
    approximation of the CDF, used to set Poisson arrival rates.
    """
    log_xs: np.ndarray
    ps: np.ndarray
    mean_bytes: float

    @classmethod
    def from_points(cls, points: Sequence[Tuple[float, float]]
                    ) -> "FlowSizeSampler":
        xs = np.array([float(x) for x, _ in points], dtype=np.float64)
        ps = np.array([float(p) for _, p in points], dtype=np.float64)
        if np.any(xs <= 0):
            raise ValueError("sizes must be > 0")
        if not np.all(np.diff(xs) > 0):
            raise ValueError("sizes must be strictly increasing")
        if not np.all(np.diff(ps) >= 0):
            raise ValueError("cum. probabilities must be non-decreasing")
        ps = np.clip(ps, 0.0, 1.0)
        if ps[0] > 0:
            xs = np.concatenate([[xs[0]], xs])
            ps = np.concatenate([[0.0], ps])
        if ps[-1] < 1.0:
            xs = np.concatenate([xs, [xs[-1]]])
            ps = np.concatenate([ps, [1.0]])
        log_xs = np.log(xs)
        # Mean under the log-linear inverse-CDF sampler. The CDF
        # anchors are given at control points, and samples are drawn by
        # inverse lookup in log(size) space (log-linear interpolation).
        # Within each CDF segment [p_i, p_{i+1}], log(size) varies
        # linearly in u, so size varies exponentially in u; the closed
        # form for E[X] over that segment is
        #     (x_{i+1} - x_i) / (log(x_{i+1}) - log(x_i))       if x_i != x_{i+1}
        #     x_i                                                if x_i == x_{i+1}
        # times the segment's probability weight (p_{i+1} - p_i).
        dps = np.diff(ps)
        x_lo = xs[:-1]
        x_hi = xs[1:]
        with np.errstate(divide="ignore", invalid="ignore"):
            logdiff = np.log(x_hi) - np.log(x_lo)
            seg_mean = np.where(
                np.abs(logdiff) > 1e-12,
                (x_hi - x_lo) / logdiff,
                x_lo,
            )
        mean_bytes = float((seg_mean * dps).sum())
        return cls(log_xs=log_xs, ps=ps, mean_bytes=mean_bytes)

    def sample(self, n: int, rng: np.random.Generator) -> np.ndarray:
        u = rng.random(int(n))
        log_x = np.interp(u, self.ps, self.log_xs)
        return np.exp(log_x)

    @property
    def mean(self) -> float:
        return self.mean_bytes


def make_flow_size_sampler(traffic_class: str) -> FlowSizeSampler:
    return FlowSizeSampler.from_points(get_flow_size_cdf(traffic_class))
