"""Digitized CDFs approximated from IMC'17 and contemporaneous FB/Meta
datacenter traffic characterization.

Sources (approximate digitization; the exact numbers are interpolated
from published figures, so keep these as a *reasonable* realism anchor
rather than a bit-exact reproduction):

* Zhang et al., "High-resolution measurement of data center microbursts",
  IMC 2017. Main source for **inter-packet arrival-time CDFs** at
  microsecond resolution, per cluster (Web / Cache-follower / Hadoop).
* Roy et al., "Inside the Social Network's (Datacenter) Network",
  SIGCOMM 2015. Main source for **flow-size** and **packet-size** CDFs
  per cluster.

Usage
-----
Each CDF is a list of (value, cum_probability) control points, sorted
ascending by value. The ``CdfSampler`` performs a piecewise-loglinear
inverse lookup to draw realistic samples. We sample on log(value)
because all three underlying distributions span multiple orders of
magnitude.

If you have better digitization (or access to the original raw data),
drop new (value, cum_prob) points here; every trace generator reads
from this table via :func:`get_cdf`.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

import numpy as np


CdfPoints = List[Tuple[float, float]]


# ---------------------------------------------------------------------------
# Per-class CDFs. Keep (value, cum_prob) sorted ascending by value.
# cum_prob MUST monotonically increase from (>=0) up to 1.0.
# ---------------------------------------------------------------------------

# ----- Flow size (bytes) -----
_FLOW_SIZE_BYTES: Dict[str, CdfPoints] = {
    # Web: short request/response, heavy tail
    "web": [
        (1e2, 0.10), (1e3, 0.40), (1e4, 0.80),
        (1e5, 0.95), (1e6, 0.99), (1e7, 1.0),
    ],
    # Cache follower (memcached-like): mostly small values
    "cache": [
        (1e2, 0.30), (1e3, 0.80), (1e4, 0.95),
        (1e5, 0.99), (1e6, 1.0),
    ],
    # Hadoop: strongly bimodal; lots of small ACKs + big bulk transfers
    "hadoop": [
        (1e3, 0.10), (1e4, 0.30), (1e5, 0.50),
        (1e6, 0.70), (1e7, 0.85), (1e8, 0.95), (1e9, 1.0),
    ],
}


# ----- Packet size (bytes) -----
_PKT_SIZE_BYTES: Dict[str, CdfPoints] = {
    "web": [
        (64, 0.30), (100, 0.50), (500, 0.65),
        (1000, 0.80), (1500, 1.0),
    ],
    "cache": [
        (64, 0.40), (96, 0.65), (128, 0.82),
        (256, 0.90), (500, 0.95), (1500, 1.0),
    ],
    "hadoop": [
        (64, 0.10), (200, 0.15), (500, 0.20),
        (1000, 0.30), (1500, 1.0),
    ],
}


# ----- Inter-packet arrival within an active burst (ns) -----
# Taken from IMC'17 microsecond-resolution CDFs: small IATs dominate
# cache + hadoop (continuous traffic) while web is bursty with large
# inter-request gaps.
_IAT_NS: Dict[str, CdfPoints] = {
    "web": [
        (1e3, 0.20), (1e4, 0.40), (1e5, 0.65),
        (1e6, 0.85), (1e7, 1.0),
    ],
    "cache": [
        (5e2, 0.20), (1e3, 0.40), (1e4, 0.70),
        (1e5, 0.90), (1e6, 1.0),
    ],
    "hadoop": [
        (2e2, 0.20), (5e2, 0.50), (1e3, 0.75),
        (1e4, 0.92), (1e5, 1.0),
    ],
}


# ----- Burst (ON period) duration in ns -----
# How long a flow stays active before going idle. IMC'17 reports
# microbursts of ~25us at 99.99% and larger ON regions for bulk flows.
_ON_NS: Dict[str, CdfPoints] = {
    "web": [
        (1e4, 0.20), (5e4, 0.50), (2e5, 0.80), (1e6, 0.98), (1e7, 1.0),
    ],
    "cache": [
        (5e3, 0.20), (2e4, 0.55), (1e5, 0.85), (1e6, 1.0),
    ],
    "hadoop": [
        (1e5, 0.20), (1e6, 0.55), (1e7, 0.85), (1e8, 1.0),
    ],
}


# ----- OFF period duration in ns -----
_OFF_NS: Dict[str, CdfPoints] = {
    "web": [
        (1e5, 0.20), (5e5, 0.55), (2e6, 0.80), (1e7, 0.95), (1e8, 1.0),
    ],
    "cache": [
        (1e4, 0.20), (5e4, 0.55), (2e5, 0.85), (1e6, 1.0),
    ],
    "hadoop": [
        (1e4, 0.20), (1e5, 0.55), (1e6, 0.85), (1e7, 1.0),
    ],
}


_ALL_TABLES: Dict[str, Dict[str, CdfPoints]] = {
    "flow_size_bytes": _FLOW_SIZE_BYTES,
    "pkt_size_bytes": _PKT_SIZE_BYTES,
    "iat_ns": _IAT_NS,
    "on_ns": _ON_NS,
    "off_ns": _OFF_NS,
}

CLASSES = ("web", "cache", "hadoop")


def get_cdf(quantity: str, traffic_class: str) -> CdfPoints:
    """Return the raw CDF control points for a (quantity, class)."""
    if quantity not in _ALL_TABLES:
        raise KeyError(
            f"unknown CDF quantity {quantity}; "
            f"available: {list(_ALL_TABLES)}"
        )
    table = _ALL_TABLES[quantity]
    if traffic_class not in table:
        raise KeyError(
            f"unknown traffic_class {traffic_class}; "
            f"available for {quantity}: {list(table)}"
        )
    return table[traffic_class]


# ---------------------------------------------------------------------------
# Inverse-CDF sampling (log-linear interpolation)
# ---------------------------------------------------------------------------
@dataclass
class CdfSampler:
    """Inverse-CDF sampler with log-linear interpolation between control
    points. Handles wide dynamic ranges (bytes from 1e2 to 1e9) without
    concentrating probability mass at the endpoints."""
    xs: np.ndarray   # values, log-spaced (log(value))
    ps: np.ndarray   # cumulative probabilities

    @classmethod
    def from_points(cls, points: Sequence[Tuple[float, float]]) -> "CdfSampler":
        if len(points) < 2:
            raise ValueError("CDF needs >= 2 points")
        xs = np.array([float(x) for x, _ in points], dtype=np.float64)
        ps = np.array([float(p) for _, p in points], dtype=np.float64)
        if np.any(xs <= 0):
            raise ValueError("values must be positive (log domain)")
        if not np.all(np.diff(xs) > 0):
            raise ValueError("values must be strictly increasing")
        if not np.all(np.diff(ps) >= 0):
            raise ValueError("probabilities must be non-decreasing")
        # Clamp bounds to [0, 1]. Users sometimes supply 0.995 at the
        # tail; we renormalize so u=1 always maps to the largest value.
        ps = np.clip(ps, 0.0, 1.0)
        if ps[0] > 0:
            xs = np.concatenate([[xs[0]], xs])
            ps = np.concatenate([[0.0], ps])
        if ps[-1] < 1.0:
            xs = np.concatenate([xs, [xs[-1]]])
            ps = np.concatenate([ps, [1.0]])
        return cls(np.log(xs), ps)

    def sample(self, n: int, rng: np.random.Generator) -> np.ndarray:
        """Draw ``n`` samples in the original (linear) unit."""
        u = rng.random(n)
        # piecewise-linear inverse CDF in log(value) domain
        log_x = np.interp(u, self.ps, self.xs)
        return np.exp(log_x)


def make_sampler(quantity: str, traffic_class: str) -> CdfSampler:
    return CdfSampler.from_points(get_cdf(quantity, traffic_class))
