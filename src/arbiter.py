"""HW stream arbiter between stateless and stateful egress.

Applied per bin: if aggregate bytes exceed link capacity * bin, proportionally
drop per configured policy. Applied BEFORE packets enter the host pipeline
(because link saturation is an egress-side effect upstream of the DMA path).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class Arbiter:
    policy: str
    w_sl: int
    w_sf: int
    max_bytes_per_bin: float

    def __post_init__(self) -> None:
        if self.policy not in ("wrr", "strict_priority_stateless",
                               "strict_priority_stateful", "drr", "random"):
            raise ValueError(f"unknown arbiter policy: {self.policy}")
        self._drr_s = 0.0
        self._drr_t = 0.0

    def scale_factors(self, bytes_sl: float, bytes_sf: float) -> tuple[float, float]:
        """Return (scale_sl, scale_sf) in [0,1] that, when applied to the
        stateless and stateful byte totals, satisfy the link cap per policy."""
        total = bytes_sl + bytes_sf
        cap = self.max_bytes_per_bin
        if total <= cap or total == 0.0:
            return 1.0, 1.0
        if self.policy == "wrr":
            fs = self.w_sl / (self.w_sl + self.w_sf)
            ft = 1.0 - fs
            share_s = min(bytes_sl, cap * fs)
            share_t = min(bytes_sf, cap * ft)
            leftover = cap - share_s - share_t
            if leftover > 0 and bytes_sl - share_s > 0:
                add = min(bytes_sl - share_s, leftover)
                share_s += add
                leftover -= add
            if leftover > 0 and bytes_sf - share_t > 0:
                share_t += min(bytes_sf - share_t, leftover)
        elif self.policy == "strict_priority_stateless":
            share_s = min(bytes_sl, cap)
            share_t = min(bytes_sf, cap - share_s)
        elif self.policy == "strict_priority_stateful":
            share_t = min(bytes_sf, cap)
            share_s = min(bytes_sl, cap - share_t)
        elif self.policy == "drr":
            qs = self.w_sl * cap / (self.w_sl + self.w_sf)
            qt = self.w_sf * cap / (self.w_sl + self.w_sf)
            self._drr_s += qs
            self._drr_t += qt
            share_s = min(bytes_sl, self._drr_s)
            share_t = min(bytes_sf, self._drr_t)
            self._drr_s -= share_s
            self._drr_t -= share_t
            tot = share_s + share_t
            if tot > cap:
                f = cap / tot
                share_s *= f
                share_t *= f
        else:   # random / fallback
            f = cap / total
            share_s = bytes_sl * f
            share_t = bytes_sf * f
        return (share_s / bytes_sl if bytes_sl > 0 else 0.0,
                share_t / bytes_sf if bytes_sf > 0 else 0.0)
