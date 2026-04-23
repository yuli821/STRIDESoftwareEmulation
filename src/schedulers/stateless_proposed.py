"""Stateless bucket-reassignment schedulers.

Two orthogonal design axes:

* **signal**:  how the scheduler decides which queues are hot.
    ``qp``      : pressure only  (H = P)
    ``pred``    : predictor only (H = R_hat)
    ``pred_qp`` : weighted blend (H = alpha * P + (1 - alpha) * R_hat)

* **policy**:  how moves are executed.
    ``greedy``  : Algorithm 2 iterative greedy loop:
                  pick hottest hot, coldest cold, move heaviest bucket
                  that satisfies the paper's fit condition, refresh
                  hot/cold sets, repeat until empty or cap reached.
    ``oneshot`` : single pass:
                  for each hot queue (in descending H), move its
                  heaviest bucket to the globally coldest unused queue.
                  No fit check, no iterative refresh. Cheaper, faster,
                  but can overshoot destination load.

The concrete scheduler name is ``{signal}_{policy}``. Supported names:

* ``static``            -- no reassignments, baseline.
* ``qp_oneshot``        -- pressure-only, oneshot
* ``qp_greedy``         -- pressure-only, greedy
* ``pred_oneshot``      -- predictor-only, oneshot
* ``pred_greedy``       -- predictor-only, greedy
* ``pred_qp_oneshot``   -- blended, oneshot
* ``pred_qp_greedy``    -- blended, greedy (paper's Algorithm 2)

Backward-compat aliases (accepted in YAML):
    ewma_greedy       -> pred_qp_greedy
    reactive_greedy   -> qp_greedy
    reactive_oneshot  -> qp_oneshot
    proposed          -> pred_qp_greedy
    current_only      -> qp_greedy
    reactive_no_pred  -> qp_oneshot
"""
from __future__ import annotations

from typing import Dict, Tuple
import numpy as np

from .base import BaseStatelessScheduler
from ..config import StatelessSchedulerConfig


SIGNALS = ("qp", "pred", "pred_qp")
POLICIES = ("oneshot", "greedy")


def parse_scheduler_type(t: str) -> Tuple[str, str]:
    """Return ``(signal, policy)`` for a canonical ``{signal}_{policy}``
    name. Raises if unknown."""
    for pol in POLICIES:
        suffix = "_" + pol
        if t.endswith(suffix):
            signal = t[: -len(suffix)]
            if signal in SIGNALS:
                return signal, pol
    raise ValueError(f"unknown scheduler_type: {t!r}. "
                     f"Expected {{signal}}_{{policy}} with "
                     f"signal in {SIGNALS}, policy in {POLICIES}.")


def _compute_signal(signal: str, P: np.ndarray, pred_risk: np.ndarray,
                    alpha_blend: float) -> np.ndarray:
    if signal == "qp":
        return P.copy()
    if signal == "pred":
        if pred_risk is None or pred_risk.size == 0:
            return P.copy()
        return np.clip(pred_risk, 0.0, 1.0)
    if signal == "pred_qp":
        if pred_risk is None or pred_risk.size == 0:
            return P.copy()
        a = float(alpha_blend)
        return a * P + (1.0 - a) * pred_risk
    raise ValueError(f"unknown signal: {signal}")


class GreedyStatelessScheduler(BaseStatelessScheduler):
    """Greedy iterative reassignment (paper's Algorithm 2).

    ``signal`` selects the hotness function; the core loop is the same.
    """

    def __init__(self, cfg: StatelessSchedulerConfig, signal: str) -> None:
        assert signal in SIGNALS, signal
        self.cfg = cfg
        self.signal = signal
        self.name = f"{signal}_greedy"
        self.moves_this_epoch = 0

    def step(self, epoch, telem, current_table, pred_risk):
        P = telem["P_q"]
        B_q_gen = telem["B_q_gen"].copy()
        B_b = telem["B_b"].copy()

        H = _compute_signal(self.signal, P, pred_risk, self.cfg.alpha_blend)

        new_table = current_table.copy()
        avg_B = B_q_gen.mean()
        # Fit-condition tolerance (Algorithm 2): the destination queue's
        # post-move byte count must stay within (1+tolerance) * mean.
        fit_slack = self.cfg.fit_condition_tolerance * (avg_B + 1.0)

        hot = np.where(H > self.cfg.tau_hot_s)[0]
        cold = np.where(H < self.cfg.tau_cold_s)[0]
        self.moves_this_epoch = 0

        while len(hot) > 0 and len(cold) > 0 and \
                self.moves_this_epoch < self.cfg.max_moves_per_epoch:
            q_src = int(hot[np.argmax(H[hot])])
            q_dst = int(cold[np.argmin(H[cold])])
            bucket_ids = np.where(new_table == q_src)[0]
            if bucket_ids.size == 0:
                hot = hot[hot != q_src]
                continue
            order = bucket_ids[np.argsort(-B_b[bucket_ids])]
            moved = False
            for b in order:
                b = int(b)
                new_B_src = B_q_gen[q_src] - B_b[b]
                new_B_dst = B_q_gen[q_dst] + B_b[b]
                if new_B_dst <= avg_B + fit_slack:
                    new_table[b] = q_dst
                    B_q_gen[q_src] = new_B_src
                    B_q_gen[q_dst] = new_B_dst
                    self.moves_this_epoch += 1
                    denom = avg_B + 1e-9
                    H[q_src] = max(0.0, H[q_src] - 0.5 * B_b[b] / denom)
                    H[q_dst] = min(1.0, H[q_dst] + 0.5 * B_b[b] / denom)
                    moved = True
                    break
            if not moved:
                cold = np.setdiff1d(cold, [q_dst])
            hot = np.where(H > self.cfg.tau_hot_s)[0]
            if moved:
                cold = np.where(H < self.cfg.tau_cold_s)[0]
        return new_table


class OneshotStatelessScheduler(BaseStatelessScheduler):
    """Single-pass threshold reactive scheduler.

    ``signal`` selects the hotness function; for each queue with
    ``H > tau_hot_s`` (processed in descending H order) we move the
    heaviest bucket to the globally coolest queue that hasn't been used
    as destination yet this epoch. No fit-condition check.
    """

    def __init__(self, cfg: StatelessSchedulerConfig, signal: str) -> None:
        assert signal in SIGNALS, signal
        self.cfg = cfg
        self.signal = signal
        self.name = f"{signal}_oneshot"
        self.moves_this_epoch = 0

    def step(self, epoch, telem, current_table, pred_risk):
        P = telem["P_q"]
        B_b = telem["B_b"]
        H = _compute_signal(self.signal, P, pred_risk, self.cfg.alpha_blend)

        new_table = current_table.copy()
        self.moves_this_epoch = 0
        hot = np.where(H > self.cfg.tau_hot_s)[0]
        used: set[int] = set()
        for q_src in [int(hot[i]) for i in np.argsort(-H[hot])]:
            if self.moves_this_epoch >= self.cfg.max_moves_per_epoch:
                break
            bucket_ids = np.where(new_table == q_src)[0]
            if bucket_ids.size == 0:
                continue
            b = int(bucket_ids[np.argmax(B_b[bucket_ids])])
            q_dst = None
            for q in np.argsort(H):
                q = int(q)
                if q == q_src or q in used:
                    continue
                q_dst = q
                break
            if q_dst is None:
                break
            new_table[b] = q_dst
            used.add(q_dst)
            self.moves_this_epoch += 1
        return new_table


# ---------------------------------------------------------------------------
# Aliases for older scheduler names (kept so existing YAMLs keep working).
# ---------------------------------------------------------------------------
LEGACY_ALIASES: Dict[str, str] = {
    # Current-generation (pre-matrix) names
    "ewma_greedy": "pred_qp_greedy",
    "reactive_greedy": "qp_greedy",
    "reactive_oneshot": "qp_oneshot",
    # Original names from the first iteration
    "proposed": "pred_qp_greedy",
    "current_only": "qp_greedy",
    "reactive_no_pred": "qp_oneshot",
}


def canonicalize(t: str) -> str:
    return LEGACY_ALIASES.get(t, t)
