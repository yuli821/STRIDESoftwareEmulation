"""Metrics logging + summary stats."""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Dict, List

import numpy as np
import pandas as pd


def fairness(x: np.ndarray, eps: float = 1e-9) -> float:
    """Jain's fairness index on a non-negative vector.

    Returns 1.0 when all entries are equal (perfect fairness) and 1/N in
    the worst case (one entry concentrates all the mass). Reported in
    CSV/summary simply as ``fairness_*``.
    """
    x = np.asarray(x, dtype=np.float64)
    return float((x.sum()) ** 2 / (x.size * (x * x).sum() + eps))


def imbalance(x: np.ndarray, eps: float = 1e-9) -> float:
    """Gini coefficient of a non-negative vector, re-exposed with a more
    intuitive label: 0.0 == perfectly balanced load, 1.0 == all load on
    a single queue. Used to quantify how skewed the per-queue byte
    distribution is.
    """
    x = np.asarray(x, dtype=np.float64)
    if x.size == 0 or x.sum() == 0:
        return 0.0
    sx = np.sort(x)
    n = x.size
    cum = np.cumsum(sx)
    return float((2.0 * np.sum(np.arange(1, n + 1) * sx) / (n * cum[-1] + eps))
                 - (n + 1.0) / n)


# Backward-compat aliases (so any external scripts keep working).
jain = fairness
gini = imbalance


@dataclass
class DomainMetricsLog:
    name: str
    rows: List[Dict] = field(default_factory=list)
    bucket_trace_rows: List[Dict] = field(default_factory=list)
    reassign_counts: List[int] = field(default_factory=list)

    def log_epoch(self, epoch: int, telem: Dict[str, np.ndarray],
                  extra: Dict) -> None:
        P = telem["P_q"]
        row = {
            "epoch": epoch,
            "total_B_gen": float(telem["B_q_gen"].sum()),
            "total_B_adm": float(telem["B_q_adm"].sum()),
            "total_B_drop": float(telem["B_q_drop"].sum()),
            "total_N_gen": int(telem["N_q_gen"].sum()),
            "total_N_adm": int(telem["N_q_adm"].sum()),
            "total_N_drop": int(telem["N_q_drop"].sum()),
            "drop_ratio": float(telem["B_q_drop"].sum()
                                / (telem["B_q_gen"].sum() + 1e-9)),
            "P_q_max": float(P.max()),
            "P_q_mean": float(P.mean()),
            "P_q_std": float(P.std()),
            "fairness_P": fairness(np.clip(P, 1e-6, None)),
            "imbalance_B": imbalance(telem["B_q_gen"]),
        }
        row.update(extra)
        for q in range(P.size):
            row[f"P_q[{q}]"] = float(P[q])
            row[f"B_q_gen[{q}]"] = float(telem["B_q_gen"][q])
            row[f"B_q_drop[{q}]"] = float(telem["B_q_drop"][q])
            row[f"K_q[{q}]"] = int(telem["K_q"][q])
        self.rows.append(row)

    def log_bucket_trace(self, epoch: int, table: np.ndarray,
                         B_b: np.ndarray) -> None:
        for b in range(table.size):
            self.bucket_trace_rows.append({
                "epoch": epoch,
                "bucket": int(b),
                "queue": int(table[b]),
                "bytes": float(B_b[b]),
            })

    def save(self, outdir: str) -> None:
        os.makedirs(outdir, exist_ok=True)
        pd.DataFrame(self.rows).to_csv(
            os.path.join(outdir, f"timeseries_{self.name}.csv"), index=False)
        if self.bucket_trace_rows:
            pd.DataFrame(self.bucket_trace_rows).to_csv(
                os.path.join(outdir, f"bucket_trace_{self.name}.csv"),
                index=False)

    def summary(self) -> Dict:
        df = pd.DataFrame(self.rows)
        if df.empty:
            return {}
        return {
            "domain": self.name,
            "epochs": int(df.shape[0]),
            "aggregate_B_gen_GB": float(df["total_B_gen"].sum() / 1e9),
            "aggregate_B_adm_GB": float(df["total_B_adm"].sum() / 1e9),
            "aggregate_B_drop_GB": float(df["total_B_drop"].sum() / 1e9),
            "overall_drop_ratio": float(df["total_B_drop"].sum()
                                        / (df["total_B_gen"].sum() + 1e-9)),
            "mean_P_max": float(df["P_q_max"].mean()),
            "p99_P_max": float(df["P_q_max"].quantile(0.99)),
            "mean_fairness_P": float(df["fairness_P"].mean()),
            "mean_imbalance_B": float(df["imbalance_B"].mean()),
            "total_reassignments": int(sum(self.reassign_counts)),
        }


def write_summary(outdir: str, payload: Dict) -> None:
    os.makedirs(outdir, exist_ok=True)
    with open(os.path.join(outdir, "summary.json"), "w") as f:
        json.dump(payload, f, indent=2, default=float)
