"""Auto-generated matplotlib plots."""
from __future__ import annotations

import os
from typing import Dict, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


def _load(outdir: str, domain: str) -> Optional[pd.DataFrame]:
    p = os.path.join(outdir, f"timeseries_{domain}.csv")
    if not os.path.exists(p):
        return None
    return pd.read_csv(p)


def plot_run(outdir: str) -> None:
    for domain in ("stateless", "stateful"):
        df = _load(outdir, domain)
        if df is None or df.empty:
            continue
        _plot_pressure(df, outdir, domain)
        _plot_drops(df, outdir, domain)
        _plot_fairness(df, outdir, domain)
        _plot_reassign(df, outdir, domain)


def _plot_pressure(df, outdir, domain):
    fig, ax = plt.subplots(figsize=(8, 4))
    q_cols = [c for c in df.columns if c.startswith("P_q[")]
    for c in q_cols:
        ax.plot(df["epoch"], df[c], alpha=0.4, linewidth=0.8)
    ax.plot(df["epoch"], df["P_q_max"], color="k", linewidth=1.6, label="max")
    ax.plot(df["epoch"], df["P_q_mean"], color="r", linewidth=1.2, label="mean")
    ax.set_xlabel("epoch"); ax.set_ylabel("P_q")
    ax.set_title(f"{domain} queue pressure"); ax.legend()
    fig.tight_layout(); fig.savefig(os.path.join(outdir, f"pressure_{domain}.png"), dpi=120)
    plt.close(fig)


def _plot_drops(df, outdir, domain):
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(df["epoch"], df["drop_ratio"], color="C3")
    ax.set_xlabel("epoch"); ax.set_ylabel("drop ratio")
    ax.set_title(f"{domain} drop ratio")
    fig.tight_layout(); fig.savefig(os.path.join(outdir, f"drops_{domain}.png"), dpi=120)
    plt.close(fig)


def _plot_fairness(df, outdir, domain):
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(df["epoch"], df["fairness_P"],
            label="fairness of pressure (1=perfect)")
    ax.plot(df["epoch"], df["imbalance_B"],
            label="byte imbalance (0=balanced)")
    ax.set_xlabel("epoch"); ax.set_ylabel("value")
    ax.set_title(f"{domain} balance metrics"); ax.legend()
    fig.tight_layout(); fig.savefig(os.path.join(outdir, f"balance_{domain}.png"), dpi=120)
    plt.close(fig)


def _plot_reassign(df, outdir, domain):
    col = "reassignments" if "reassignments" in df.columns else \
          ("handoffs_committed" if "handoffs_committed" in df.columns else None)
    if col is None:
        return
    fig, ax = plt.subplots(figsize=(8, 3))
    ax.bar(df["epoch"], df[col], width=1.0, color="C2")
    ax.set_xlabel("epoch"); ax.set_ylabel(col)
    ax.set_title(f"{domain} {col}")
    fig.tight_layout(); fig.savefig(os.path.join(outdir, f"reassign_{domain}.png"), dpi=120)
    plt.close(fig)


def plot_comparison(runs: Dict[str, str], outdir: str,
                    domain: str = "stateless",
                    include: Optional[list] = None) -> None:
    """Overlay time-series for the given set of runs.

    If ``include`` is provided, only runs whose name is in that list are
    plotted (preserving the order of ``include``). This is used e.g. for
    the stateful-flow comparison, where several stateless ablations share
    the same stateful scheduler and would otherwise produce duplicate
    overlapping lines.
    """
    os.makedirs(outdir, exist_ok=True)
    dfs = {name: _load(d, domain) for name, d in runs.items()}
    dfs = {k: v for k, v in dfs.items() if v is not None and not v.empty}
    if include is not None:
        dfs = {k: dfs[k] for k in include if k in dfs}
    if not dfs:
        return
    for metric, ylab, title in [
        ("P_q_max", "max P_q", "peak queue pressure"),
        ("drop_ratio", "drop ratio", "drop ratio"),
        ("fairness_P", "fairness (1 = equal load)", "queue-pressure fairness"),
        ("imbalance_B", "imbalance (0 = equal load)", "byte imbalance across queues"),
    ]:
        fig, ax = plt.subplots(figsize=(9, 4))
        for name, df in dfs.items():
            ax.plot(df["epoch"], df[metric], label=name, alpha=0.85)
        ax.set_xlabel("epoch"); ax.set_ylabel(ylab)
        ax.set_title(f"{domain} {title}: experiment comparison"); ax.legend()
        fig.tight_layout()
        fig.savefig(os.path.join(outdir, f"compare_{metric}_{domain}.png"), dpi=120)
        plt.close(fig)
