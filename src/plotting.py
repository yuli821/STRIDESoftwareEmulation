"""Auto-generated matplotlib plots."""
from __future__ import annotations

import os
from typing import Dict, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

_BOUNDED_METRICS = {"P_q", "P_q_max", "drop_ratio", "fairness_P",
                    "fairness_B", "imbalance_B",
                    "fairness_B_cum", "imbalance_B_cum",
                    "fairness_B_win", "imbalance_B_win"}

# Rolling-window width for byte-fairness/imbalance as a *fraction* of the
# run length. 20 % is short enough to drop the initial transient after
# roughly the first fifth of the horizon while still averaging out the
# per-interval 0/0 flaps on bursty workloads. A hard floor of 20
# intervals keeps the very short test runs from degenerating into
# per-interval noise.
_ROLLING_WINDOW_FRAC = 0.20
_ROLLING_WINDOW_MIN = 20

_SERIES_COLORS = {
    "static": "#779ebd",
    "pred": "#bdbb55",
    "pred_qp": "#90ac7c",
    "qp": "#7c9559",
    "adaptive": "#7c9559",
}

# Thesis-style fairness overlay: run order and (marker, hollow, dash)
_THESIS_FAIRNESS_STATELESS_ORDER = (
    "static",
    "pred_oneshot",
    "qp_oneshot",
    "pred_qp_oneshot",
)
_THESIS_FAIRNESS_STATEFUL_ORDER = ("static", "proposed")
_THESIS_FAIRNESS_MARKERS = (
    # (marker, use hollow face for square, linestyle)
    ("s", True, "-"),   # hollow square
    ("x", False, "-"),  # cross
    ("o", False, "-"),  # round point
    ("P", False, "-"),  # plus (4th series)
)
_THESIS_FONT_PT = 28
_THESIS_FONT_FAMILY = "Arial"

# Thesis fairness overlay only: matches common tab palette (blue, orange,
# green, yellow) for static / pred / qp / pred_qp.  Stateful second
# curve (``adaptive``) reuses orange so it stays distinct from blue.
_THESIS_FAIRNESS_COLORS = {
    "static": "#1f77b4",
    "pred": "#ff7f0e",
    "qp": "#2ca02c",
    "pred_qp": "#FFD700",
    "adaptive": "#ff7f0e",
}


def _thesis_fairness_color(label: str) -> str:
    return _THESIS_FAIRNESS_COLORS.get(label, "#333333")


def _style_axis(ax) -> None:
    """Apply a lightweight 'academic paper' plotting style."""
    ax.grid(True, which="major", linestyle="--", linewidth=0.6, alpha=0.35)
    ax.grid(True, which="minor", linestyle=":", linewidth=0.5, alpha=0.2)
    ax.tick_params(axis="both", labelsize=9)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)


def _set_unit_ylim(ax, metric: str) -> None:
    """Fix [0, 1] y-range for bounded metrics."""
    if metric in _BOUNDED_METRICS:
        ax.set_ylim(0.0, 1.0)


def _legend_label(name: str, domain: str) -> str:
    """Strip the ``_oneshot`` suffix from scheduler variant names so
    plot legends show ``qp`` / ``pred`` / ``pred_qp`` instead of
    ``qp_oneshot`` / ``pred_oneshot`` / ``pred_qp_oneshot``. Greedy
    variants have been dropped from the evaluation matrix, so the
    suffix carries no information.
    """
    if name.endswith("_oneshot"):
        name = name[: -len("_oneshot")]
    if domain == "stateful" and name == "proposed":
        return "adaptive"
    return name


def _x_col(df: pd.DataFrame) -> str:
    """Prefer new naming while keeping backward compatibility."""
    return "time_interval" if "time_interval" in df.columns else "epoch"


def _x_label() -> str:
    return "time_interval"


def _load(outdir: str, domain: str) -> Optional[pd.DataFrame]:
    p = os.path.join(outdir, f"timeseries_{domain}.csv")
    if not os.path.exists(p):
        return None
    df = pd.read_csv(p)
    df = _attach_cumulative_byte_balance(df)
    df = _attach_rolling_byte_balance(df)
    return df


def _attach_cumulative_byte_balance(df: pd.DataFrame) -> pd.DataFrame:
    """Compute and attach cumulative byte-fairness / imbalance columns.

    Per-interval ``fairness_B`` on a bursty realistic workload returns
    ``1.0`` on every interval where ``total_B_gen == 0`` (the documented
    0/0 fallback in ``metrics.fairness``). At 30 us stateless intervals
    this creates an artificial oscillation between 1.0 and the real
    busy-interval value, which is noise, not scheduler behavior.

    The cumulative variants use the running per-queue byte totals up to
    time ``t``. They are always defined after the first packet, never
    snap to 1.0 on idle intervals, and monotonically converge to the
    steady-state long-run byte share, which is the scheduler quality
    signal we actually want to plot.
    """
    bcols = [c for c in df.columns if c.startswith("B_q_gen[")]
    if not bcols:
        return df
    bcols = sorted(bcols,
                   key=lambda c: int(c.split("[")[1].split("]")[0]))
    cum = df[bcols].cumsum().to_numpy(dtype=np.float64)
    N = cum.shape[1]
    sums = cum.sum(axis=1)
    sq = (cum * cum).sum(axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        fair = (sums * sums) / (N * sq + 1e-20)
    fair = np.where(sums > 0.0, fair, np.nan)
    sorted_cum = np.sort(cum, axis=1)
    idx = np.arange(1, N + 1, dtype=np.float64)
    weighted = (sorted_cum * idx).sum(axis=1)
    gini = np.where(
        sums > 0.0,
        2.0 * weighted / (N * sums + 1e-20) - (N + 1.0) / N,
        np.nan,
    )
    df = df.copy()
    df["fairness_B_cum"] = fair
    df["imbalance_B_cum"] = gini
    return df


def _attach_rolling_byte_balance(df: pd.DataFrame) -> pd.DataFrame:
    """Compute and attach rolling-window byte-fairness / imbalance
    columns (``fairness_B_win`` / ``imbalance_B_win``).

    The cumulative variant integrates from ``t = 0`` and therefore
    carries a long-memory "initial-transient drag": bytes sent before
    the scheduler converged are never forgotten, so the curve sits
    strictly below the steady-state per-interval fairness for the
    entire horizon. That is correct long-run accounting but it hides
    how balanced the scheduler actually is once it has settled.

    The rolling-window variant sums per-queue bytes only over the last
    ``W`` intervals, with
    ``W = max(_ROLLING_WINDOW_MIN, floor(n * _ROLLING_WINDOW_FRAC))``
    where ``n`` is the number of logged intervals. It drops the
    transient after roughly the first ``W`` intervals and then tracks
    the *current* byte share the scheduler is producing, which is the
    right thesis signal for "has the scheduler converged, and to
    what?".
    """
    bcols = [c for c in df.columns if c.startswith("B_q_gen[")]
    if not bcols:
        return df
    bcols = sorted(bcols,
                   key=lambda c: int(c.split("[")[1].split("]")[0]))
    n = len(df)
    window = max(_ROLLING_WINDOW_MIN, int(n * _ROLLING_WINDOW_FRAC))
    window = max(1, min(window, n))
    rsum = (
        df[bcols]
        .rolling(window=window, min_periods=1)
        .sum()
        .to_numpy(dtype=np.float64)
    )
    N = rsum.shape[1]
    sums = rsum.sum(axis=1)
    sq = (rsum * rsum).sum(axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        fair = (sums * sums) / (N * sq + 1e-20)
    fair = np.where(sums > 0.0, fair, np.nan)
    sorted_r = np.sort(rsum, axis=1)
    idx = np.arange(1, N + 1, dtype=np.float64)
    weighted = (sorted_r * idx).sum(axis=1)
    gini = np.where(
        sums > 0.0,
        2.0 * weighted / (N * sums + 1e-20) - (N + 1.0) / N,
        np.nan,
    )
    df = df.copy()
    df["fairness_B_win"] = fair
    df["imbalance_B_win"] = gini
    df.attrs["fairness_B_win_window"] = window
    return df


def plot_run(outdir: str) -> None:
    for domain in ("stateless", "stateful"):
        df = _load(outdir, domain)
        if df is None or df.empty:
            continue
        _plot_pressure(df, outdir, domain)
        _plot_drops(df, outdir, domain)
        _plot_fairness(df, outdir, domain)
        _plot_reassign(df, outdir, domain)
        _plot_tail_latency(df, outdir, domain)


def _plot_pressure(df, outdir, domain):
    fig, ax = plt.subplots(figsize=(8, 4))
    xcol = _x_col(df)
    q_cols = [c for c in df.columns if c.startswith("P_q[")]
    for c in q_cols:
        ax.plot(df[xcol], df[c], alpha=0.4, linewidth=0.8)
    ax.plot(df[xcol], df["P_q_max"], color="k", linewidth=1.6, label="max")
    ax.plot(df[xcol], df["P_q_mean"], color="r", linewidth=1.2, label="mean")
    ax.set_xlabel(_x_label()); ax.set_ylabel("P_q")
    ax.set_title(f"{domain} queue pressure")
    _set_unit_ylim(ax, "P_q")
    _style_axis(ax)
    ax.legend(frameon=False, fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, f"pressure_{domain}.png"), dpi=180)
    plt.close(fig)


def _plot_drops(df, outdir, domain):
    fig, ax = plt.subplots(figsize=(8, 4))
    xcol = _x_col(df)
    ax.plot(df[xcol], df["drop_ratio"], color="C3")
    ax.set_xlabel(_x_label()); ax.set_ylabel("drop ratio")
    ax.set_title(f"{domain} drop ratio")
    _set_unit_ylim(ax, "drop_ratio")
    _style_axis(ax)
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, f"drops_{domain}.png"), dpi=180)
    plt.close(fig)


def _plot_fairness(df, outdir, domain):
    """Balance plot: rolling-window byte-fairness (transient-free
    convergence curve) plus rolling-window byte-Gini. Cumulative
    byte-fairness is drawn faintly in the background as long-run
    reference.

    Per-interval ``fairness_B`` is intentionally omitted: on the
    bursty realistic workload 50%+ of intervals carry zero bytes, so
    the per-interval value flaps between 1.0 (the 0/0 fallback) and
    the real busy-interval fairness. That oscillation is a sampling
    artifact, not scheduler behavior.

    The cumulative curve carries an initial-transient drag: bytes
    sent before the scheduler converged are never forgotten, so the
    curve sits below the current steady-state. The rolling window
    averages only over the last ``W`` intervals and so recovers the
    *current* byte-share the scheduler is producing.
    """
    fig, ax = plt.subplots(figsize=(8, 4))
    xcol = _x_col(df)
    plotted_any = False
    window = df.attrs.get("fairness_B_win_window")
    win_suffix = f", window={window}" if window else ""
    if "fairness_B_win" in df.columns:
        ax.plot(df[xcol], df["fairness_B_win"],
                label=f"fairness_byte (rolling{win_suffix}, 1=perfect)",
                linewidth=1.6)
        plotted_any = True
    if "imbalance_B_win" in df.columns:
        ax.plot(df[xcol], df["imbalance_B_win"],
                label=f"imbalance_byte (rolling Gini{win_suffix}, 0=balanced)",
                linewidth=1.4, alpha=0.9)
        plotted_any = True
    if "fairness_B_cum" in df.columns:
        ax.plot(df[xcol], df["fairness_B_cum"],
                label="fairness_byte (cumulative, long-run)",
                linewidth=0.9, alpha=0.35)
        plotted_any = True
    if "fairness_P" in df.columns:
        ax.plot(df[xcol], df["fairness_P"],
                label="fairness_P (per-interval pressure)",
                linewidth=0.9, alpha=0.25)
        plotted_any = True
    if not plotted_any:
        plt.close(fig)
        return
    ax.set_xlabel(_x_label()); ax.set_ylabel("value")
    ax.set_title(f"{domain} balance metrics")
    _set_unit_ylim(ax, "fairness_B_win")
    _style_axis(ax)
    ax.legend(frameon=False, fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, f"balance_{domain}.png"), dpi=180)
    plt.close(fig)


def _plot_tail_latency(df, outdir, domain):
    """Per-epoch tail latency curves (p50 / p95 / p99 / p99.9 / max)
    in microseconds. Skip if no latency columns (e.g. legacy runs)."""
    cols = [c for c in ("lat_p50_ns", "lat_p95_ns", "lat_p99_ns",
                        "lat_p999_ns", "lat_max_ns")
            if c in df.columns]
    if not cols:
        return
    fig, ax = plt.subplots(figsize=(8, 4))
    xcol = _x_col(df)
    label_map = {
        "lat_p50_ns": "p50",
        "lat_p95_ns": "p95",
        "lat_p99_ns": "p99",
        "lat_p999_ns": "p99.9",
        "lat_max_ns": "max",
    }
    for c in cols:
        ax.plot(df[xcol], df[c] / 1e3, label=label_map[c], alpha=0.9)
    ax.set_xlabel(_x_label()); ax.set_ylabel("latency (us)")
    ax.set_title(f"{domain} end-to-end latency tail")
    ax.set_yscale("log")
    _style_axis(ax)
    ax.legend(frameon=False, fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, f"latency_{domain}.png"), dpi=180)
    plt.close(fig)


def _plot_reassign(df, outdir, domain):
    col = "reassignments" if "reassignments" in df.columns else \
          ("handoffs_committed" if "handoffs_committed" in df.columns else None)
    if col is None:
        return
    fig, ax = plt.subplots(figsize=(8, 3))
    xcol = _x_col(df)
    ax.bar(df[xcol], df[col], width=1.0, color="C2")
    ax.set_xlabel(_x_label()); ax.set_ylabel(col)
    ax.set_title(f"{domain} {col}")
    _style_axis(ax)
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, f"reassign_{domain}.png"), dpi=180)
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
    for metric, ylab, title, fname in [
        ("P_q_max", "max P_q", "peak queue pressure", "P_q_max"),
        ("drop_ratio", "drop ratio", "drop ratio", "drop_ratio"),
        # fairness_P : pressure-based (scheduler input, noisy at 30 us
        # epochs).
        ("fairness_P", "fairness (1 = equal load)",
         "queue-pressure fairness", "fairness_P"),
        # fairness_B_win / imbalance_B_win : rolling-window byte-
        # based balance metrics. The per-queue byte totals are summed
        # over the last W intervals only (W = 20% of the run length,
        # floored at 20 intervals) and Jain / Gini are applied to the
        # resulting windowed vector. This replaces:
        #   * per-interval fairness_B, which oscillates between 1.0
        #     (the 0/0 fallback on idle intervals) and the real
        #     busy-interval fairness -- a sampling artifact.
        #   * cumulative fairness_B_cum, which integrates from t=0
        #     and so drags the entire curve down by the unbalanced
        #     bytes that were generated before the scheduler
        #     converged.
        # The rolling window drops the initial transient after ~W
        # intervals and then tracks the *current* byte share the
        # scheduler is producing -- the clean thesis signal for
        # "has it converged, and to what?". File names are kept as
        # fairness_B / imbalance_B so the thesis pipeline is
        # unchanged.
        ("fairness_B_win",
         "fairness_byte (rolling window, 1 = equal load)",
         "fairness_byte across queues (rolling window)", "fairness_B"),
        ("imbalance_B_win",
         "imbalance_byte (rolling Gini, 0 = equal load)",
         "byte imbalance across queues (rolling window)", "imbalance_B"),
    ]:
        if not any(metric in df.columns for df in dfs.values()):
            continue
        fig, ax = plt.subplots(figsize=(9, 4))
        for name, df in dfs.items():
            if metric in df.columns:
                label = _legend_label(name, domain)
                ax.plot(
                    df[_x_col(df)],
                    df[metric],
                    label=label,
                    color=_SERIES_COLORS.get(label),
                    alpha=0.9,
                    linewidth=1.4,
                )
        ax.set_xlabel(_x_label()); ax.set_ylabel(ylab)
        ax.set_title(f"{domain} {title}: experiment comparison")
        _set_unit_ylim(ax, metric)
        _style_axis(ax)
        ax.legend(frameon=False, fontsize=9)
        fig.tight_layout()
        fig.savefig(os.path.join(outdir, f"compare_{fname}_{domain}.png"),
                    dpi=180)
        plt.close(fig)

    # Tail-latency overlay (log-y, microseconds). One subplot per
    # quantile of interest so curves don't obscure one another.
    for metric, title in [
        ("lat_p50_ns", "p50 latency"),
        ("lat_p99_ns", "p99 latency"),
        ("lat_p999_ns", "p99.9 latency"),
        ("lat_max_ns", "max latency"),
    ]:
        if not any(metric in df.columns for df in dfs.values()):
            continue
        fig, ax = plt.subplots(figsize=(9, 4))
        for name, df in dfs.items():
            if metric in df.columns:
                label = _legend_label(name, domain)
                ax.plot(
                    df[_x_col(df)],
                    df[metric] / 1e3,
                    label=label,
                    color=_SERIES_COLORS.get(label),
                    alpha=0.9,
                    linewidth=1.4,
                )
        ax.set_xlabel(_x_label()); ax.set_ylabel("latency (us)")
        ax.set_yscale("log")
        ax.set_title(f"{domain} {title}: experiment comparison")
        _style_axis(ax)
        ax.legend(frameon=False, fontsize=9)
        fig.tight_layout()
        fig.savefig(os.path.join(outdir,
                                 f"compare_{metric}_{domain}.png"), dpi=180)
        plt.close(fig)


def plot_thesis_fairness_B_comparison(
    runs: Dict[str, str],
    outdir: str,
    domain: str = "stateless",
    include: Optional[list] = None,
) -> None:
    """Overlay ``fairness_B_win`` with thesis-style formatting (Arial 28,
    legend above axes, distinct markers, horizontal grid only, tight
    crop).  Writes ``compare_fairness_B_<domain>.png`` into ``outdir``.

    Layout matches the attached reference: **time interval** on the
    horizontal axis (replacing ``Packet Size (B)`` in the template) and
    **byte-level fairness** (Jain index in ``[0, 1]``) on the vertical
    axis (replacing ``Throughput Ratio``).  X tick labels are rotated
    90°.  Y is zoomed to ``[0.4, 1.0]`` for a compact view of the
    fairness band where the curves separate.
    """
    from matplotlib import ticker

    os.makedirs(outdir, exist_ok=True)
    dfs = {name: _load(d, domain) for name, d in runs.items()}
    dfs = {k: v for k, v in dfs.items() if v is not None and not v.empty}
    if include is not None:
        dfs = {k: dfs[k] for k in include if k in dfs}
    if not dfs or not any("fairness_B_win" in df.columns for df in dfs.values()):
        return

    order = (list(_THESIS_FAIRNESS_STATELESS_ORDER) if domain == "stateless"
             else list(_THESIS_FAIRNESS_STATEFUL_ORDER))

    thesis_rc = {
        "font.family": "sans-serif",
        "font.sans-serif": [_THESIS_FONT_FAMILY, "DejaVu Sans", "Helvetica",
                            "Liberation Sans"],
        "font.size": _THESIS_FONT_PT,
        "axes.titlesize": _THESIS_FONT_PT,
        "axes.labelsize": _THESIS_FONT_PT,
        "xtick.labelsize": _THESIS_FONT_PT,
        "ytick.labelsize": _THESIS_FONT_PT,
        "legend.fontsize": _THESIS_FONT_PT,
    }

    with plt.rc_context(rc=thesis_rc):
        fig, ax = plt.subplots(figsize=(10.5, 4.8))
        plotted = 0
        for name in order:
            if name not in dfs:
                continue
            df = dfs[name]
            if "fairness_B_win" not in df.columns:
                continue
            mk, hollow, ls = _THESIS_FAIRNESS_MARKERS[
                plotted % len(_THESIS_FAIRNESS_MARKERS)]
            label = _legend_label(name, domain)
            color = _thesis_fairness_color(label)
            xcol = _x_col(df)
            x = df[xcol].to_numpy()
            y = df["fairness_B_win"].to_numpy()
            n = len(x)
            markevery = max(1, n // 28)
            plot_kw: dict = {
                "color": color,
                "linewidth": 2.4,
                "linestyle": ls,
                "label": label,
                "marker": mk,
                "markersize": 11,
                "markevery": markevery,
                "markeredgewidth": 1.8,
                "clip_on": False,
            }
            if mk == "s" and hollow:
                plot_kw["markerfacecolor"] = "none"
                plot_kw["markeredgecolor"] = color
            else:
                plot_kw["markerfacecolor"] = color
                plot_kw["markeredgecolor"] = color
            ax.plot(x, y, **plot_kw)
            plotted += 1

        if plotted == 0:
            plt.close(fig)
            return

        ax.set_ylim(0.4, 1.0)
        ax.yaxis.set_major_locator(ticker.MultipleLocator(0.1))
        ax.xaxis.set_major_locator(ticker.MaxNLocator(nbins=12, integer=True))
        ax.set_xlabel("Time interval")
        if domain == "stateful":
            # Nudge stateful x-label down to match stateless baseline
            # when figures are placed side-by-side.
            ax.xaxis.labelpad = 12
        ax.set_ylabel("Byte-level fairness")
        ax.yaxis.grid(True, which="major", linestyle="-", linewidth=0.9,
                    color="#d0d0d0", zorder=0)
        ax.set_axisbelow(True)
        ax.grid(False, which="minor", axis="x")
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
        ax.tick_params(axis="both", which="major", direction="out",
                       length=5, width=1.0)

        leg = ax.legend(
            ncol=plotted,
            loc="lower center",
            bbox_to_anchor=(0.5, 1.02),
            frameon=False,
            handlelength=2.0,
            handletextpad=0.45,
            columnspacing=1.0,
        )
        if leg is not None:
            for line in leg.get_lines():
                line.set_linewidth(2.4)
                line.set_markersize(11)

        plt.setp(ax.get_xticklabels(), rotation=90, ha="center", va="top")

        fig.subplots_adjust(top=0.82, bottom=0.26, left=0.14, right=0.98)
        outp = os.path.join(outdir, f"compare_fairness_B_{domain}.png")
        fig.savefig(
            outp,
            dpi=200,
            bbox_inches="tight",
            pad_inches=0.06,
        )
        if domain == "stateful":
            # Keep stateless rendering untouched; pad stateful export to
            # the same geometry as the original stateless thesis figure
            # so side-by-side placement aligns axes/titles/legend.
            try:
                from PIL import Image

                target_w, target_h = 2004, 1021
                im = Image.open(outp)
                w, h = im.size
                if w < target_w or h < target_h:
                    canvas = Image.new("RGBA", (target_w, target_h),
                                       (255, 255, 255, 255))
                    # Anchor at top-left so left/top axis geometry stays
                    # identical; extra space is added on right/bottom.
                    canvas.paste(im, (0, 0))
                    canvas.save(outp)
            except Exception:
                pass
        plt.close(fig)
