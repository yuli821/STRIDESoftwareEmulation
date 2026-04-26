#!/usr/bin/env python3
"""Re-render all plots under ``results/`` from existing CSVs.

Useful after a plotting-side change (e.g. adding cumulative
byte-fairness) when we don't want to re-run the simulator. For every
``results/**/comparison_<suite>/`` directory we find per-run
subdirectories that contain ``timeseries_<domain>.csv`` and replot
both the per-run and the overlay comparison plots.
"""
from __future__ import annotations

import argparse
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

from src import plotting  # noqa: E402


def _iter_comparison_dirs(root: str):
    for dirpath, dirnames, _ in os.walk(root):
        base = os.path.basename(dirpath)
        if base.startswith("comparison_"):
            yield dirpath


def _collect_runs(comp_dir: str) -> dict[str, str]:
    runs: dict[str, str] = {}
    for name in sorted(os.listdir(comp_dir)):
        sub = os.path.join(comp_dir, name)
        if not os.path.isdir(sub):
            continue
        has_ts = any(
            os.path.exists(os.path.join(sub, f"timeseries_{d}.csv"))
            for d in ("stateless", "stateful")
        )
        if has_ts:
            runs[name] = sub
    return runs


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=None,
                    help="results/ root (default: <repo>/results)")
    args = ap.parse_args()

    root = args.root or os.path.join(os.path.dirname(HERE), "results")
    if not os.path.isdir(root):
        raise SystemExit(f"no such directory: {root}")

    comp_dirs = list(_iter_comparison_dirs(root))
    print(f"found {len(comp_dirs)} comparison directories under {root}")

    for cd in comp_dirs:
        runs = _collect_runs(cd)
        if not runs:
            continue
        for sub in runs.values():
            plotting.plot_run(sub)
        base = os.path.basename(cd)
        suite = base[len("comparison_"):] if base.startswith("comparison_") \
            else base
        if suite in ("stateless_only", "stateless"):
            plotting.plot_comparison(runs, cd, domain="stateless")
        elif suite in ("stateful_only", "stateful"):
            plotting.plot_comparison(runs, cd, domain="stateful")
        else:
            plotting.plot_comparison(runs, cd, domain="stateless")
            plotting.plot_comparison(runs, cd, domain="stateful")
        print(f"  replotted: {cd}  ({len(runs)} runs)")


if __name__ == "__main__":
    main()
