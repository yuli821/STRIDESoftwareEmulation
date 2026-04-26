#!/usr/bin/env python3
"""Regenerate thesis-style ``compare_fairness_B_*.png`` for synthetic and
realistic workloads (stateless + stateful) using existing CSVs.

See ``plotting.plot_thesis_fairness_B_comparison`` for styling (Arial 28,
legend above axes, markers, tight crop).
"""
from __future__ import annotations

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

from src import plotting  # noqa: E402


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


TARGETS = [
    os.path.join(ROOT, "results/synthetic/comparison_stateless_only"),
    os.path.join(ROOT, "results/synthetic/comparison_stateful_only"),
    os.path.join(ROOT, "results/test/hal_hadoop_only/comparison_stateless_only"),
    os.path.join(ROOT, "results/test/hal_hadoop_only/comparison_stateful_only"),
]


def main() -> None:
    for comp_dir in TARGETS:
        if not os.path.isdir(comp_dir):
            print(f"skip (missing): {comp_dir}")
            continue
        base = os.path.basename(comp_dir)
        suite = base[len("comparison_"):] if base.startswith("comparison_") \
            else base
        runs = _collect_runs(comp_dir)
        if not runs:
            print(f"skip (no runs): {comp_dir}")
            continue
        if suite in ("stateless_only", "stateless"):
            plotting.plot_thesis_fairness_B_comparison(
                runs, comp_dir, domain="stateless")
            print(f"thesis fairness B (stateless): {comp_dir}")
        elif suite in ("stateful_only", "stateful"):
            plotting.plot_thesis_fairness_B_comparison(
                runs, comp_dir, domain="stateful")
            print(f"thesis fairness B (stateful): {comp_dir}")
        else:
            print(f"skip (unexpected suite {suite!r}): {comp_dir}")


if __name__ == "__main__":
    main()
