#!/usr/bin/env python3
"""Run all configs in a suite side-by-side and produce a comparison
report + overlay plots.

Preset suites:

Phase-1 isolated-domain evaluation (recommended first, one domain at a
time with the other fully disabled so the scheduler is measured
without cross-domain PCIe interference or scheduling noise):

* ``--suite stateless_only`` -> runs ``configs/stateless_only/*.yaml``.
  Only the stateless domain runs; the stateful domain is off.
* ``--suite stateful_only``  -> runs ``configs/stateful_only/*.yaml``.
  Only the stateful domain runs; the stateless domain is off.

Phase-2 coexistence / realism evaluation (both domains enabled, shared
PCIe link, one scheduler under test while the other is held on a
canonical baseline):

* ``--suite stateless`` -> runs ``configs/stateless/*.yaml``.
* ``--suite stateful``  -> runs ``configs/stateful/*.yaml``.

You can also pass ``--configs-dir PATH`` to run a custom set, or
``--configs f1.yaml f2.yaml ...`` to enumerate explicit files.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import time
from typing import Optional

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

from src.config import Config
from src.sim import Simulator
from src import plotting


def run_one(path: str, base_outdir: str) -> tuple[str, str]:
    cfg = Config.load(path)
    outdir = os.path.join(base_outdir, cfg.experiment.name)
    os.makedirs(outdir, exist_ok=True)
    t0 = time.time()
    sim = Simulator(cfg)
    sim.run()
    sim.save_all(outdir)
    if cfg.experiment.make_plots:
        plotting.plot_run(outdir)
    print(f"  [{cfg.experiment.name}] done in {time.time()-t0:.1f}s")
    return cfg.experiment.name, outdir


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--suite",
        choices=("stateless_only", "stateful_only",
                 "stateless", "stateful", "custom"),
        default="custom",
        help=("preset run: *_only for isolated-domain phase-1 evaluation; "
              "stateless/stateful for phase-2 coexistence"),
    )
    ap.add_argument("--configs-dir", default=None,
                    help="only used when --suite=custom")
    ap.add_argument("--out", default=None,
                    help="override output dir (default: "
                         "results/comparison_<suite>)")
    ap.add_argument("--configs", nargs="*", default=None,
                    help="explicit config files (overrides suite/dir)")
    args = ap.parse_args()

    root = os.path.dirname(HERE)

    if args.configs:
        files = args.configs
        out = args.out or os.path.join(root, "results", "comparison_custom")
        plot_stateless = True
        plot_stateful = True
    elif args.suite == "stateless_only":
        dir_path = os.path.join(root, "configs", "stateless_only")
        files = sorted(glob.glob(os.path.join(dir_path, "*.yaml")))
        out = args.out or os.path.join(root, "results",
                                       "comparison_stateless_only")
        plot_stateless = True
        plot_stateful = False
    elif args.suite == "stateful_only":
        dir_path = os.path.join(root, "configs", "stateful_only")
        files = sorted(glob.glob(os.path.join(dir_path, "*.yaml")))
        out = args.out or os.path.join(root, "results",
                                       "comparison_stateful_only")
        plot_stateless = False
        plot_stateful = True
    elif args.suite == "stateless":
        dir_path = os.path.join(root, "configs", "stateless")
        files = sorted(glob.glob(os.path.join(dir_path, "*.yaml")))
        out = args.out or os.path.join(root, "results",
                                       "comparison_stateless")
        plot_stateless = True
        plot_stateful = False
    elif args.suite == "stateful":
        dir_path = os.path.join(root, "configs", "stateful")
        files = sorted(glob.glob(os.path.join(dir_path, "*.yaml")))
        out = args.out or os.path.join(root, "results",
                                       "comparison_stateful")
        plot_stateless = False
        plot_stateful = True
    else:
        dir_path = args.configs_dir or os.path.join(root, "configs")
        files = sorted(glob.glob(os.path.join(dir_path, "*.yaml")))
        out = args.out or os.path.join(root, "results", "comparison_custom")
        plot_stateless = True
        plot_stateful = True

    os.makedirs(out, exist_ok=True)
    print(f"Suite: {args.suite}")
    print(f"Running {len(files)} configs:")
    for f in files:
        print(f"  {f}")

    runs: dict[str, str] = {}
    summaries: dict[str, dict] = {}
    for f in files:
        name, outdir = run_one(f, out)
        runs[name] = outdir
        sp = os.path.join(outdir, "summary.json")
        if os.path.exists(sp):
            with open(sp) as fh:
                summaries[name] = json.load(fh)

    if plot_stateless:
        plotting.plot_comparison(runs, out, domain="stateless")
    if plot_stateful:
        # Include every provided run when running a stateful suite; for
        # other suites, only compare static vs the blended-greedy winner
        # to avoid drowning out the relevant curves.
        if args.suite in ("stateful", "stateful_only"):
            stateful_runs = list(runs.keys())
        else:
            stateful_runs = [n for n in ("static", "pred_qp_greedy")
                             if n in runs]
        plotting.plot_comparison(runs, out, domain="stateful",
                                 include=stateful_runs)

    rows = []
    for name, s in summaries.items():
        sl = s.get("stateless", {}); sf = s.get("stateful", {})
        wr = s.get("workload_realism", {})
        def _max_err(side):
            xs = wr.get(side, [])
            if not xs:
                return 0.0
            return max(abs(x.get("rate_error_fraction", 0.0)) for x in xs)
        rows.append({
            "config": name,
            "sl_drop_ratio": sl.get("overall_drop_ratio", 0.0),
            "sl_mean_Pmax": sl.get("mean_P_max", 0.0),
            "sl_p99_Pmax": sl.get("p99_P_max", 0.0),
            "sl_mean_fairness": sl.get("mean_fairness_P", 0.0),
            "sl_mean_imbalance": sl.get("mean_imbalance_B", 0.0),
            "sl_total_reassign": sl.get("total_reassignments", 0),
            "sl_rate_err_max": _max_err("stateless"),
            # End-to-end tail latency (epoch-averaged quantiles + worst).
            "sl_mean_lat_p99_us": sl.get("mean_lat_p99_ns", 0.0) / 1e3,
            "sl_mean_lat_p999_us": sl.get("mean_lat_p999_ns", 0.0) / 1e3,
            "sl_worst_lat_p99_us": sl.get("worst_lat_p99_ns", 0.0) / 1e3,
            "sl_worst_lat_max_us": sl.get("worst_lat_max_ns", 0.0) / 1e3,
            "sf_drop_ratio": sf.get("overall_drop_ratio", 0.0),
            "sf_mean_Pmax": sf.get("mean_P_max", 0.0),
            "sf_mean_fairness": sf.get("mean_fairness_P", 0.0),
            "sf_mean_imbalance": sf.get("mean_imbalance_B", 0.0),
            "sf_handoffs_committed": sf.get("total_reassignments", 0),
            "sf_rate_err_max": _max_err("stateful"),
            "sf_mean_lat_p99_us": sf.get("mean_lat_p99_ns", 0.0) / 1e3,
            "sf_mean_lat_p999_us": sf.get("mean_lat_p999_ns", 0.0) / 1e3,
            "sf_worst_lat_p99_us": sf.get("worst_lat_p99_ns", 0.0) / 1e3,
            "sf_worst_lat_max_us": sf.get("worst_lat_max_ns", 0.0) / 1e3,
        })
    import pandas as pd
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(out, "comparison_summary.csv"), index=False)
    with open(os.path.join(out, "comparison_summary.json"), "w") as f:
        json.dump(summaries, f, indent=2, default=float)

    print("\nComparison summary:")
    print(df.to_string(index=False))
    print(f"\nOutputs under: {out}")


if __name__ == "__main__":
    main()
