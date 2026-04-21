#!/usr/bin/env python3
"""Run all configs side-by-side and produce a comparison report."""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import time

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
    sim = Simulator(cfg); sim.run(); sim.save_all(outdir)
    if cfg.experiment.make_plots:
        plotting.plot_run(outdir)
    print(f"  [{cfg.experiment.name}] done in {time.time()-t0:.1f}s")
    return cfg.experiment.name, outdir


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--configs-dir", default="configs")
    ap.add_argument("--out", default="results/comparison")
    ap.add_argument("--configs", nargs="*", default=None)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    files = args.configs or sorted(glob.glob(os.path.join(args.configs_dir, "*.yaml")))
    print(f"Running {len(files)} configs:")
    for f in files:
        print(f"  {f}")

    runs: dict[str, str] = {}
    summaries: dict[str, dict] = {}
    for f in files:
        name, outdir = run_one(f, args.out)
        runs[name] = outdir
        sp = os.path.join(outdir, "summary.json")
        if os.path.exists(sp):
            with open(sp) as fh:
                summaries[name] = json.load(fh)

    plotting.plot_comparison(runs, args.out, domain="stateless")
    stateful_runs = [n for n in ("static", "ewma_greedy") if n in runs]
    plotting.plot_comparison(runs, args.out, domain="stateful",
                             include=stateful_runs)

    rows = []
    for name, s in summaries.items():
        sl = s.get("stateless", {}); sf = s.get("stateful", {})
        rows.append({
            "config": name,
            "sl_drop_ratio": sl.get("overall_drop_ratio", 0.0),
            "sl_mean_Pmax": sl.get("mean_P_max", 0.0),
            "sl_p99_Pmax": sl.get("p99_P_max", 0.0),
            "sl_mean_fairness": sl.get("mean_fairness_P", 0.0),
            "sl_mean_imbalance": sl.get("mean_imbalance_B", 0.0),
            "sl_total_reassign": sl.get("total_reassignments", 0),
            "sf_drop_ratio": sf.get("overall_drop_ratio", 0.0),
            "sf_mean_Pmax": sf.get("mean_P_max", 0.0),
            "sf_mean_fairness": sf.get("mean_fairness_P", 0.0),
            "sf_mean_imbalance": sf.get("mean_imbalance_B", 0.0),
            "sf_handoffs_committed": sf.get("total_reassignments", 0),
        })
    import pandas as pd
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(args.out, "comparison_summary.csv"), index=False)
    with open(os.path.join(args.out, "comparison_summary.json"), "w") as f:
        json.dump(summaries, f, indent=2, default=float)

    print("\nComparison summary:"); print(df.to_string(index=False))
    print(f"\nOutputs under: {args.out}")


if __name__ == "__main__":
    main()
