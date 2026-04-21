#!/usr/bin/env python3
"""Run a single experiment from a YAML config."""
from __future__ import annotations

import argparse
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

from src.config import Config
from src.sim import Simulator
from src import plotting


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--outdir", default=None)
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()

    cfg = Config.load(args.config)
    outdir = args.outdir or os.path.join(cfg.experiment.results_dir,
                                         cfg.experiment.name)
    os.makedirs(outdir, exist_ok=True)

    t0 = time.time()
    sim = Simulator(cfg)
    sim.run()
    sim.save_all(outdir)
    dt = time.time() - t0

    if cfg.experiment.make_plots:
        plotting.plot_run(outdir)

    if not args.quiet:
        print(f"[{cfg.experiment.name}] done in {dt:.1f}s -> {outdir}")
        summary_path = os.path.join(outdir, "summary.json")
        if os.path.exists(summary_path):
            with open(summary_path) as f:
                print(f.read())


if __name__ == "__main__":
    main()
