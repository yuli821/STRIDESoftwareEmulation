#!/usr/bin/env python3
"""Collect (X, y) training pairs for the TCN predictor.

For every (workload, RSS, seed) combination declared in the ML config
(see ``configs/ml/tcn_pred.yaml``), this script spins up the emulator
with:

  - ``stateless_scheduler.scheduler_type = static``   (RSS never moves)
  - ``predictor_type = none``                         (no predictor output
                                                       leaks into the trace)
  - ``enable_stateful = false``                       (only stateless
                                                       epoch boundaries
                                                       matter)

It swaps the sim's predictor for a tiny recorder that keeps every
per-epoch feature tensor ``feat[t] = (N, 4)``. After the run it turns
those per-run timelines into (X, y) pairs where

  X[t, q] = feat[t - W + 1 : t + 1, q, :].T   # shape (4, W)
  y[t, q] = feat[t + 1, q, 3]                 # P_q at t+1, in [0, 1]

train/val splits are held out BY RUN so every epoch of a val run is
strictly unseen at training time, and per-channel mean / std are
computed from the TRAIN fold only.

Output: ``data/tcn_dataset.npz`` with

    X_train      : (M_train, 4, W) float32
    y_train      : (M_train,)      float32
    w_train      : (M_train,)      float32   # loss weights (target_weight_k)
    X_val        : (M_val,   4, W) float32
    y_val        : (M_val,)        float32
    w_val        : (M_val,)        float32
    channel_mean : (4,)            float32   # train-fold per-channel mean
    channel_std  : (4,)            float32   # train-fold per-channel std
    W            : ()              int32
    train_runs   : list[str]                 # for reproducibility
    val_runs     : list[str]
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from typing import Any, Dict, List, Tuple

import numpy as np
import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

from src.config import Config  # noqa: E402
from src.sim import Simulator  # noqa: E402
from src.predictors.base import BasePredictor  # noqa: E402


# ---------------------------------------------------------------------------
# Recording predictor: observes features, returns zeros. Used at dataset
# collection time ONLY -- the scheduler is forced to static so the
# zero output is never consulted.
# ---------------------------------------------------------------------------
class _RecordingPredictor(BasePredictor):
    def __init__(self, num_queues: int, window: int) -> None:
        super().__init__(num_queues, window)
        self.feats: List[np.ndarray] = []

    def observe(self, feat: np.ndarray) -> None:
        super().observe(feat)
        self.feats.append(feat.copy())

    def predict(self) -> np.ndarray:
        return np.zeros(self.num_queues, dtype=np.float64)


# ---------------------------------------------------------------------------
# Config assembly: reuse scripts/gen_configs.BASE and overlay whatever
# the workload spec in the ML YAML asks for. We rebuild the sim's
# BASE config in-code rather than reading one of the gen_*_configs
# outputs because we want explicit control over every knob and
# zero coupling to the eval YAMLs.
# ---------------------------------------------------------------------------
def _build_cfg(workload_spec: Dict[str, Any], *, rss: str, seed: int,
               num_epochs_stateless: int,
               force_scheduler: str, force_predictor: str) -> Config:
    # Lazily import so the script runs even if gen_configs.BASE is
    # edited while this script is live.
    sys.path.insert(0, HERE)
    from gen_configs import BASE, _deepcopy  # type: ignore

    cfg: Dict[str, Any] = _deepcopy(BASE)

    # Time: short runs for fast dataset collection.
    cfg["time"]["num_epochs"] = int(num_epochs_stateless)
    cfg["time"]["num_epochs_stateless"] = int(num_epochs_stateless)
    cfg["time"]["num_epochs_stateful"] = 0

    # Topology: match the stateless_only / TEST setup (4 queues, 4 cores)
    # used in both evaluation suites so the model sees the same queue
    # count at training and inference time.
    topo = cfg["topology"]
    topo["num_cores_stateless"] = 4
    topo["num_stateless_queues"] = 4
    topo["descriptor_ring_depth_stateless"] = 512
    topo["initial_rss_stateless"] = rss

    host = cfg["host"]
    host["stateless_t_app_mean_ns"] = 1000.0
    host["stateless_t_app_jitter_ns"] = 150.0
    host["pcie_bandwidth_gbps"] = 128.0

    # Workload: copy whatever the YAML spec provides into cfg["workload"].
    w = cfg["workload"]
    for k, v in workload_spec.items():
        if k == "name":
            continue
        w[k] = v

    # Predictor / scheduler force-overrides.
    cfg["predictor"]["predictor_type"] = force_predictor
    cfg["predictor"]["W_window_epochs"] = int(cfg["predictor"]
                                              .get("W_window_epochs", 8))
    cfg["predictor"]["tcn_checkpoint"] = ""

    cfg["stateless_scheduler"]["scheduler_type"] = force_scheduler
    # Static scheduler shouldn't move buckets; still good to cap moves.
    cfg["stateless_scheduler"]["max_moves_per_epoch"] = 0

    cfg["stateful_scheduler"]["scheduler_type"] = "static"
    cfg["stateful_scheduler"]["max_concurrent_handoffs"] = 0

    # Experiment: disable stateful + plot/log + silence.
    exp = cfg["experiment"]
    exp["name"] = f"tcn_ds_{workload_spec['name']}_{rss}_s{seed}"
    exp["rng_seed"] = int(seed)
    exp["enable_stateful"] = False
    exp["enable_stateless"] = True
    exp["log_time_series"] = False
    exp["log_per_bucket_trace"] = False
    exp["make_plots"] = False

    return Config.from_dict(cfg)


def _run_one(cfg: Config) -> np.ndarray:
    """Run the sim and return its recorded feat timeline, shape
    ``(T, N_queues, 4)``."""
    sim = Simulator(cfg)
    rec = _RecordingPredictor(sim.stateless.num_queues,
                              cfg.predictor.W_window_epochs)
    sim.predictor = rec
    sim.run()
    if not rec.feats:
        return np.zeros((0, sim.stateless.num_queues, 4), dtype=np.float32)
    return np.stack(rec.feats, axis=0).astype(np.float32)


def _windows_from_timeline(timeline: np.ndarray, W: int
                           ) -> Tuple[np.ndarray, np.ndarray]:
    """Turn a (T, N, 4) feat timeline into (X, y) pairs.

    ``X[i] = timeline[t-W+1 : t+1, q, :].T`` (shape (4, W)).
    ``y[i] = timeline[t+1, q, 3]`` (P_q at t+1 in [0, 1]).

    Valid t spans W-1 .. T-2; queue index q is flattened into the
    sample axis.
    """
    T, N, C = timeline.shape
    if T <= W:
        return (np.zeros((0, C, W), dtype=np.float32),
                np.zeros((0,), dtype=np.float32))
    # Pre-allocate.
    n_samples = (T - W) * N
    X = np.zeros((n_samples, C, W), dtype=np.float32)
    y = np.zeros((n_samples,), dtype=np.float32)
    idx = 0
    for t in range(W - 1, T - 1):
        # window of shape (W, N, C)
        win = timeline[t - W + 1 : t + 1]
        # target at t+1
        tgt = timeline[t + 1, :, 3]
        # per-queue samples
        for q in range(N):
            X[idx] = win[:, q, :].T          # (C, W)
            y[idx] = tgt[q]
            idx += 1
    X = X[:idx]
    y = y[:idx]
    # Target is P_q which is bounded by construction; clip as defence.
    y = np.clip(y, 0.0, 1.0)
    return X, y


# ---------------------------------------------------------------------------
# Main driver.
# ---------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ml-config", default="configs/ml/tcn_pred.yaml")
    ap.add_argument("--out", default=None,
                    help="override the dataset path (default: YAML "
                         "data.dataset_path)")
    args = ap.parse_args()

    ml_path = args.ml_config
    if not os.path.isabs(ml_path):
        ml_path = os.path.join(ROOT, ml_path)
    with open(ml_path) as f:
        ml = yaml.safe_load(f)

    W = int(ml["model"]["window"])
    data_cfg = ml["data"]
    out_path = args.out or os.path.join(ROOT, data_cfg["dataset_path"])
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    def _run_split(seeds: List[int]) -> List[Tuple[str, np.ndarray]]:
        out: List[Tuple[str, np.ndarray]] = []
        for ws in data_cfg["train_workloads"]:
            for rss in data_cfg["rss_variants"]:
                for seed in seeds:
                    tag = f"{ws['name']}/{rss}/s{seed}"
                    t0 = time.time()
                    cfg = _build_cfg(
                        ws, rss=rss, seed=int(seed),
                        num_epochs_stateless=int(
                            data_cfg["num_epochs_stateless"]),
                        force_scheduler=data_cfg["force_scheduler"],
                        force_predictor=data_cfg["force_predictor"],
                    )
                    tl = _run_one(cfg)
                    dt = time.time() - t0
                    print(f"  [{tag}] T={tl.shape[0]:4d} "
                          f"epochs  ({dt:.1f}s)")
                    out.append((tag, tl))
        return out

    print("collecting TRAIN runs")
    train_runs = _run_split(list(data_cfg["train_seeds"]))
    print("collecting VAL runs")
    val_runs = _run_split(list(data_cfg["val_seeds"]))

    # Windows.
    Xt_list, yt_list = [], []
    for tag, tl in train_runs:
        Xw, yw = _windows_from_timeline(tl, W)
        Xt_list.append(Xw); yt_list.append(yw)
    Xv_list, yv_list = [], []
    for tag, tl in val_runs:
        Xw, yw = _windows_from_timeline(tl, W)
        Xv_list.append(Xw); yv_list.append(yw)

    X_train = np.concatenate(Xt_list, axis=0) if Xt_list else np.zeros(
        (0, 4, W), dtype=np.float32)
    y_train = np.concatenate(yt_list, axis=0) if yt_list else np.zeros(
        (0,), dtype=np.float32)
    X_val = np.concatenate(Xv_list, axis=0) if Xv_list else np.zeros(
        (0, 4, W), dtype=np.float32)
    y_val = np.concatenate(yv_list, axis=0) if yv_list else np.zeros(
        (0,), dtype=np.float32)

    # Per-channel normalization computed on TRAIN only, collapsing
    # over (samples, time).
    if X_train.size > 0:
        mean = X_train.mean(axis=(0, 2)).astype(np.float32)        # (4,)
        std = X_train.std(axis=(0, 2)).astype(np.float32)
        # Guard zero / near-zero std (constant channel); fallback to 1
        # so the (x-mu)/sigma pass becomes (x-mu).
        std = np.where(std > 1e-8, std, 1.0).astype(np.float32)
    else:
        mean = np.zeros(4, dtype=np.float32)
        std = np.ones(4, dtype=np.float32)

    # Sample weights per the ML YAML knob (default k=0 -> uniform).
    k = float(ml["training"]["target_weight_k"])
    w_train = (1.0 + k * y_train).astype(np.float32)
    w_val = (1.0 + k * y_val).astype(np.float32)

    # Log dataset statistics so the user can sanity-check quickly.
    print()
    print("=== dataset summary ===")
    print(f"  train: X {X_train.shape}  y {y_train.shape}")
    print(f"  val  : X {X_val.shape}    y {y_val.shape}")
    print(f"  W    : {W}")
    print(f"  channel_mean : {mean.tolist()}")
    print(f"  channel_std  : {std.tolist()}")
    print(f"  y_train mean : {float(y_train.mean()) if y_train.size else 0:.4f}")
    print(f"  y_train > 0.2 fraction : "
          f"{float((y_train > 0.2).mean()) if y_train.size else 0:.3f}")
    print(f"  sample weight mean (k={k}) : "
          f"{float(w_train.mean()) if w_train.size else 1.0:.4f}")

    np.savez(
        out_path,
        X_train=X_train,
        y_train=y_train,
        w_train=w_train,
        X_val=X_val,
        y_val=y_val,
        w_val=w_val,
        channel_mean=mean,
        channel_std=std,
        W=np.int32(W),
        train_runs=np.array([t for t, _ in train_runs]),
        val_runs=np.array([t for t, _ in val_runs]),
    )
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
