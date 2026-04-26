#!/usr/bin/env python3
"""Regenerate all per-experiment YAML files from a shared baseline.

Layout
------

``configs/hal_<mix>/stateless_only/*.yaml`` and
``configs/hal_<mix>/stateful_only/*.yaml`` are generated for every
composite mix in ``MIXES`` below. The top-level ``configs/stateless/``
and ``configs/stateful/`` phase-2 coexistence suites are also emitted
using the ``balanced`` mix so the old plumbing keeps working.

For every mix we emit the full scheduler sweep (STATELESS_VARIANTS for
stateless_only, STATEFUL_VARIANTS for stateful_only).

Run whenever you change any shared workload / host / topology knob.
"""
from __future__ import annotations

import os
import sys
import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)


# ----------------------------------------------------------------------
# Shared baseline.
# ----------------------------------------------------------------------
BASE = {
    "time": {
        "clk_period_ns": 4.0,
        "delta_bin_ns": 10000.0,          # 10 us bins
        "H_bins_per_epoch": 10,           # legacy fallback
        # Per-domain epoch budgets:
        #   stateless: 2000 epochs * 3 bins * 10 us = 60 ms
        #   stateful :  500 epochs * 100 bins * 10 us = 500 ms
        # Stateful horizon is intentionally shorter than a naive
        # num_epochs=2000 setup (which would yield 2 s of stateful
        # traffic and ~5.4M packets to simulate through the Python
        # hot loop). 500 epochs is still a large statistical sample
        # (500 Layer-1 lognormal rate draws) and enough for the
        # stateful scheduler to complete dozens of handoff migrations.
        "num_epochs": 2000,
        "num_epochs_stateless": 0,        # 0 -> fall back to num_epochs (2000)
        "num_epochs_stateful": 500,
        "stateless_epoch_bins": 3,        # 30 us per stateless epoch
        "stateful_epoch_bins": 100,       # 1 ms per stateful epoch
    },
    "topology": {
        "num_stateless_queues": 8,
        "num_stateful_queues": 8,
        "num_stateless_buckets": 128,
        "num_stateful_buckets": 128,
        # 8 cores per domain (= 8 RX queues), per user spec.
        "num_cores_stateless": 8,
        "num_cores_stateful": 8,
        "queue_to_core_map_stateless": "one_to_one",
        "queue_to_core_map_stateful": "one_to_one",
        # Deterministic initial RSS: bucket_id % num_queues.
        # With the HAL-composite workload the burstiness is already
        # bursty enough from the log-normal rate process and the
        # mice/elephant size mixture; we don't need random initial
        # assignment to force hot-bucket collisions.
        "initial_rss_stateless": "modulo",
        "initial_rss_stateful": "modulo",
        "descriptor_ring_depth_stateless": 2048,
        "descriptor_ring_depth_stateful": 2048,
    },
    "workload": {
        # Huang et al. (HAL, ISCA'24) two-layer bursty composite
        # workload. See src/hal_workload.py. Per-class mix and total
        # offered load are overridden in make_* below.
        "source": "hal_composite",
        "trace_mix_stateless": [],        # unused under hal_composite
        "trace_mix_stateful":  [],        # unused under hal_composite
        "per_flow_rate_gbps": 10.0,
        "mtu_bytes": 1500,
        # HAL composite knobs (overridden per mix).
        "hal_mix_web":    1.0 / 3.0,
        "hal_mix_cache":  1.0 / 3.0,
        "hal_mix_hadoop": 1.0 / 3.0,
        # Aggregate offered load. We target ~78% of PCIe (64 Gbps)
        # so bursts from the lognormal tail drive real PCIe
        # contention without the link being permanently saturated.
        "hal_total_gbps": 50.0,
        # HAL Fig. 8 uses a 100 Gbps link clip for the rate process.
        "hal_link_gbps": 100.0,
        # 1 ms matches the visible time-scale of rate change in
        # HAL Fig. 8 (Huang et al. ISCA'24).
        "hal_rate_update_ns": 1_000_000.0,
        "max_link_gbps": 200.0,
        "hash_mode": "toeplitz",
        "pattern_shift_period_epochs": 300,
    },
    "host": {
        "stateless_t_app_mean_ns": 400.0,
        "stateless_t_app_jitter_ns": 60.0,
        "stateful_t_app_mean_ns": 2000.0,
        "stateful_t_app_jitter_ns": 400.0,
        "stateful_per_conn_lookup_ns": 80.0,
        "t_writeback_ns": 500.0,
        "rtt_calibration_t_app_base_ns": 300.0,
        "rtt_table": [
            [64,   3800.0],
            [128,  3950.0],
            [256,  3980.0],
            [512,  4000.0],
            [1024, 4100.0],
            [2048, 4200.0],
            [4096, 4500.0],
        ],
        "use_pcie_link": True,
        # Host PCIe link capacity. Per user: the PCIe is 64 Gbps, so
        # the maximum bandwidth at the host is also 64 Gbps.
        "pcie_bandwidth_gbps": 64.0,
        "fpga_egress_fifo_bytes": 262144,
        "pcie_setup_ns": 0.0,
    },
    "arbiter": {
        "policy": "wrr",
        "wrr_weight_stateless": 2,
        "wrr_weight_stateful": 1,
    },
    "telemetry": {
        "w1": 0.3,
        "w2": 0.2,
        "w3": 0.5,
        "eps": 1.0e-09,
    },
    "predictor": {
        "predictor_type": "ewma",
        "W_window_epochs": 8,
        "ewma_alpha": 0.3,
    },
    "stateless_scheduler": {
        "scheduler_type": "static",
        "alpha_blend": 0.8,
        "tau_hot_s": 0.35,
        "tau_cold_s": 0.20,
        "fit_condition_tolerance": 0.1,
        "max_moves_per_epoch": 16,
    },
    "stateful_scheduler": {
        "scheduler_type": "proposed",
        "eta1": 0.6,
        "eta2": 0.4,
        "tau_hot_t": 0.30,
        "tau_cold_t": 0.15,
        "lambda_t": 0.01,
        "epsilon_t": 0.001,
        "handoff_latency_ewma_alpha": 0.2,
        "max_concurrent_handoffs": 2,
        "handoff_buffer_bytes": 1000000,
        "handoff_drain_mean_ns": 20000.0,
        "handoff_drain_std_ns": 10000.0,
        "handoff_migration_mean_ns": 60000.0,
        "handoff_migration_std_ns": 15000.0,
        "handoff_ack_mean_ns": 20000.0,
        "handoff_ack_std_ns": 5000.0,
    },
    "experiment": {
        "name": "REPLACE_ME",
        "rng_seed": 12648430,
        "results_dir": "results",
        "log_time_series": True,
        "log_per_bucket_trace": True,
        "make_plots": True,
        "enable_stateless": True,
        "enable_stateful": True,
    },
}


# Stateless matrix: vary stateless_scheduler; hold stateful static.
STATELESS_VARIANTS = [
    # Only the *_oneshot family is evaluated: greedy variants have
    # been dropped from the sweep (they make nearly identical
    # scheduling decisions as the corresponding *_oneshot variant on
    # this hardware model, so running both is redundant; the legend
    # label in plots simply drops the "_oneshot" subscript).
    "static",
    "qp_oneshot",
    "pred_oneshot",
    "pred_qp_oneshot",
]

# Stateful pair: (scheduler_type, config_name).
STATEFUL_VARIANTS = [
    ("static",   "static"),
    ("proposed", "proposed"),
]


# ----------------------------------------------------------------------
# Mixes.
#
# Each entry is (web_frac, cache_frac, hadoop_frac, total_gbps). Mix
# fractions sum to 1; per-class Layer-1 target mean rate is
# ``frac * total_gbps`` Gbps, with burstiness drawn from the HAL
# paper's per-class lognormal rate process clipped at hal_link_gbps.
#
# Single-class mixes use the HAL paper's published per-class average
# link-utilisation directly (Huang et al. ISCA'24 Fig. 8):
#
#     web    1.6  Gbps
#     cache  5.2  Gbps
#     hadoop 10.9 Gbps
#
# These are far below the 64 Gbps PCIe link, so the host is NOT
# bandwidth-saturated on average; the bursts from the heavy-tailed
# lognormal (sigma = 1.97 / 7.55 / 6.56 respectively, all clipped at
# 100 Gbps) are what drive transient overshoot and queue imbalance.
# This is the regime in which the scheduler under test can actually
# differentiate itself, rather than being masked by PCIe saturation.
# ----------------------------------------------------------------------
MIXES: dict[str, tuple[float, float, float, float]] = {
    "web_only":    (1.0, 0.0, 0.0, 1.6),
    "cache_only":  (0.0, 1.0, 0.0, 5.2),
    "hadoop_only": (0.0, 0.0, 1.0, 10.9),
}


def _deepcopy(d):
    return yaml.safe_load(yaml.safe_dump(d))


def _dump(cfg: dict, path: str) -> None:
    with open(path, "w") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)


def cleanup_stale_yaml(dir_path: str, allowed_stems: set[str]) -> None:
    """Remove ``*.yaml`` in ``dir_path`` whose stem is not in
    ``allowed_stems``.

    ``run_comparison`` globs every ``*.yaml`` in a suite directory.
    After ``STATELESS_VARIANTS`` (or stateful names) shrink, older
    files such as ``pred_greedy.yaml`` would otherwise keep being run
    and plotted.
    """
    if not os.path.isdir(dir_path):
        return
    want = {f"{s}.yaml" for s in allowed_stems}
    for fn in os.listdir(dir_path):
        if not fn.endswith(".yaml"):
            continue
        if fn not in want:
            os.remove(os.path.join(dir_path, fn))


def _apply_mix(cfg: dict,
               web: float, cache: float, hadoop: float,
               total_gbps: float) -> None:
    cfg["workload"]["hal_mix_web"]    = float(web)
    cfg["workload"]["hal_mix_cache"]  = float(cache)
    cfg["workload"]["hal_mix_hadoop"] = float(hadoop)
    cfg["workload"]["hal_total_gbps"] = float(total_gbps)


# Path (relative to repo root) to the trained TCN weights file. The
# data-gen / training scripts write to models/tcn_pred.pt; only the
# ``pred_oneshot`` and ``pred_qp_oneshot`` stateless variants actually
# consume this (other variants don't use the predictor output).
TCN_CHECKPOINT_PATH = "models/tcn_pred.pt"


def _apply_predictor_for_stateless(cfg: dict, sl_type: str) -> None:
    """Route the stateless pred / pred_qp variants through the trained
    TCN; leave all other variants on the EWMA baseline.

    ``static`` and ``qp_*`` variants don't consume predictor output, so
    forcing them onto TCN would pointlessly require the weights file
    to exist for their YAMLs.
    """
    uses_pred = ("pred" in sl_type)
    if uses_pred:
        cfg["predictor"]["predictor_type"] = "tcn"
        cfg["predictor"]["tcn_checkpoint"] = TCN_CHECKPOINT_PATH


def make_stateless(sl_type: str, mix: tuple[float, float, float, float]) -> dict:
    """Coexistence stateless matrix: both domains enabled, stateless
    scheduler varied, stateful held static. Same hal_composite mix
    is used for *both* domains (stateless sees the composite with
    proto=UDP, stateful with proto=TCP)."""
    cfg = _deepcopy(BASE)
    cfg["stateless_scheduler"]["scheduler_type"] = sl_type
    cfg["stateful_scheduler"]["scheduler_type"] = "static"
    cfg["stateful_scheduler"]["max_concurrent_handoffs"] = 0
    cfg["experiment"]["name"] = sl_type
    cfg["experiment"]["enable_stateless"] = True
    cfg["experiment"]["enable_stateful"] = True
    _apply_mix(cfg, *mix)
    _apply_predictor_for_stateless(cfg, sl_type)
    return cfg


def make_stateful(sf_type: str, name: str,
                  mix: tuple[float, float, float, float]) -> dict:
    cfg = _deepcopy(BASE)
    cfg["stateless_scheduler"]["scheduler_type"] = "pred_qp_greedy"
    cfg["stateful_scheduler"]["scheduler_type"] = sf_type
    if sf_type == "static":
        cfg["stateful_scheduler"]["max_concurrent_handoffs"] = 0
    cfg["experiment"]["name"] = name
    cfg["experiment"]["enable_stateless"] = True
    cfg["experiment"]["enable_stateful"] = True
    _apply_mix(cfg, *mix)
    return cfg


def make_stateless_only(sl_type: str,
                        mix: tuple[float, float, float, float]) -> dict:
    """Phase-1 stateless-only: only the stateless domain runs. The
    hal_composite workload is delivered to the stateless pipeline
    (proto=UDP). 8 cores = 8 RX queues."""
    cfg = _deepcopy(BASE)
    cfg["stateless_scheduler"]["scheduler_type"] = sl_type
    cfg["stateful_scheduler"]["scheduler_type"] = "static"
    cfg["stateful_scheduler"]["max_concurrent_handoffs"] = 0
    cfg["experiment"]["name"] = sl_type
    cfg["experiment"]["enable_stateless"] = True
    cfg["experiment"]["enable_stateful"] = False
    _apply_mix(cfg, *mix)
    _apply_predictor_for_stateless(cfg, sl_type)
    return cfg


def make_stateful_only(sf_type: str, name: str,
                       mix: tuple[float, float, float, float]) -> dict:
    """Phase-1 stateful-only: only the stateful domain runs. The
    hal_composite workload is delivered to the stateful pipeline
    (proto=TCP). 8 cores = 8 RX queues."""
    cfg = _deepcopy(BASE)
    cfg["stateless_scheduler"]["scheduler_type"] = "static"
    cfg["stateful_scheduler"]["scheduler_type"] = sf_type
    if sf_type == "static":
        cfg["stateful_scheduler"]["max_concurrent_handoffs"] = 0
    cfg["experiment"]["name"] = name
    cfg["experiment"]["enable_stateless"] = False
    cfg["experiment"]["enable_stateful"] = True
    _apply_mix(cfg, *mix)
    return cfg


def main() -> None:
    total_written = 0
    # Per-mix phase-1 (*_only) suites.
    for mix_name, mix in MIXES.items():
        mix_root = os.path.join(ROOT, "configs", f"hal_{mix_name}")
        sl_only_dir = os.path.join(mix_root, "stateless_only")
        sf_only_dir = os.path.join(mix_root, "stateful_only")
        os.makedirs(sl_only_dir, exist_ok=True)
        os.makedirs(sf_only_dir, exist_ok=True)
        cleanup_stale_yaml(sl_only_dir, set(STATELESS_VARIANTS))
        cleanup_stale_yaml(sf_only_dir,
                           {name for _, name in STATEFUL_VARIANTS})

        for v in STATELESS_VARIANTS:
            _dump(make_stateless_only(v, mix),
                  os.path.join(sl_only_dir, f"{v}.yaml"))
            total_written += 1
        for sf_type, name in STATEFUL_VARIANTS:
            _dump(make_stateful_only(sf_type, name, mix),
                  os.path.join(sf_only_dir, f"{name}.yaml"))
            total_written += 1

    print(f"wrote {total_written} per-mix configs under "
          f"configs/hal_<mix>/{{stateless_only,stateful_only}}/")
    print(f"mixes: {sorted(MIXES.keys())}")


if __name__ == "__main__":
    main()
