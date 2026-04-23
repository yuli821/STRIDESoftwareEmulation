#!/usr/bin/env python3
"""Regenerate all per-experiment YAML files under configs/stateless and
configs/stateful from a shared base template.

Run this whenever you change the shared workload/host/topology knobs so
every experiment stays in sync.
"""
from __future__ import annotations

import os
import sys
import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)


# ----------------------------------------------------------------------
# Shared baseline. Every stateless / stateful experiment uses this and
# then overrides only the scheduler-specific knobs + experiment.name.
# ----------------------------------------------------------------------
BASE = {
    "time": {
        "clk_period_ns": 4.0,
        "delta_bin_ns": 10000.0,
        "H_bins_per_epoch": 10,           # legacy fallback
        "num_epochs": 500,                # outer (stateless) epochs
        "stateless_epoch_bins": 5,        # 50 us per stateless epoch
        "stateful_epoch_bins": 100,       # 1 ms per stateful epoch
    },
    "topology": {
        "num_stateless_queues": 8,
        "num_stateful_queues": 8,
        "num_stateless_buckets": 128,
        "num_stateful_buckets": 128,
        "num_cores_stateless": 8,
        "num_cores_stateful": 8,
        "queue_to_core_map_stateless": "one_to_one",
        "queue_to_core_map_stateful": "one_to_one",
        # Random RSS so hot buckets can collide on the same queue.
        "initial_rss_stateless": "random",
        "initial_rss_stateful": "random",
        "descriptor_ring_depth_stateless": 2048,
        "descriptor_ring_depth_stateful": 2048,
    },
    "workload": {
        # CDF-driven per-flow generation based on digitized IMC'17 and
        # Roy '15 CDFs (flow size / packet size / IAT / ON-OFF).
        "source": "imc17_cdf",
        # Per-class flow counts chosen so the *target* per-flow rate
        # is within the natural range of IMC'17 CDFs (cache: a few
        # Mbps, web: ~10 Mbps, hadoop: ~Gbps). This keeps the
        # per-class time_scale near 1 and avoids pathological cycle
        # counts in the CDF-driven generator.
        #
        # Stateless aggregate ~56 Gbps, close to the 8-core capacity
        # (~64 Gbps at T_app=400 ns for the IMIX mix below), so random
        # 5-tuple + random RSS collisions reliably push 1-2 queues
        # over their core budget -> credit drops + P_q rises.
        "trace_mix_stateless": [
            {"kind": "cache",  "n_flows": 500, "gbps": 18.0},
            {"kind": "web",    "n_flows": 400, "gbps": 13.0},
            {"kind": "hadoop", "n_flows": 28,  "gbps": 25.0},
        ],
        # Stateful aggregate ~21 Gbps, close to the ~24 Gbps stateful
        # 8-core cap at T_app=2 us -> skew causes hotspots.
        "trace_mix_stateful": [
            {"kind": "web",    "n_flows": 240, "gbps": 11.0},
            {"kind": "hadoop", "n_flows": 18,  "gbps": 10.0},
        ],
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
        "pcie_bandwidth_gbps": 128.0,
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
    # Overridden per experiment below.
    "stateless_scheduler": {
        "scheduler_type": "static",
        "alpha_blend": 0.8,
        # Hot/cold thresholds calibrated to the steady-state P_q under
        # this workload (~0.4 mean_P_max). tau_hot ~= 0.35 ensures the
        # scheduler fires regularly without thrashing.
        "tau_hot_s": 0.35,
        "tau_cold_s": 0.20,
        "fit_condition_tolerance": 0.1,
        "max_moves_per_epoch": 16,
    },
    "stateful_scheduler": {
        "scheduler_type": "proposed",
        "eta1": 0.6,
        "eta2": 0.4,
        # Stateful steady-state P_q peaks around 0.35; using 0.30 hot
        # keeps the stateful scheduler from being a no-op.
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
        # Both domains enabled by default (coexistence / phase-2 style).
        # Isolated-domain configs flip these per suite.
        "enable_stateless": True,
        "enable_stateful": True,
    },
}


# ----------------------------------------------------------------------
# Stateless matrix: vary stateless_scheduler.scheduler_type; keep
# stateful fixed to ``static`` so the stateless effect is isolated.
# ----------------------------------------------------------------------
STATELESS_VARIANTS = [
    "static",
    "qp_oneshot",
    "qp_greedy",
    "pred_oneshot",
    "pred_greedy",
    "pred_qp_oneshot",
    "pred_qp_greedy",
]

# ----------------------------------------------------------------------
# Stateful pair: vary stateful_scheduler.scheduler_type; hold stateless
# on the strong variant so stateful effects aren't masked by unbalanced
# stateless queues.
# ----------------------------------------------------------------------
STATEFUL_VARIANTS = [
    ("static",   "static"),     # stateful_scheduler.scheduler_type
    ("proposed", "proposed"),
]


def _deepcopy(d):
    return yaml.safe_load(yaml.safe_dump(d))


def _dump(cfg: dict, path: str) -> None:
    with open(path, "w") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)


def make_stateless(sl_type: str) -> dict:
    """Coexistence stateless matrix: both domains enabled, stateless
    scheduler varied, stateful held static. Used for phase-2 realism
    runs where we want to confirm isolated-phase trends survive
    cross-domain interference on the shared PCIe link."""
    cfg = _deepcopy(BASE)
    cfg["stateless_scheduler"]["scheduler_type"] = sl_type
    cfg["stateful_scheduler"]["scheduler_type"] = "static"
    cfg["stateful_scheduler"]["max_concurrent_handoffs"] = 0
    cfg["experiment"]["name"] = sl_type
    cfg["experiment"]["enable_stateless"] = True
    cfg["experiment"]["enable_stateful"] = True
    return cfg


def make_stateful(sf_type: str, name: str) -> dict:
    """Coexistence stateful pair: both domains enabled, stateless held
    on the canonical strong variant, stateful varied."""
    cfg = _deepcopy(BASE)
    cfg["stateless_scheduler"]["scheduler_type"] = "pred_qp_greedy"
    cfg["stateful_scheduler"]["scheduler_type"] = sf_type
    if sf_type == "static":
        cfg["stateful_scheduler"]["max_concurrent_handoffs"] = 0
    cfg["experiment"]["name"] = name
    cfg["experiment"]["enable_stateless"] = True
    cfg["experiment"]["enable_stateful"] = True
    return cfg


def make_stateless_only(sl_type: str) -> dict:
    """Phase-1 stateless-only: stateful domain fully disabled so the
    algorithm is evaluated in isolation (no cross-domain PCIe
    interference, no stateful scheduling noise)."""
    cfg = _deepcopy(BASE)
    cfg["stateless_scheduler"]["scheduler_type"] = sl_type
    cfg["stateful_scheduler"]["scheduler_type"] = "static"
    cfg["stateful_scheduler"]["max_concurrent_handoffs"] = 0
    # Empty the stateful workload so even if some future code path
    # peeks at trace_mix_stateful it finds nothing.
    cfg["workload"]["trace_mix_stateful"] = []
    cfg["experiment"]["name"] = sl_type
    cfg["experiment"]["enable_stateless"] = True
    cfg["experiment"]["enable_stateful"] = False
    return cfg


def make_stateful_only(sf_type: str, name: str) -> dict:
    """Phase-1 stateful-only: stateless domain fully disabled."""
    cfg = _deepcopy(BASE)
    cfg["stateless_scheduler"]["scheduler_type"] = "static"
    cfg["stateful_scheduler"]["scheduler_type"] = sf_type
    if sf_type == "static":
        cfg["stateful_scheduler"]["max_concurrent_handoffs"] = 0
    cfg["workload"]["trace_mix_stateless"] = []
    cfg["experiment"]["name"] = name
    cfg["experiment"]["enable_stateless"] = False
    cfg["experiment"]["enable_stateful"] = True
    return cfg


def main() -> None:
    sl_dir = os.path.join(ROOT, "configs", "stateless")
    sf_dir = os.path.join(ROOT, "configs", "stateful")
    sl_only_dir = os.path.join(ROOT, "configs", "stateless_only")
    sf_only_dir = os.path.join(ROOT, "configs", "stateful_only")
    for d in (sl_dir, sf_dir, sl_only_dir, sf_only_dir):
        os.makedirs(d, exist_ok=True)

    for v in STATELESS_VARIANTS:
        _dump(make_stateless(v), os.path.join(sl_dir, f"{v}.yaml"))
        _dump(make_stateless_only(v),
              os.path.join(sl_only_dir, f"{v}.yaml"))
    for sf_type, name in STATEFUL_VARIANTS:
        _dump(make_stateful(sf_type, name),
              os.path.join(sf_dir, f"{name}.yaml"))
        _dump(make_stateful_only(sf_type, name),
              os.path.join(sf_only_dir, f"{name}.yaml"))

    print(f"wrote {len(STATELESS_VARIANTS)} coexistence stateless configs "
          f"under {sl_dir}")
    print(f"wrote {len(STATEFUL_VARIANTS)} coexistence stateful configs "
          f"under {sf_dir}")
    print(f"wrote {len(STATELESS_VARIANTS)} isolated stateless-only configs "
          f"under {sl_only_dir}")
    print(f"wrote {len(STATEFUL_VARIANTS)} isolated stateful-only configs "
          f"under {sf_only_dir}")


if __name__ == "__main__":
    main()
