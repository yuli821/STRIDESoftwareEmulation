#!/usr/bin/env python3
"""Generate diagnostic SYNTHETIC configs:
``source=synthetic_sustained`` + the same TEST host/topology setup
(4 RX queues, 4 cores, 1000 ns t_app, skewed RSS, 128 Gbps PCIe).

This emits configs under:

    configs/synthetic/stateless_only/*.yaml
    configs/synthetic/stateful_only/*.yaml

Results from runs using these configs are written under
``results/synthetic/comparison_<suite>/`` by
``scripts/run_comparison.py --synthetic ...``.

Workload design
---------------

The synthetic workload is a DIAGNOSTIC microbenchmark, not a published
trace. It is constructed so the "ideal" scheduler output is known
a-priori: under a uniform RSS mapping, per-queue load is identical by
construction (N identical CBR flows); under the skewed RSS mapping
(50 / 30 / 10 / 10 % of the indirection table) the static per-queue
load is deterministically mis-weighted from t=0 and a correctly
working adaptive scheduler should drive byte-imbalance to ~0 and
Jain's fairness to ~1 within its first few epochs.

Parameters
----------

* ``synth_base_gbps = 40 Gbps`` -- below the 4-core * 12 Gbps/queue =
  48 Gbps aggregate host drain, but above the 12 Gbps/queue drain for
  the hot queue under the 50 %-skewed RSS, so static creates
  sustained queue pressure on Q0 from t=0.
* ``synth_burst_gbps = 30 Gbps`` for 1 ms every 5 ms -- aggregate
  peak is 70 Gbps (still well below the 128 Gbps PCIe). Tests
  scheduler responsiveness to transient overload.
* ``synth_n_flows = 64`` for sustained, ``synth_burst_n_flows = 32``
  for each burst. High flow concurrency per queue (16 flows/queue
  under uniform mapping) so Jain's / Gini have enough samples to be
  stable.
"""
from __future__ import annotations

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)

from gen_configs import (  # type: ignore  noqa: E402
    BASE, STATELESS_VARIANTS, STATEFUL_VARIANTS,
    _deepcopy, _dump, cleanup_stale_yaml,
)


def _apply_synthetic_overrides(cfg: dict, *, domain: str) -> None:
    """Overlay the synthetic microbenchmark setup onto a BASE config.

    Matches the TEST host/topology setup (4 queues = 4 cores,
    realistic per-packet t_app, ring 512, skewed RSS, 128 Gbps PCIe)
    so the host-side physics is identical to the realistic TEST runs
    and only the packet stream differs.

    Offered-load design: avg aggregate is set to ~70-75 % of each
    domain's host aggregate drain so queues stay under sustained
    pressure but PCIe (128 Gbps) is never the bottleneck. Both
    components (elephants running constantly + mice arriving as a
    Poisson process) produce smoothly varying aggregate-byte
    telemetry (no periodic step).

    Mice are intentionally *fewer but longer-lived* than in the
    earlier 100 k/s draft: the driver of per-epoch load stability is
    the number of *concurrent* mice at any instant, not the total
    arrival count. Longer per-mouse lifetimes make many mice overlap
    in time, so the per-queue byte count per 30 us epoch has small
    relative variance even with modest arrival rates. This is also
    more representative of a realistic datacenter (hundreds to low
    thousands of short flows per host per second, Roy et al.
    SIGCOMM'15; Kandula et al. IMC'09).

    * ``stateless``: t_app 1000 ns -> 12 Gbps/queue -> 48 Gbps
      aggregate drain. Elephants: 32 flows @ 30 Gbps aggregate
      (~940 Mbps each). Mice: 20 k arrivals/s, E[flow size] = 20 KB
      (lognormal, sigma 1), per-flow rate 0.2 Gbps -> per-mouse
      lifetime ~800 us, concurrent mice ~16, mean mice aggregate
      ~3.2 Gbps. Total avg ~33 Gbps (69 % drain). Total mice per
      60 ms horizon ~1 200.
    * ``stateful``: t_app 3000 ns + 80 ns conn lookup -> ~3.9 Gbps/
      queue -> ~15.6 Gbps drain. Elephants: 32 flows @ 10 Gbps
      aggregate (~313 Mbps each). Mice: 8 k arrivals/s,
      E[flow size] 24 KB, per-flow 0.2 Gbps -> per-mouse lifetime
      ~960 us, concurrent mice ~8, mean mice aggregate ~1.5 Gbps.
      Total avg ~11.5 Gbps (74 % drain). Total mice per 500 ms
      horizon ~4 000.
    """
    # Topology: 4 queues, 4 cores, one-to-one, skewed RSS.
    topo = cfg["topology"]
    topo["num_cores_stateless"] = 4
    topo["num_cores_stateful"] = 4
    topo["num_stateless_queues"] = 4
    topo["num_stateful_queues"] = 4
    topo["initial_rss_stateless"] = "skewed"
    topo["initial_rss_stateful"] = "skewed"
    topo["descriptor_ring_depth_stateless"] = 512
    topo["descriptor_ring_depth_stateful"] = 512

    host = cfg["host"]
    host["stateless_t_app_mean_ns"] = 1000.0
    host["stateful_t_app_mean_ns"]  = 3000.0
    host["stateless_t_app_jitter_ns"] = 150.0
    host["stateful_t_app_jitter_ns"]  = 500.0
    host["pcie_bandwidth_gbps"] = 128.0

    w = cfg["workload"]
    w["source"] = "synthetic_sustained"
    w["synth_mtu_bytes"] = 1500
    w["synth_elephant_n_flows"] = 32
    w["synth_mice_size_sigma"] = 1.0
    if domain == "stateless":
        w["synth_elephant_total_gbps"] = 30.0
        w["synth_mice_arrival_rate_per_sec"] = 20_000.0
        w["synth_mice_mean_flow_bytes"] = 20_000.0
        w["synth_mice_per_flow_gbps"] = 0.2
    elif domain == "stateful":
        w["synth_elephant_total_gbps"] = 10.0
        w["synth_mice_arrival_rate_per_sec"] = 8_000.0
        w["synth_mice_mean_flow_bytes"] = 24_000.0
        w["synth_mice_per_flow_gbps"] = 0.2
    else:
        raise ValueError(f"unknown domain {domain!r}")


def _make(fn: str, *, sl_type: str | None = None,
          sf_type: str | None = None,
          name: str | None = None) -> dict:
    """Build a config by reusing the BASE and the stateless/stateful
    make_* helpers, then overlay the synthetic overrides. ``mix`` is
    unused by synthetic_sustained; we pass a zero-sum placeholder to
    keep the helper signature happy."""
    from gen_configs import (  # type: ignore  noqa: E402
        make_stateless_only, make_stateful_only,
    )
    placeholder_mix = (1.0, 0.0, 0.0, 0.0)  # irrelevant under synthetic
    if fn == "stateless_only":
        cfg = make_stateless_only(sl_type, placeholder_mix)
        _apply_synthetic_overrides(cfg, domain="stateless")
    elif fn == "stateful_only":
        cfg = make_stateful_only(sf_type, name, placeholder_mix)
        _apply_synthetic_overrides(cfg, domain="stateful")
    else:
        raise ValueError(fn)
    return cfg


def main() -> None:
    total = 0
    root_dir = os.path.join(ROOT, "configs", "synthetic")
    sl_only_dir = os.path.join(root_dir, "stateless_only")
    sf_only_dir = os.path.join(root_dir, "stateful_only")
    os.makedirs(sl_only_dir, exist_ok=True)
    os.makedirs(sf_only_dir, exist_ok=True)
    cleanup_stale_yaml(sl_only_dir, set(STATELESS_VARIANTS))
    cleanup_stale_yaml(sf_only_dir, {name for _, name in STATEFUL_VARIANTS})

    for v in STATELESS_VARIANTS:
        _dump(_make("stateless_only", sl_type=v),
              os.path.join(sl_only_dir, f"{v}.yaml"))
        total += 1
    for sf_type, name in STATEFUL_VARIANTS:
        _dump(_make("stateful_only", sf_type=sf_type, name=name),
              os.path.join(sf_only_dir, f"{name}.yaml"))
        total += 1

    print(f"wrote {total} SYNTHETIC configs under "
          f"configs/synthetic/{{stateless_only,stateful_only}}/")


if __name__ == "__main__":
    main()
