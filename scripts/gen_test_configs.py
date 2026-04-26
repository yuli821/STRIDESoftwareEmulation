#!/usr/bin/env python3
"""Generate TEST configs: HAL-native workload + realistic SNIC
host-side parameters so the scheduler sees real queue pressure.

This emits configs under:

    configs/test/hal_<mix>/stateless_only/*.yaml
    configs/test/hal_<mix>/stateful_only/*.yaml

Results from runs using these configs are written under
``results/test/hal_<mix>/comparison_<suite>/`` by
``scripts/run_comparison.py --test ...`` (the ``--test`` flag simply
rebases input/output paths under ``configs/test`` / ``results/test``).

Deltas from the main BASE config (see scripts/gen_configs.py):

* ``num_cores_stateless``, ``num_cores_stateful``,
  ``num_stateless_queues``, and ``num_stateful_queues`` all lowered to 4
  (4 RX queues, 4 cores, one-to-one). This models a 4-core SmartNIC host
  with a 4-queue RSS indirection table.
* ``stateless_t_app_mean_ns`` and ``stateful_t_app_mean_ns`` raised to
  1000 ns (realistic SmartNIC per-packet processing cost), which caps
  per-queue drain rate at ~12 Gbps for 1500 B MTU. Under HAL's bursty
  lognormal rate process, bursts routinely exceed this per-queue
  drain rate, filling the ring.
* ``descriptor_ring_depth_stateless`` and ``_stateful`` lowered to 512
  so the ring actually saturates inside a 1-ms burst.
* ``initial_rss_stateless`` and ``initial_rss_stateful`` set to
  ``"skewed"``: two hot queues hold 50% and 30% of the RSS indirection
  table entries; the remaining two queues split 20%. This models a
  pathological RSS hash distribution (Woo et al., RSS++ NSDI'19;
  Liu et al., SIGCOMM'22) and is the regime where an adaptive
  scheduler should measurably outperform static bucket-to-queue
  mapping.
* ``pcie_bandwidth_gbps`` raised to 128 Gbps, above the aggregate
  offered load, so drops happen at the host ring (the thing the
  scheduler controls) rather than at the PCIe FIFO (which is upstream
  of the scheduler and therefore masks differentiation).
* Mixes start with single-class (web_only / cache_only / hadoop_only)
  at HAL-native per-class averages (1.6 / 5.2 / 10.9 Gbps, Huang et
  al. ISCA'24 Fig. 8) multiplied by ``TEST_N_TENANTS``.
* Per-domain epoch counts inherited from BASE: stateless 2000 epochs
  (60 ms horizon) and stateful 500 epochs (500 ms horizon). The
  stateful horizon is capped because the simulator's per-packet
  Python loop in ``sim._process_bin_merged`` is the wall-clock
  bottleneck and a 2 s stateful horizon would generate ~5.4M packets
  per variant (~10-15 min per variant).
"""
from __future__ import annotations

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)

from gen_configs import (  # type: ignore  noqa: E402
    BASE, STATELESS_VARIANTS, STATEFUL_VARIANTS,
    _deepcopy, _dump, _apply_mix, cleanup_stale_yaml,
)


TEST_N_TENANTS = 3  # Multi-tenant HAL aggregation (each tenant = native HAL).


# Test-specific overrides.
def _apply_test_overrides(cfg: dict) -> None:
    topo = cfg["topology"]
    topo["num_cores_stateless"] = 4
    topo["num_cores_stateful"] = 4
    # Queue count must match core count. Leaving num_queues > num_cores
    # structurally caps Jain's fairness index at num_cores / num_queues
    # because unmapped queues stay idle while loaded queues pile up,
    # irrespective of scheduler quality.
    topo["num_stateless_queues"] = 4
    topo["num_stateful_queues"] = 4
    # Skewed RSS indirection table: two "hot" queues absorb 50% and 30%
    # of buckets respectively; remaining two queues share 20%. This
    # creates a structural load imbalance that the scheduler can
    # measurably correct by migrating buckets off hot queues, as in
    # RSS++ NSDI'19.
    topo["initial_rss_stateless"] = "skewed"
    topo["initial_rss_stateful"] = "skewed"
    # Ring size reduced so bursts saturate the ring before the scheduler
    # can react via migration; the scheduler is now forced to decide
    # between leaving a bucket on a hot queue (tail latency hit) vs
    # moving it (handoff cost).
    topo["descriptor_ring_depth_stateless"] = 512
    topo["descriptor_ring_depth_stateful"] = 512

    host = cfg["host"]
    # 1000 ns per packet -> per-queue drain rate for MTU = 12 Gbps.
    # With 4 queues -> aggregate host drain rate ~48 Gbps.
    host["stateless_t_app_mean_ns"] = 1000.0
    host["stateful_t_app_mean_ns"]  = 3000.0
    host["stateless_t_app_jitter_ns"] = 150.0
    host["stateful_t_app_jitter_ns"]  = 500.0
    # Raise PCIe well above the aggregate offered load so drops happen
    # at the host descriptor ring (which the scheduler controls) rather
    # than at the PCIe FIFO (which is upstream of the scheduler and
    # therefore masks any differentiation).
    host["pcie_bandwidth_gbps"] = 128.0

    # Multi-tenant HAL aggregation: Layer-1 rate per class = sum of N
    # independent lognormal timelines, each targeting the HAL native
    # per-class mean. This keeps HAL paper fidelity per tenant while
    # driving aggregate offered load up toward the host's 48 Gbps
    # drain capacity.
    cfg["workload"]["hal_n_tenants"] = int(TEST_N_TENANTS)


# TEST mixes.
#
# Single-class TEST mixes use HAL paper native per-class averages
# (web 1.6, cache 5.2, hadoop 10.9 Gbps) multiplied by N_TENANTS so
# aggregate mean = N_TENANTS * native. Composite TEST mixes add
# classes together at native rates (again scaled by N_TENANTS).
_NATIVE: dict[str, float] = {"web": 1.6, "cache": 5.2, "hadoop": 10.9}
_N = TEST_N_TENANTS


def _m(web: float, cache: float, hadoop: float
       ) -> tuple[float, float, float, float]:
    """Build a mix tuple (web_frac, cache_frac, hadoop_frac, total_gbps)
    at HAL native rates scaled by ``TEST_N_TENANTS``."""
    per_class = {
        "web": web * _NATIVE["web"],
        "cache": cache * _NATIVE["cache"],
        "hadoop": hadoop * _NATIVE["hadoop"],
    }
    total_native = sum(per_class.values())
    if total_native <= 0:
        return (0.0, 0.0, 0.0, 0.0)
    total_gbps = _N * total_native
    # Fractions are proportional to per_class native shares, since
    # hal_total_gbps * frac reproduces per_class * N in the generator.
    fw = per_class["web"]    / total_native
    fc = per_class["cache"]  / total_native
    fh = per_class["hadoop"] / total_native
    return (fw, fc, fh, total_gbps)


TEST_MIXES: dict[str, tuple[float, float, float, float]] = {
    "web_only":    _m(1, 0, 0),    # N*1.6  = 4.8  Gbps
    "cache_only":  _m(0, 1, 0),    # N*5.2  = 15.6 Gbps
    "hadoop_only": _m(0, 0, 1),    # N*10.9 = 32.7 Gbps
    "hadoop_plus_cache": _m(0, 1, 1),  # N*(10.9+5.2) = 48.3 Gbps
    "hadoop_cache_web":  _m(1, 1, 1),  # N*(10.9+5.2+1.6) = 53.1 Gbps
}


def _make(fn, *, mix, sl_type=None, sf_type=None, name=None) -> dict:
    """Thin wrapper: reuse the BASE config and stateless/stateful
    make_* helpers, then overlay the test overrides."""
    from gen_configs import (  # type: ignore  noqa: E402
        make_stateless_only, make_stateful_only,
    )
    if fn == "stateless_only":
        cfg = make_stateless_only(sl_type, mix)
    elif fn == "stateful_only":
        cfg = make_stateful_only(sf_type, name, mix)
    else:
        raise ValueError(fn)
    _apply_test_overrides(cfg)
    return cfg


def main() -> None:
    total_written = 0
    for mix_name, mix in TEST_MIXES.items():
        mix_root = os.path.join(ROOT, "configs", "test", f"hal_{mix_name}")
        sl_only_dir = os.path.join(mix_root, "stateless_only")
        sf_only_dir = os.path.join(mix_root, "stateful_only")
        os.makedirs(sl_only_dir, exist_ok=True)
        os.makedirs(sf_only_dir, exist_ok=True)
        cleanup_stale_yaml(sl_only_dir, set(STATELESS_VARIANTS))
        cleanup_stale_yaml(sf_only_dir,
                           {name for _, name in STATEFUL_VARIANTS})

        for v in STATELESS_VARIANTS:
            _dump(_make("stateless_only", mix=mix, sl_type=v),
                  os.path.join(sl_only_dir, f"{v}.yaml"))
            total_written += 1
        for sf_type, name in STATEFUL_VARIANTS:
            _dump(_make("stateful_only", mix=mix, sf_type=sf_type, name=name),
                  os.path.join(sf_only_dir, f"{name}.yaml"))
            total_written += 1

    print(f"wrote {total_written} TEST configs under "
          f"configs/test/hal_<mix>/{{stateless_only,stateful_only}}/")
    print(f"TEST mixes: {sorted(TEST_MIXES.keys())}")


if __name__ == "__main__":
    main()
