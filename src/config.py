"""Central configuration for the software emulator.

All user-tunable parameters live here as dataclasses. A YAML file selects any
subset to override defaults (see configs/*.yaml).
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional, Union

import yaml


# ---------------------------------------------------------------------------
# Time model
# ---------------------------------------------------------------------------
@dataclass
class TimeConfig:
    clk_period_ns: float = 4.0              # FPGA clock (250 MHz)
    delta_bin_ns: float = 10_000.0          # telemetry bin delta
    H_bins_per_epoch: int = 10              # default per-domain epoch length
    num_epochs: int = 1000                  # shared-default epoch count

    # Per-domain epoch count. If 0, fall back to ``num_epochs``. The
    # stateless domain uses a short epoch (30 us) so the scheduler
    # reacts quickly, while the stateful domain uses a much longer
    # epoch (1 ms) to avoid handoff thrashing. A shared ``num_epochs``
    # therefore produces a stateful simulation horizon that is 30-50x
    # longer than the stateless one, even though the stateful
    # pipeline also does strictly more per-packet work (PCIe
    # try_transmit + conn-track lookup + handoff redirection). To keep
    # stateful wall-clock tractable without sacrificing stateless
    # statistical richness, each domain can set its own epoch count.
    num_epochs_stateless: int = 0
    num_epochs_stateful: int = 0

    # Per-domain epoch bins. If 0, fall back to H_bins_per_epoch.
    # Stateless schedulers typically want short epochs to react quickly
    # (e.g. 5-10 bins = 50-100 us). Stateful schedulers have much higher
    # per-handoff cost, so they benefit from longer epochs (e.g. 50-200
    # bins = 500 us - 2 ms) to avoid thrashing.
    stateless_epoch_bins: int = 0
    stateful_epoch_bins: int = 0

    @property
    def epoch_ns(self) -> float:
        return self.delta_bin_ns * self.H_bins_per_epoch

    def stateless_epoch_ns(self) -> float:
        H = self.stateless_epoch_bins or self.H_bins_per_epoch
        return self.delta_bin_ns * H

    def stateful_epoch_ns(self) -> float:
        H = self.stateful_epoch_bins or self.H_bins_per_epoch
        return self.delta_bin_ns * H

    def num_epochs_for(self, domain: str) -> int:
        """Resolve the per-domain epoch count, falling back to
        ``num_epochs`` when the per-domain override is 0."""
        if domain == "stateless":
            return int(self.num_epochs_stateless or self.num_epochs)
        if domain == "stateful":
            return int(self.num_epochs_stateful or self.num_epochs)
        raise ValueError(f"unknown domain {domain!r}")


# ---------------------------------------------------------------------------
# Topology (queues, buckets, cores, descriptor ring depth)
# ---------------------------------------------------------------------------
@dataclass
class TopologyConfig:
    num_stateless_queues: int = 8
    num_stateful_queues: int = 8
    num_stateless_buckets: int = 128
    num_stateful_buckets: int = 128

    num_cores_stateless: int = 8
    num_cores_stateful: int = 8

    queue_to_core_map_stateless: str = "one_to_one"  # one_to_one | round_robin | block
    queue_to_core_map_stateful: str = "one_to_one"

    initial_rss_stateless: str = "modulo"            # modulo | random | skewed
    initial_rss_stateful: str = "modulo"

    descriptor_ring_depth_stateless: int = 2048       # D_q for stateless queues
    descriptor_ring_depth_stateful: int = 2048        # D_q for stateful queues


# ---------------------------------------------------------------------------
# Workload (trace-based)
#
# Three trace sources selectable via `source`:
#   * trace_mix      : built from a list of {kind, n_flows, gbps} specs where
#                      kind in {web, cache, hadoop, synthetic_rates}.
#   * trace_csv      : load per-packet traces from CSV(s).
#   * synthetic_rates: build one TraceSet directly from num_flows +
#                      target_gbps + flow_rate_distribution + burstiness.
#
# The synthetic_rates knobs (num_flows_*, zipf_s, burstiness_model, ...) are
# shared with the `synthetic_rates` kind inside `trace_mix`.
# ---------------------------------------------------------------------------
@dataclass
class WorkloadConfig:
    # hal_composite: PRIMARY -- two-layer bursty composite workload,
    #                Huang et al. (HAL), ISCA 2024, Fig. 8. Layer 1 is a
    #                per-class log-normal rate process (web mu/sigma =
    #                -1.37/1.97, cache -9/7.55, hadoop -4.18/6.56) clipped
    #                to link_gbps and re-scaled to a configurable
    #                aggregate target. Layer 2 is a persistent flow
    #                table: each new flow is sampled with a traffic
    #                class, flow size, sending rate, duration, 5-tuple,
    #                and RSS bucket, and stays active across epochs
    #                until its byte budget or sampled duration is
    #                exhausted. Per-class flow-size distributions are
    #                chosen to match Roy et al. SIGCOMM'15 Fig. 9 (Meta
    #                Web/Cache/Hadoop clusters) and the user
    #                descriptions: web = small request/response mixture,
    #                cache = tens-of-KB + MB-scale mixture, hadoop =
    #                mice/elephant (70% < 10KB, median < 1KB, small
    #                multi-MB tail). See ``src/hal_workload.py``.
    # poisson_flow : legacy stateless generator. Poisson flow arrivals
    #                + per-flow size from a published CDF (pFabric
    #                SIGCOMM'13 / PIAS NSDI'15 / Homa SIGCOMM'18 / NDP
    #                SIGCOMM'17 methodology). Kept for regression.
    # rpc          : legacy stateful generator. Long-lived connections
    #                carrying RPCs with exponential think-time (Homa
    #                SIGCOMM'18 workload model). Kept for regression.
    # imc17_cdf    : deprecated -- per-flow ON/OFF model digitized from
    #                port-level IMC'17 statistics; kept for backward
    #                comparison only. IMC'17 characterizes per-port,
    #                not per-flow.
    # trace_mix    : legacy hand-tuned generators for web/cache/hadoop.
    # trace_csv    : load packet traces from CSV.
    # synthetic_rates: zipf / heavy-hitter / uniform rate dist + cbr/onoff.
    # synthetic_sustained: diagnostic CBR workload with optional
    #                periodic burst overlay. Purpose: isolate scheduler
    #                behaviour from workload variance. See
    #                ``synth_*`` knobs below.
    source: str = "hal_composite"  # hal_composite | poisson_flow | rpc |
                                   # imc17_cdf | trace_mix | trace_csv |
                                   # synthetic_rates | synthetic_sustained

    # ----- trace_csv path -----
    trace_file_stateless: Optional[str] = None
    trace_file_stateful: Optional[str] = None

    # ----- trace_mix path -----
    # Each entry: {kind, n_flows, gbps, [per_flow_rate_gbps], [mtu_bytes], ...}
    #
    # For ``source: poisson_flow`` / ``source: rpc`` the allowed kinds are
    # the four published flow-size CDFs:
    #   - ``web_search``     (DCTCP, Alizadeh et al. SIGCOMM'10)
    #   - ``data_mining``    (VL2,   Greenberg et al. SIGCOMM'09)
    #   - ``cache_follower`` (FB,    Roy et al. SIGCOMM'15)
    #   - ``hadoop``         (FB,    Roy et al. SIGCOMM'15)
    # For the rpc source, ``n_flows`` is the number of long-lived
    # connections. For poisson_flow, ``n_flows`` is *ignored* (the flow
    # count is the stochastic outcome of the Poisson arrival process).
    #
    # For ``source: imc17_cdf`` / ``source: trace_mix`` the legacy kinds
    # ``web`` / ``cache`` / ``hadoop`` are used instead.
    trace_mix_stateless: List[dict] = field(default_factory=lambda: [
        {"kind": "web_search",  "gbps": 20.0},
        {"kind": "data_mining", "gbps": 20.0},
    ])
    trace_mix_stateful: List[dict] = field(default_factory=lambda: [
        {"kind": "cache_follower", "n_flows": 120, "gbps": 8.0},
        {"kind": "hadoop",         "n_flows": 16,  "gbps": 12.0},
    ])

    # ----- synthetic_rates path (and synthetic_rates kind inside trace_mix) -----
    num_flows_stateless: int = 256
    num_flows_stateful: int = 64
    stateless_target_gbps: float = 100.0
    stateful_target_gbps: float = 50.0

    flow_rate_distribution: str = "zipf"        # uniform | zipf | heavy_hitter
    zipf_s: float = 1.2
    heavy_hitter_fraction: float = 0.1
    heavy_hitter_multiplier: float = 10.0

    packet_size_distribution: str = "imix"      # fixed | imix
    fixed_packet_bytes: int = 1024
    imix_profile: List[List[Union[int, float]]] = field(
        default_factory=lambda: [[64, 0.58], [594, 0.33], [1518, 0.09]]
    )

    burstiness_model: str = "onoff"             # cbr | onoff
    onoff_mean_on_ns: float = 200_000.0
    onoff_mean_off_ns: float = 100_000.0

    # ----- shared -----
    max_link_gbps: float = 200.0
    hash_mode: str = "toeplitz"
    pattern_shift_period_epochs: int = 300      # 0 = no drift

    # ----- poisson_flow / rpc shared knobs -----
    # Per-flow rate cap (Gbps) for within-flow back-to-back emission.
    # Conventional datacenter simulators (pFabric SIGCOMM'13, Homa
    # SIGCOMM'18) model each flow saturating its share of the NIC,
    # typically line-rate. 10 Gbps is a good default so heavy-tail
    # flows complete within typical simulation horizons; set smaller
    # per-spec if you want to model rate-limited flows explicitly.
    per_flow_rate_gbps: float = 10.0
    mtu_bytes: int = 1500
    # Minimum mean think time between RPCs on a connection (ns). The
    # actual mean is auto-calibrated upward when needed to match
    # ``gbps``; this is a floor.
    rpc_think_time_mean_ns: float = 50_000.0

    # ----- hal_composite knobs -----------------------------------------
    # Huang et al. (HAL, ISCA'24) two-layer bursty composite workload.
    # See ``src/hal_workload.py`` for the full methodology.
    #
    # ``hal_mix_*`` are the time-average mix fractions across web / cache
    # / hadoop; must sum to 1. ``hal_total_gbps`` is the aggregate target
    # offered load -- the per-class targets are
    # ``mix_c * hal_total_gbps``. ``hal_link_gbps`` is the clip threshold
    # for the lognormal rate process (100 Gbps reproduces HAL Fig. 8
    # averages of 1.6/5.2/10.9 Gbps for web/cache/hadoop without
    # re-scaling). ``hal_rate_update_ns`` is the Layer-1 resampling
    # period; 1 ms matches the visible time-scale of rate change in
    # HAL Fig. 8.
    hal_mix_web: float = 1.0 / 3.0
    hal_mix_cache: float = 1.0 / 3.0
    hal_mix_hadoop: float = 1.0 / 3.0
    hal_total_gbps: float = 50.0
    hal_link_gbps: float = 100.0
    hal_rate_update_ns: float = 1_000_000.0
    # Number of independent HAL tenants. Per-class Layer-1 rate is the
    # sum of ``hal_n_tenants`` independent lognormal timelines, each
    # targeting ``(mix_c * hal_total_gbps) / hal_n_tenants``. Default 1
    # reproduces the single-process HAL Fig. 8 exactly; raising it
    # models multi-tenant host-sharing (the aggregate has the same
    # mean but lower variance; coincident bursts become rarer).
    hal_n_tenants: int = 1

    # ----- synthetic_sustained knobs ----------------------------------
    # Diagnostic mice/elephant workload for scheduler microbenchmarks.
    # Two components:
    #
    # 1. ELEPHANT flows: ``synth_elephant_n_flows`` long-lived CBR
    #    flows, each emitting at a stable per-flow rate for the full
    #    horizon. Stable 5-tuples -> fixed bucket assignment -> once
    #    the scheduler rebalances the RSS table the per-queue
    #    elephant load is uniform. Carries most of the aggregate
    #    offered load.
    #
    # 2. MICE flows: Poisson-arrival short-lived CBR bursts. Arrival
    #    rate ``synth_mice_arrival_rate_per_sec``; each mouse has a
    #    flow size drawn from a lognormal (``synth_mice_mean_flow_bytes``
    #    is the target mean, ``synth_mice_size_sigma`` is the
    #    dispersion) and sends packets back-to-back at
    #    ``synth_mice_per_flow_gbps`` until the byte budget is
    #    exhausted. Fresh 5-tuple per mouse so RSS-placement is
    #    continuously randomised, producing smooth Poisson-style
    #    fluctuation around the sustained elephant baseline instead
    #    of the deterministic 5-ms step that the older burst-overlay
    #    design produced.
    #
    # Justification: mice/elephant concurrency is the canonical Meta
    # workload shape (Roy et al. SIGCOMM 2015; Kandula et al. IMC
    # 2009; Benson et al. IMC 2010) -- hundreds of simultaneous
    # short-lived flows layered on a small set of long-lived bulk
    # transfers. This is NOT a published trace; it is a controlled
    # microbenchmark whose ideal scheduler output is still known
    # a-priori (near-uniform long-run byte share under a working
    # adaptive scheduler).
    synth_elephant_n_flows: int = 32
    synth_elephant_total_gbps: float = 30.0
    synth_mice_arrival_rate_per_sec: float = 100_000.0
    synth_mice_mean_flow_bytes: float = 8_000.0
    synth_mice_size_sigma: float = 1.0
    synth_mice_per_flow_gbps: float = 1.0
    synth_mtu_bytes: int = 1500


# ---------------------------------------------------------------------------
# Host / pipeline model
#
# Models the full FPGA -> DMA -> host ring -> core -> writeback pipeline. The
# critical calibration input is the measured end-to-end RTT table for a
# stateless packet (FPGA -> host -> FPGA) under no congestion. From that we
# derive T_dma(size) = RTT(size) - T_app_base - T_wb.
#
# Sizes, throughput, and occupancy are all measured end-to-end at packet
# granularity (see src/host_pipeline.py).
# ---------------------------------------------------------------------------
@dataclass
class HostModelConfig:
    # Per-packet T_app (core processing time) at STEADY state, exclusive-core.
    # DPDK poller: ~300 ns. Kernel TCP RX: ~2 us (literature: Cloudflare blog,
    # Intel TCP benchmarks). You MUST override these if you have measurements.
    stateless_t_app_mean_ns: float = 300.0
    stateless_t_app_jitter_ns: float = 50.0
    stateful_t_app_mean_ns: float = 2000.0
    stateful_t_app_jitter_ns: float = 400.0

    # Extra per-connection hash-table lookup cost for the stateful (TCP) path.
    stateful_per_conn_lookup_ns: float = 80.0

    # Descriptor writeback time (a single TLP back over PCIe).
    t_writeback_ns: float = 500.0

    # Uncongested FPGA -> host -> FPGA RTT, used to calibrate T_dma.
    # Measured on VCK190 + QDMA + PCIe (stateless path). Packet sizes in bytes,
    # RTT in nanoseconds.
    rtt_table: List[List[float]] = field(default_factory=lambda: [
        [64,   3800.0],
        [128,  3950.0],
        [256,  3980.0],
        [512,  4000.0],
        [1024, 4100.0],
        [2048, 4200.0],
        [4096, 4500.0],
    ])

    # The base T_app used to decompose the RTT table into T_dma(size).
    # Should equal the measurement's implicit per-packet core cost at zero
    # load. For the stateless calibration we use the stateless baseline.
    rtt_calibration_t_app_base_ns: float = 300.0

    # -----------------------------------------------------------------
    # PCIe link (shared between stateless and stateful domains).
    #
    # The FPGA has a finite egress staging FIFO in front of the PCIe link.
    # Packets from both domains contend for the link; when pending bytes
    # would exceed fifo_bytes a packet is dropped at the FPGA ("PCIe drop")
    # independent of descriptor-credit availability.
    #
    # Setting use_pcie_link = false falls back to the old per-packet
    # T_dma(size) model with no bandwidth contention (useful as a baseline).
    # -----------------------------------------------------------------
    use_pcie_link: bool = True
    pcie_bandwidth_gbps: float = 64.0          # VCK190 measured effective
    fpga_egress_fifo_bytes: int = 262_144      # 256 KB
    pcie_setup_ns: float = 0.0                 # additional per-packet setup
                                               # beyond what the RTT table
                                               # already captures


# ---------------------------------------------------------------------------
# HW stream arbiter (stateless vs stateful egress)
# ---------------------------------------------------------------------------
@dataclass
class HwArbiterConfig:
    policy: str = "wrr"              # wrr | strict_priority_stateless
                                     # | strict_priority_stateful | drr | random
    wrr_weight_stateless: int = 2
    wrr_weight_stateful: int = 1


# ---------------------------------------------------------------------------
# Telemetry: pressure weights
# ---------------------------------------------------------------------------
@dataclass
class TelemetryConfig:
    w1: float = 0.3   # occupancy O_q
    w2: float = 0.2   # occupancy growth G_q+
    w3: float = 0.5   # drop ratio L_q
    eps: float = 1.0e-9


# ---------------------------------------------------------------------------
# Stateless predictor
# ---------------------------------------------------------------------------
@dataclass
class PredictorConfig:
    predictor_type: str = "ewma"     # ewma | linear | oracle | tcn | none
    W_window_epochs: int = 8
    ewma_alpha: float = 0.3
    linear_lookback: int = 4
    tcn_channels: int = 16
    tcn_kernel: int = 3
    tcn_layers: int = 2
    # Path (relative to repo root or absolute) to the trained TCN
    # weights file produced by ``scripts/train_tcn.py``. Required when
    # ``predictor_type == "tcn"``; ignored otherwise.
    tcn_checkpoint: str = ""


# ---------------------------------------------------------------------------
# Schedulers
# ---------------------------------------------------------------------------
@dataclass
class StatelessSchedulerConfig:
    """Stateless scheduler selection along two orthogonal axes:

    * **signal**:  ``qp`` (pressure only), ``pred`` (predictor only),
                   ``pred_qp`` (weighted blend of predictor + pressure)
    * **policy**:  ``greedy`` (Algorithm 2 iterative greedy loop with
                   fit-condition gate) or ``oneshot`` (single pass,
                   move heaviest bucket of each hot queue to current
                   coldest queue, no fit-condition gate)

    Canonical names are ``{signal}_{policy}``:
      - static             (no scheduler)
      - qp_oneshot         pressure only, oneshot
      - qp_greedy          pressure only, greedy
      - pred_oneshot       predictor only, oneshot
      - pred_greedy        predictor only, greedy
      - pred_qp_oneshot    blend, oneshot
      - pred_qp_greedy     blend, greedy  (paper's Algorithm 2)

    Backward-compatible aliases:
      - ewma_greedy         -> pred_qp_greedy
      - reactive_greedy     -> qp_greedy
      - reactive_oneshot    -> qp_oneshot
      - proposed            -> pred_qp_greedy
      - current_only        -> qp_greedy
      - reactive_no_pred    -> qp_oneshot
    """
    scheduler_type: str = "pred_qp_greedy"
    # Weight on P_q when blending with predictor. H = alpha*P + (1-alpha)*R.
    # Only used when scheduler_type includes both ``pred`` and ``qp``.
    alpha_blend: float = 0.8
    tau_hot_s: float = 0.55
    tau_cold_s: float = 0.25
    # Tolerance for the "fit condition" in Algorithm 2: a candidate move of
    # bucket b from q_src to q_dst is accepted only if
    #     B_dst + B_b <= avg_B * (1 + fit_condition_tolerance) + 1
    # i.e. the destination must still fit within (1+tolerance) of the mean
    # load after the move. Larger values relax the constraint and allow
    # more reassignments per epoch.
    fit_condition_tolerance: float = 0.1
    # Deprecated alias for fit_condition_tolerance, retained so old YAML
    # configs still load.
    eps_B_fit_margin_frac: Optional[float] = None
    max_moves_per_epoch: int = 16

    def __post_init__(self) -> None:
        if self.eps_B_fit_margin_frac is not None:
            self.fit_condition_tolerance = float(self.eps_B_fit_margin_frac)


@dataclass
class StatefulSchedulerConfig:
    scheduler_type: str = "proposed"   # static | proposed
    eta1: float = 0.6
    eta2: float = 0.4
    tau_hot_t: float = 0.55
    tau_cold_t: float = 0.25
    lambda_t: float = 0.01
    epsilon_t: float = 0.001

    # Handoff latency is decomposed into three software-handoff phases:
    #   drain     : source core drains in-flight packets of the affected
    #               bucket -- duration varies with current queue depth and
    #               app processing speed.
    #   migration : kernel transfers per-flow metadata + application-thread
    #               ownership + TX-side ownership to the destination core.
    #   ack       : ACK TLP back to FPGA before it commits the RSS table.
    #
    # Each is modeled as Gaussian(mean, std). The total handoff latency per
    # migration is the sum of the three samples (clamped at >= 1 ns). The
    # EWMA of realised totals feeds the scheduler's penalty term.
    handoff_drain_mean_ns: float = 20_000.0
    handoff_drain_std_ns: float = 10_000.0
    handoff_migration_mean_ns: float = 60_000.0
    handoff_migration_std_ns: float = 15_000.0
    handoff_ack_mean_ns: float = 20_000.0
    handoff_ack_std_ns: float = 5_000.0

    handoff_latency_ewma_alpha: float = 0.2
    max_concurrent_handoffs: int = 2
    handoff_buffer_bytes: int = 1_000_000
    ready_flag_init: bool = True

    # Convenience fallback: if these are set in YAML (backward-compat), they
    # override the three-phase decomposition as a single Gaussian.
    handoff_latency_mean_ns: Optional[float] = None
    handoff_latency_std_ns: Optional[float] = None


# ---------------------------------------------------------------------------
# Experiment / logging
# ---------------------------------------------------------------------------
@dataclass
class ExperimentConfig:
    name: str = "default"
    rng_seed: int = 0xC0FFEE
    results_dir: str = "results"
    log_time_series: bool = True
    log_per_bucket_trace: bool = True
    make_plots: bool = True

    # Domain-isolation switches. The algorithm-evaluation methodology runs
    # in two phases:
    #   Phase 1 (isolation): run one domain at a time with the other fully
    #     disabled. Traces, pipelines, schedulers, and telemetry for the
    #     disabled side are skipped so there is no cross-domain
    #     interference on the shared PCIe link or in the output logs.
    #   Phase 2 (coexistence): both domains enabled, sharing the PCIe
    #     link, to verify realism under interference.
    # Setting ``enable_stateless: false`` (resp. ``enable_stateful``) makes
    # the simulator behave as a single-domain run for that side only.
    enable_stateless: bool = True
    enable_stateful: bool = True


# ---------------------------------------------------------------------------
# Root config + YAML loader
# ---------------------------------------------------------------------------
@dataclass
class Config:
    time: TimeConfig = field(default_factory=TimeConfig)
    topology: TopologyConfig = field(default_factory=TopologyConfig)
    workload: WorkloadConfig = field(default_factory=WorkloadConfig)
    host: HostModelConfig = field(default_factory=HostModelConfig)
    arbiter: HwArbiterConfig = field(default_factory=HwArbiterConfig)
    telemetry: TelemetryConfig = field(default_factory=TelemetryConfig)
    predictor: PredictorConfig = field(default_factory=PredictorConfig)
    stateless_scheduler: StatelessSchedulerConfig = field(default_factory=StatelessSchedulerConfig)
    stateful_scheduler: StatefulSchedulerConfig = field(default_factory=StatefulSchedulerConfig)
    experiment: ExperimentConfig = field(default_factory=ExperimentConfig)

    def __post_init__(self) -> None:
        t = self.telemetry
        s = t.w1 + t.w2 + t.w3
        if abs(s - 1.0) > 1e-6:
            raise ValueError(f"telemetry weights w1+w2+w3 must sum to 1 (got {s})")
        ss = self.stateful_scheduler
        se = ss.eta1 + ss.eta2
        if abs(se - 1.0) > 1e-6:
            raise ValueError(f"stateful weights eta1+eta2 must sum to 1 (got {se})")
        if not (0.0 <= self.stateless_scheduler.alpha_blend <= 1.0):
            raise ValueError("alpha_blend must be in [0,1]")

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Config":
        def build(sub_cls, key):
            return sub_cls(**(d.get(key, {}) or {}))
        return cls(
            time=build(TimeConfig, "time"),
            topology=build(TopologyConfig, "topology"),
            workload=build(WorkloadConfig, "workload"),
            host=build(HostModelConfig, "host"),
            arbiter=build(HwArbiterConfig, "arbiter"),
            telemetry=build(TelemetryConfig, "telemetry"),
            predictor=build(PredictorConfig, "predictor"),
            stateless_scheduler=build(StatelessSchedulerConfig, "stateless_scheduler"),
            stateful_scheduler=build(StatefulSchedulerConfig, "stateful_scheduler"),
            experiment=build(ExperimentConfig, "experiment"),
        )

    @classmethod
    def load(cls, path: str) -> "Config":
        with open(path, "r") as f:
            return cls.from_dict(yaml.safe_load(f) or {})

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def save(self, path: str) -> None:
        with open(path, "w") as f:
            yaml.safe_dump(self.to_dict(), f, sort_keys=False)
