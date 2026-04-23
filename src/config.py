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
    num_epochs: int = 1000                  # simulation horizon in stateless epochs

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

    initial_rss_stateless: str = "modulo"            # modulo | random
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
    # imc17_cdf    : per-flow generator driven by digitized CDFs from
    #                Zhang et al. IMC'17 (web / cache / hadoop). Preferred.
    # trace_mix    : legacy hand-tuned generators for web/cache/hadoop.
    # trace_csv    : load packet traces from CSV.
    # synthetic_rates: zipf / heavy-hitter / uniform rate dist + cbr/onoff.
    source: str = "imc17_cdf"   # imc17_cdf | trace_mix | trace_csv | synthetic_rates

    # ----- trace_csv path -----
    trace_file_stateless: Optional[str] = None
    trace_file_stateful: Optional[str] = None

    # ----- trace_mix path -----
    # Each entry: {kind: web|cache|hadoop|synthetic_rates, n_flows: int, gbps: float,
    #              [rate_distribution: uniform|zipf|heavy_hitter], [zipf_s], ...}
    trace_mix_stateless: List[dict] = field(default_factory=lambda: [
        {"kind": "cache",  "n_flows": 96, "gbps": 30.0},
        {"kind": "web",    "n_flows": 96, "gbps": 20.0},
        {"kind": "hadoop", "n_flows": 32, "gbps": 30.0},
    ])
    trace_mix_stateful: List[dict] = field(default_factory=lambda: [
        {"kind": "web",    "n_flows": 48, "gbps": 10.0},
        {"kind": "hadoop", "n_flows": 16, "gbps": 15.0},
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
