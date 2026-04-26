"""Top-level simulation driver.

Three-layer per-bin pipeline (shared PCIe between stateless and stateful):

  1. Pull per-flow packet streams from both domains.
  2. HW arbiter (optional) coarsely caps total bytes per bin.
  3. Redirect packets belonging to in-progress stateful handoff buckets
     into the handoff buffer.
  4. Merge remaining packets from BOTH domains in global timestamp order
     and process one-by-one:

        FPGA layer  : advance queue clock; check V_q (credit). If 0 -> drop.
        PCIe layer  : try to transmit through shared PCIeLink.
                      - If egress FIFO full -> drop (no credit consumed).
                      - Else link busy for size*8/B ns; returns t_ring_arrive.
        Host layer  : queue pipeline uses t_ring_arrive; schedules core
                      service then writeback; credit returns after T_wb.

  5. Advance every queue's clock to t_end; snapshot V_q.

At epoch boundary:

  6. Collapse per-bin stats into epoch-level B_b, N_b, B_q_*, K_q, P_q.
  7. Stateless predictor + scheduler -> commit new stateless RSS table.
  8. Advance handoffs by epoch_ns; release buffered packets for each
     completed handoff into the destination queue; stateful scheduler
     evaluates new handoff requests.
  9. Log metrics (including PCIe drops split by cause), reset accumulators.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

from .config import Config
from .hashing import Toeplitz
from .rss import RSSIndirectionTable
from .traces import TraceSet, build_traceset_from_workload
from .telemetry import DomainTelemetry
from .arbiter import Arbiter
from .host_pipeline import (HostDomainPipeline, build_domain_pipeline,
                            build_queue_to_core)
from .pcie import PCIeLink
from .predictors import make_predictor
from .schedulers import make_stateless_scheduler, make_stateful_scheduler
from .handoff import HandoffManager
from .metrics import DomainMetricsLog


# ----------------------------------------------------------------------
# Domain bundle
# ----------------------------------------------------------------------
@dataclass
class DomainState:
    name: str
    num_queues: int
    num_buckets: int
    D_q: int
    trace: TraceSet
    rss: RSSIndirectionTable
    pipeline: HostDomainPipeline
    telem: DomainTelemetry
    log: DomainMetricsLog
    trace_diag: List = field(default_factory=list)
    # Per-domain epoch shape. ``epoch_bins`` is the number of bins the
    # domain accumulates before the scheduler fires. Stateless typically
    # short, stateful long.
    epoch_bins: int = 10
    epoch_idx: int = 0
    bin_in_epoch: int = 0
    # Each domain runs for ``num_epochs * epoch_bins * delta_bin_ns``
    # nanoseconds; once the sim clock passes this horizon the domain's
    # trace stream is exhausted and its scheduler stops firing. The
    # overall sim runs to ``max(stateless_horizon, stateful_horizon)``.
    horizon_ns: int = 0
    # When false, this domain is present only as an empty placeholder:
    # no traces are generated, no packets are admitted, the scheduler
    # never fires, and no CSV/summary rows are emitted for it.
    enabled: bool = True


# ----------------------------------------------------------------------
# Domain init
# ----------------------------------------------------------------------
def _init_domain(cfg: Config, name: str, rng: np.random.Generator,
                 hasher: Toeplitz, enabled: bool = True) -> DomainState:
    is_sf = (name == "stateful")
    n_q = cfg.topology.num_stateful_queues if is_sf else cfg.topology.num_stateless_queues
    n_b = cfg.topology.num_stateful_buckets if is_sf else cfg.topology.num_stateless_buckets
    init = cfg.topology.initial_rss_stateful if is_sf else cfg.topology.initial_rss_stateless
    D = (cfg.topology.descriptor_ring_depth_stateful if is_sf
         else cfg.topology.descriptor_ring_depth_stateless)
    n_cores = cfg.topology.num_cores_stateful if is_sf else cfg.topology.num_cores_stateless
    q2c_policy = (cfg.topology.queue_to_core_map_stateful if is_sf
                  else cfg.topology.queue_to_core_map_stateless)
    t_app_mean = cfg.host.stateful_t_app_mean_ns if is_sf else cfg.host.stateless_t_app_mean_ns
    t_app_jit = cfg.host.stateful_t_app_jitter_ns if is_sf else cfg.host.stateless_t_app_jitter_ns
    per_conn_lk = cfg.host.stateful_per_conn_lookup_ns if is_sf else 0.0

    # Each domain accumulates ``num_epochs_<domain>`` of scheduler
    # firings of its OWN epoch length. Its trace horizon is therefore
    # ``num_epochs_<domain> * epoch_bins * delta_bin_ns`` and is
    # independent of the other domain's horizon.
    sl_epoch_bins_cfg = (cfg.time.stateless_epoch_bins
                         or cfg.time.H_bins_per_epoch)
    sf_epoch_bins_cfg = (cfg.time.stateful_epoch_bins
                         or cfg.time.H_bins_per_epoch)
    my_epoch_bins = sf_epoch_bins_cfg if is_sf else sl_epoch_bins_cfg
    my_num_epochs = cfg.time.num_epochs_for(name)
    my_total_bins = my_num_epochs * my_epoch_bins
    horizon_ns = int(cfg.time.delta_bin_ns * my_total_bins + 1)

    if enabled:
        trace, trace_diag = build_traceset_from_workload(
            cfg.workload, horizon_ns, n_b, hasher, rng, domain=name)
    else:
        # Disabled domain: an empty TraceSet so any accidental access
        # yields zero packets. We still build the rest of the state
        # (pipeline, telemetry) so downstream code can treat the field
        # uniformly; ``DomainState.enabled`` gates real activity.
        trace = TraceSet(flows=[], horizon_ns=horizon_ns, num_buckets=n_b)
        trace_diag = []

    rss = RSSIndirectionTable(num_buckets=n_b, num_queues=n_q, init=init, rng=rng)

    pipeline = build_domain_pipeline(
        n_queues=n_q, n_cores=n_cores, D_q=D,
        t_app_mean_ns=t_app_mean, t_app_jitter_ns=t_app_jit,
        t_wb_ns=cfg.host.t_writeback_ns,
        per_conn_lookup_ns=per_conn_lk,
        rtt_table=cfg.host.rtt_table,
        t_app_base_ns=cfg.host.rtt_calibration_t_app_base_ns,
        policy=q2c_policy, rng=rng,
    )
    # Per-domain epoch length: the number of bins accumulated before
    # the domain's scheduler fires. If the YAML does not override it,
    # both domains fall back to the global H_bins_per_epoch.
    epoch_bins = (cfg.time.stateful_epoch_bins if is_sf
                  else cfg.time.stateless_epoch_bins)
    if epoch_bins <= 0:
        epoch_bins = cfg.time.H_bins_per_epoch
    telem = DomainTelemetry(
        H=epoch_bins, num_buckets=n_b, num_queues=n_q, D_q=D,
        w1=cfg.telemetry.w1, w2=cfg.telemetry.w2, w3=cfg.telemetry.w3,
        eps=cfg.telemetry.eps,
    )
    log = DomainMetricsLog(name=name)
    return DomainState(name=name, num_queues=n_q, num_buckets=n_b, D_q=D,
                       trace=trace, rss=rss, pipeline=pipeline,
                       telem=telem, log=log, trace_diag=trace_diag,
                       epoch_bins=epoch_bins,
                       horizon_ns=horizon_ns,
                       enabled=enabled)


# ----------------------------------------------------------------------
# Simulator
# ----------------------------------------------------------------------
class Simulator:
    def __init__(self, cfg: Config) -> None:
        self.cfg = cfg
        self.rng = np.random.default_rng(cfg.experiment.rng_seed)
        self.hasher = Toeplitz()

        self.enable_sl = bool(cfg.experiment.enable_stateless)
        self.enable_sf = bool(cfg.experiment.enable_stateful)
        if not (self.enable_sl or self.enable_sf):
            raise ValueError(
                "experiment.enable_stateless and enable_stateful cannot "
                "both be false; at least one domain must be enabled")

        self.stateless = _init_domain(cfg, "stateless", self.rng, self.hasher,
                                      enabled=self.enable_sl)
        self.stateful = _init_domain(cfg, "stateful", self.rng, self.hasher,
                                     enabled=self.enable_sf)

        max_bytes_per_bin = (cfg.workload.max_link_gbps * 1e9
                             * cfg.time.delta_bin_ns * 1e-9 / 8.0)
        self.arb = Arbiter(policy=cfg.arbiter.policy,
                           w_sl=cfg.arbiter.wrr_weight_stateless,
                           w_sf=cfg.arbiter.wrr_weight_stateful,
                           max_bytes_per_bin=max_bytes_per_bin)

        # Shared PCIe link (or None if disabled).
        if cfg.host.use_pcie_link:
            self.pcie: PCIeLink | None = PCIeLink(
                bandwidth_gbps=cfg.host.pcie_bandwidth_gbps,
                fifo_bytes=cfg.host.fpga_egress_fifo_bytes,
                setup_ns=cfg.host.pcie_setup_ns,
            )
        else:
            self.pcie = None

        self.predictor = make_predictor(self.stateless.num_queues, cfg.predictor)
        self.stateless_sched = make_stateless_scheduler(cfg.stateless_scheduler)

        ss = cfg.stateful_scheduler
        self.handoff_mgr = HandoffManager(
            n_queues=self.stateful.num_queues,
            drain_mean_ns=ss.handoff_drain_mean_ns,
            drain_std_ns=ss.handoff_drain_std_ns,
            migration_mean_ns=ss.handoff_migration_mean_ns,
            migration_std_ns=ss.handoff_migration_std_ns,
            ack_mean_ns=ss.handoff_ack_mean_ns,
            ack_std_ns=ss.handoff_ack_std_ns,
            single_phase_mean_ns=ss.handoff_latency_mean_ns,
            single_phase_std_ns=ss.handoff_latency_std_ns,
            ewma_alpha=ss.handoff_latency_ewma_alpha,
            max_concurrent=ss.max_concurrent_handoffs,
            buffer_capacity_bytes=ss.handoff_buffer_bytes,
        )
        # Stateful scheduler normalizes handoff penalty by its epoch
        # duration, which can now differ from the stateless one.
        self.stateful_sched = make_stateful_scheduler(
            cfg.stateful_scheduler, self.handoff_mgr, self.rng,
            cfg.time.stateful_epoch_ns(),
        )

    # ------------------------------------------------------------------
    # Per-bin driver: merges stateless + stateful packet streams and
    # serializes them through the shared PCIe link.
    # ------------------------------------------------------------------
    def _apply_arbiter(self, ts: np.ndarray, sz: np.ndarray, bk: np.ndarray,
                       scale: float):
        if scale >= 1.0 or ts.size == 0:
            return ts, sz, bk
        keep = self.rng.random(ts.size) < scale
        return ts[keep], sz[keep], bk[keep]

    def _apply_handoff_redirection(self, ts: np.ndarray, sz: np.ndarray,
                                   bk: np.ndarray):
        if len(self.handoff_mgr.pending) == 0 or ts.size == 0:
            return ts, sz, bk
        handoff_buckets = np.array(list(self.handoff_mgr.affected_buckets()),
                                   dtype=np.int64)
        redirect = np.isin(bk, handoff_buckets)
        if not redirect.any():
            return ts, sz, bk
        for b_id in handoff_buckets:
            mask = bk == b_id
            if mask.any():
                self.handoff_mgr.absorb_bucket_traffic(
                    int(b_id), float(sz[mask].sum()), int(mask.sum()))
        return ts[~redirect], sz[~redirect], bk[~redirect]

    def _process_bin_merged(self, t_start: int, t_end: int) -> None:
        sl, sf = self.stateless, self.stateful

        # 1. pull per-packet streams from each domain, but only within
        # each domain's own horizon. Once a domain is past its horizon
        # it has exhausted its scheduler epoch quota and we stop
        # generating packets for it. Disabled domains never generate.
        if sl.enabled and t_start < sl.horizon_ns:
            ts_sl, sz_sl, bk_sl = sl.trace.generate_bin_packets(t_start, t_end)
        else:
            ts_sl = np.empty(0, dtype=np.int64)
            sz_sl = np.empty(0, dtype=np.int32)
            bk_sl = np.empty(0, dtype=np.int64)
        if sf.enabled and t_start < sf.horizon_ns:
            ts_sf, sz_sf, bk_sf = sf.trace.generate_bin_packets(t_start, t_end)
        else:
            ts_sf = np.empty(0, dtype=np.int64)
            sz_sf = np.empty(0, dtype=np.int32)
            bk_sf = np.empty(0, dtype=np.int64)

        # Pre-admission telemetry (what the FPGA generated this bin, per bucket)
        def _bbp(bk, sz, n_b):
            if bk.size == 0:
                return (np.zeros(n_b, dtype=np.float64),
                        np.zeros(n_b, dtype=np.int64))
            bb = np.bincount(bk, weights=sz.astype(np.float64), minlength=n_b)
            bp = np.bincount(bk, minlength=n_b).astype(np.int64)
            return bb, bp
        bb_sl, bp_sl = _bbp(bk_sl, sz_sl, sl.num_buckets)
        bb_sf, bp_sf = _bbp(bk_sf, sz_sf, sf.num_buckets)

        # 2. arbiter coarse byte-level throttle (usually a no-op when
        # max_link_gbps > pcie_bandwidth_gbps; PCIe layer is the real cap).
        b_sl_tot = float(sz_sl.sum()) if ts_sl.size > 0 else 0.0
        b_sf_tot = float(sz_sf.sum()) if ts_sf.size > 0 else 0.0
        scale_sl, scale_sf = self.arb.scale_factors(b_sl_tot, b_sf_tot)
        ts_sl, sz_sl, bk_sl = self._apply_arbiter(ts_sl, sz_sl, bk_sl, scale_sl)
        ts_sf, sz_sf, bk_sf = self._apply_arbiter(ts_sf, sz_sf, bk_sf, scale_sf)

        # 3. handoff redirection for stateful
        ts_sf, sz_sf, bk_sf = self._apply_handoff_redirection(
            ts_sf, sz_sf, bk_sf)

        # 4. reset per-bin counters
        sl.pipeline.reset_bin_stats()
        sf.pipeline.reset_bin_stats()
        if self.pcie is not None:
            self.pcie.reset_bin_stats()

        # 5. merge both domains by timestamp and process one packet at a time
        if ts_sl.size + ts_sf.size > 0:
            ts_all = np.concatenate([ts_sl, ts_sf])
            sz_all = np.concatenate([sz_sl, sz_sf])
            bk_all = np.concatenate([bk_sl, bk_sf])
            dom_tag = np.concatenate([
                np.zeros(ts_sl.size, dtype=np.int8),
                np.ones(ts_sf.size, dtype=np.int8),
            ])
            order = np.argsort(ts_all, kind="stable")
            ts_all = ts_all[order]
            sz_all = sz_all[order]
            bk_all = bk_all[order]
            dom_tag = dom_tag[order]

            for i in range(ts_all.size):
                is_sf = bool(dom_tag[i] == 1)
                dom = sf if is_sf else sl
                bucket = int(bk_all[i])
                size = int(sz_all[i])
                t_gen = float(ts_all[i])
                q_id = int(dom.rss.table[bucket])
                qp = dom.pipeline.queues[q_id]

                # FPGA credit check (per-queue, per-domain).
                qp.advance_to(t_gen)
                if qp.credits_at_fpga <= 0:
                    qp.stats.drop_pkts += 1
                    qp.stats.drop_bytes += float(size)
                    qp.stats.credit_drop_pkts += 1
                    qp.stats.credit_drop_bytes += float(size)
                    continue

                # PCIe layer: try to serialize through the shared link.
                if self.pcie is not None:
                    dom_name = "stateful" if is_sf else "stateless"
                    t_complete = self.pcie.try_transmit(t_gen, size, dom_name)
                    if t_complete is None:
                        qp.record_pcie_drop(size)
                        continue
                    qp.try_admit(t_gen, size, t_ring_arrive_ns=t_complete)
                else:
                    qp.try_admit(t_gen, size)

        # 6. advance both domains' clocks to t_end
        sl.pipeline.advance_all_to(float(t_end))
        sf.pipeline.advance_all_to(float(t_end))
        if self.pcie is not None:
            self.pcie.advance_to(float(t_end))

        # 7. push per-domain telemetry for this bin, using each domain's
        # own bin-in-epoch counter. Skip entirely for disabled domains so
        # their CSVs stay empty.
        if sl.enabled:
            self._record_domain_bin(sl, sl.bin_in_epoch, bb_sl, bp_sl)
        if sf.enabled:
            self._record_domain_bin(sf, sf.bin_in_epoch, bb_sf, bp_sf)

    def _record_domain_bin(self, dom: DomainState, bin_idx: int,
                           bucket_bytes_gen: np.ndarray,
                           bucket_pkts_gen: np.ndarray) -> None:
        Q = dom.num_queues
        adm_pkts = np.zeros(Q, dtype=np.int64)
        adm_bytes = np.zeros(Q, dtype=np.float64)
        drop_pkts = np.zeros(Q, dtype=np.int64)
        drop_bytes = np.zeros(Q, dtype=np.float64)
        credits_ret = np.zeros(Q, dtype=np.int64)
        V_end = np.zeros(Q, dtype=np.int64)
        latency_samples = []
        for q, qp in enumerate(dom.pipeline.queues):
            adm_pkts[q] = qp.stats.adm_pkts
            adm_bytes[q] = qp.stats.adm_bytes
            drop_pkts[q] = qp.stats.drop_pkts
            drop_bytes[q] = qp.stats.drop_bytes
            credits_ret[q] = qp.stats.credits_returned
            V_end[q] = qp.snapshot_V()
            if qp.stats.latency_samples_ns:
                latency_samples.extend(qp.stats.latency_samples_ns)
        dom.telem.record_bin(
            k=bin_idx,
            bucket_bytes=bucket_bytes_gen,
            bucket_pkts=bucket_pkts_gen,
            rss_table=dom.rss.table,
            adm_pkts=adm_pkts, adm_bytes=adm_bytes,
            drop_pkts=drop_pkts, drop_bytes=drop_bytes,
            K_q_this_bin=credits_ret,
            V_q_end=V_end,
            latency_samples_ns=latency_samples,
        )

    # ------------------------------------------------------------------
    # Per-domain epoch boundaries.
    #
    # The two domains now advance independently: each domain has its own
    # ``epoch_bins`` count (``stateless_epoch_bins`` / ``stateful_epoch_bins``
    # in TimeConfig). When a domain's local bin counter reaches its
    # epoch size, we finalize that domain's telemetry, run its scheduler,
    # then reset its per-epoch state. The two epoch boundaries may or may
    # not coincide.
    # ------------------------------------------------------------------
    def _on_stateless_epoch_boundary(self, sl_epoch: int) -> None:
        sl = self.stateless
        sl_telem = sl.telem.finalize_epoch(self.cfg.time.delta_bin_ns)

        # Stateless: predict + schedule.
        feat = np.stack([
            sl_telem["B_q_gen"], sl_telem["R_q_peak"],
            sl_telem["K_q"].astype(np.float64), sl_telem["P_q"],
        ], axis=1)
        self.predictor.observe(feat)
        pred_risk = self.predictor.predict()
        new_sl_table = self.stateless_sched.step(
            epoch=sl_epoch, telem=sl_telem,
            current_table=sl.rss.table, pred_risk=pred_risk)
        sl.rss.table = new_sl_table
        sl_moves = int(getattr(self.stateless_sched, "moves_this_epoch", 0))
        sl.log.reassign_counts.append(sl_moves)

        # Stateless-side workload drift: only perturb stateless flows.
        # Stateful drift is applied at the stateful epoch boundary so
        # that stateful-only runs also experience workload changes.
        shift = self.cfg.workload.pattern_shift_period_epochs
        if shift > 0 and sl_epoch > 0 and sl_epoch % shift == 0:
            self._shift_pattern(sl)

        sl.log.log_epoch(sl_epoch, sl_telem, extra={
            "reassignments": sl_moves,
            "pred_risk_max": float(pred_risk.max()) if pred_risk.size else 0.0,
            "pcie_drop_pkts": int(
                self._pcie_sl_epoch_drop_by_domain.get("stateless", 0)),
        })

        # Reset epoch-local PCIe accumulators that track the stateless
        # domain's window.
        self._pcie_sl_epoch_drop_by_domain = {"stateless": 0, "stateful": 0}

        if self.cfg.experiment.log_per_bucket_trace:
            sl.log.log_bucket_trace(sl_epoch, sl.rss.table, sl_telem["B_b"])

        sl.telem.reset_epoch()
        sl.telem.set_V_start(sl.pipeline.snapshot_V())
        sl.epoch_idx += 1
        sl.bin_in_epoch = 0

    def _on_stateful_epoch_boundary(self, sf_epoch: int) -> None:
        sf = self.stateful
        sf_telem = sf.telem.finalize_epoch(self.cfg.time.delta_bin_ns)

        # Stateful-side workload drift. Applied independently of stateless
        # drift so stateful-only runs also see periodic pattern shifts.
        shift = self.cfg.workload.pattern_shift_period_epochs
        if shift > 0 and sf_epoch > 0 and sf_epoch % shift == 0:
            self._shift_pattern(sf)

        # Stateful: advance handoffs, evaluate new moves.
        self.stateful_sched.tick_handoffs(sf_epoch)
        new_sf_table = self.stateful_sched.step(
            epoch=sf_epoch, telem=sf_telem, current_table=sf.rss.table)
        finished = getattr(self.stateful_sched, "_finished", [])
        released_b = sum(h.buffered_bytes for h in finished)
        released_p = sum(h.buffered_pkts for h in finished)
        released_dropped_p = 0
        sf.rss.table = new_sf_table
        sf.log.reassign_counts.append(len(finished))

        # Replay buffered packets of each completed handoff into the
        # destination queue pipeline. Per the design document, once the
        # kernel ACKs the handoff, the FPGA commits the new RSS entry and
        # releases the buffered packets to the destination queue. Packets
        # that can't fit in the destination queue's ring are dropped there.
        epoch_end_ns = float((sf_epoch + 1)
                             * self.cfg.time.stateful_epoch_ns()) - 1.0
        for h in finished:
            if h.buffered_pkts <= 0:
                continue
            qp_dst = sf.pipeline.queues[h.q_dst]
            avg_size = max(1, int(h.buffered_bytes // max(1, h.buffered_pkts)))
            for _ in range(h.buffered_pkts):
                if not qp_dst.try_admit(epoch_end_ns, avg_size):
                    released_dropped_p += 1

        sf.log.log_epoch(sf_epoch, sf_telem, extra={
            "handoffs_committed": len(finished),
            "handoffs_pending": len(self.handoff_mgr.pending),
            "released_buffered_bytes": float(released_b),
            "released_buffered_pkts": int(released_p),
            "released_buffered_pkts_dropped_at_dst": int(released_dropped_p),
            "handoff_buffer_overflow_pkts": int(
                sum(h.buffer_overflow_pkts for h in finished)),
            "pcie_drop_pkts": int(
                self._pcie_sf_epoch_drop_by_domain.get("stateful", 0)),
            "pcie_accept_pkts": int(self._pcie_sf_epoch_accept_pkts),
            "pcie_accept_bytes": float(self._pcie_sf_epoch_accept_bytes),
            "pcie_fifo_peak_bytes": float(self._pcie_sf_epoch_fifo_peak),
        })

        self._pcie_sf_epoch_accept_pkts = 0
        self._pcie_sf_epoch_accept_bytes = 0.0
        self._pcie_sf_epoch_drop_by_domain = {"stateless": 0, "stateful": 0}
        self._pcie_sf_epoch_fifo_peak = 0.0

        if self.cfg.experiment.log_per_bucket_trace:
            sf.log.log_bucket_trace(sf_epoch, sf.rss.table, sf_telem["B_b"])

        sf.telem.reset_epoch()
        sf.telem.set_V_start(sf.pipeline.snapshot_V())
        sf.epoch_idx += 1
        sf.bin_in_epoch = 0

    def _shift_pattern(self, dom: DomainState) -> None:
        flows = dom.trace.flows
        if not flows:
            return
        n_shift = max(1, len(flows) // 10)
        idx = self.rng.choice(len(flows), size=n_shift, replace=False)
        for i in idx:
            flows[i].src_port = int(self.rng.integers(1024, 65535))
        dom.trace.rebucket(self.hasher)

    # ------------------------------------------------------------------
    # Top-level
    # ------------------------------------------------------------------
    def run(self) -> None:
        cfg = self.cfg
        dt = int(cfg.time.delta_bin_ns)

        sl, sf = self.stateless, self.stateful
        sl.telem.set_V_start(sl.pipeline.snapshot_V())
        sf.telem.set_V_start(sf.pipeline.snapshot_V())

        # Per-domain epoch-local PCIe accumulators.
        self._pcie_sl_epoch_drop_by_domain = {"stateless": 0, "stateful": 0}
        self._pcie_sf_epoch_accept_pkts = 0
        self._pcie_sf_epoch_accept_bytes = 0.0
        self._pcie_sf_epoch_drop_by_domain = {"stateless": 0, "stateful": 0}
        self._pcie_sf_epoch_fifo_peak = 0.0

        # Each domain gets ``num_epochs_<domain>`` scheduler firings of
        # its own ``epoch_bins`` length. The overall sim runs for
        #   max(sl_target_epochs*sl_bins, sf_target_epochs*sf_bins)
        # bins so each enabled domain can reach its own target epoch
        # count. For isolated single-domain runs we size ``total_bins``
        # off the enabled domain only.
        sl_epoch_bins = sl.epoch_bins
        sf_epoch_bins = sf.epoch_bins
        sl_target = cfg.time.num_epochs_for("stateless")
        sf_target = cfg.time.num_epochs_for("stateful")
        domain_total_bins = []
        if self.enable_sl:
            domain_total_bins.append(sl_target * sl_epoch_bins)
        if self.enable_sf:
            domain_total_bins.append(sf_target * sf_epoch_bins)
        total_bins = max(domain_total_bins)
        sl_epoch = 0
        sf_epoch = 0

        t = 0
        for _ in range(total_bins):
            t_start, t_end = t, t + dt
            self._process_bin_merged(t_start, t_end)
            self._accumulate_pcie_epoch()
            t = t_end

            if self.enable_sl:
                sl.bin_in_epoch += 1
            if self.enable_sf:
                sf.bin_in_epoch += 1

            if (self.enable_sl
                    and sl.bin_in_epoch >= sl_epoch_bins
                    and sl_epoch < sl_target):
                self._on_stateless_epoch_boundary(sl_epoch)
                sl_epoch += 1
            if (self.enable_sf
                    and sf.bin_in_epoch >= sf_epoch_bins
                    and sf_epoch < sf_target):
                self._on_stateful_epoch_boundary(sf_epoch)
                sf_epoch += 1
            # If a domain has hit its target epoch count we stop its
            # bin counter from climbing further so we do not trigger
            # an extra epoch finalization at horizon end.
            if sl_epoch >= sl_target:
                sl.bin_in_epoch = 0
            if sf_epoch >= sf_target:
                sf.bin_in_epoch = 0

    def _accumulate_pcie_epoch(self) -> None:
        """After each bin, fold the PCIe stats into BOTH domains'
        epoch-local accumulators. Each accumulator is reset when the
        corresponding domain closes its epoch."""
        if self.pcie is None:
            return
        s = self.pcie.stats
        # Stateless-epoch-local accumulators: we only need the per-domain
        # PCIe-drop counts so we can log them alongside each stateless
        # epoch.
        for d, n in s.dropped_pkts_by_domain.items():
            self._pcie_sl_epoch_drop_by_domain[d] = (
                self._pcie_sl_epoch_drop_by_domain.get(d, 0) + n
            )
        # Stateful-epoch-local accumulators: richer, since we also
        # report FIFO peak and accepted byte volume over the stateful
        # epoch window.
        self._pcie_sf_epoch_accept_pkts += s.accepted_pkts
        self._pcie_sf_epoch_accept_bytes += s.accepted_bytes
        for d, n in s.dropped_pkts_by_domain.items():
            self._pcie_sf_epoch_drop_by_domain[d] = (
                self._pcie_sf_epoch_drop_by_domain.get(d, 0) + n
            )
        if s.fifo_peak_bytes > self._pcie_sf_epoch_fifo_peak:
            self._pcie_sf_epoch_fifo_peak = s.fifo_peak_bytes

    def save_all(self, outdir: str) -> None:
        from .metrics import write_summary
        if self.stateless.enabled:
            self.stateless.log.save(outdir)
        if self.stateful.enabled:
            self.stateful.log.save(outdir)

        # PCIe totals: sum the per-epoch "pcie_drop_pkts" / "pcie_accept_pkts"
        # columns we logged. These capture cross-bin totals for the whole
        # simulation.
        def _sum_col(log, col):
            return int(sum(r.get(col, 0) for r in log.rows))

        # Realism checklist: target vs realized aggregate rate per class,
        # per domain. Raises a warning (as a flag in summary) when the
        # error exceeds 5%.
        def _classify_diag(diags):
            rows = []
            for d in diags:
                t = d.get("target_gbps") or 0.0
                r = d.get("realized_gbps") or 0.0
                err = ((r - t) / t) if t > 0 else 0.0
                rows.append({**d, "rate_error_fraction": err,
                             "warn_rate_mismatch": abs(err) > 0.05})
            return rows

        workload_realism = {}
        if self.stateless.enabled:
            workload_realism["stateless"] = _classify_diag(
                self.stateless.trace_diag)
        if self.stateful.enabled:
            workload_realism["stateful"] = _classify_diag(
                self.stateful.trace_diag)

        summary = {
            "enabled_domains": {
                "stateless": bool(self.enable_sl),
                "stateful": bool(self.enable_sf),
            },
            "epochs": {
                "stateless_epoch_bins": int(self.stateless.epoch_bins),
                "stateful_epoch_bins": int(self.stateful.epoch_bins),
                "delta_bin_ns": float(self.cfg.time.delta_bin_ns),
                "stateless_epoch_ns": float(self.cfg.time.stateless_epoch_ns()),
                "stateful_epoch_ns": float(self.cfg.time.stateful_epoch_ns()),
            },
            "workload_realism": workload_realism,
            "pcie": {
                "bandwidth_gbps": (self.cfg.host.pcie_bandwidth_gbps
                                   if self.pcie is not None else None),
                "fifo_bytes": (self.cfg.host.fpga_egress_fifo_bytes
                               if self.pcie is not None else None),
                "total_accept_pkts": (_sum_col(self.stateful.log,
                                               "pcie_accept_pkts")
                                      if self.stateful.enabled else 0),
                "total_pcie_drops_stateless": (_sum_col(self.stateless.log,
                                                        "pcie_drop_pkts")
                                               if self.stateless.enabled
                                               else 0),
                "total_pcie_drops_stateful": (_sum_col(self.stateful.log,
                                                       "pcie_drop_pkts")
                                              if self.stateful.enabled
                                              else 0),
            },
        }
        if self.stateless.enabled:
            summary["stateless"] = self.stateless.log.summary()
        if self.stateful.enabled:
            summary["stateful"] = self.stateful.log.summary()
            summary["handoffs_total"] = len(
                self.handoff_mgr.completed_history)
            summary["handoffs_mean_latency_ns"] = (
                float(np.mean([h.latency_ns
                               for h in self.handoff_mgr.completed_history]))
                if self.handoff_mgr.completed_history else 0.0
            )
        write_summary(outdir, summary)
