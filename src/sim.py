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


# ----------------------------------------------------------------------
# Domain init
# ----------------------------------------------------------------------
def _init_domain(cfg: Config, name: str, rng: np.random.Generator,
                 hasher: Toeplitz) -> DomainState:
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

    horizon_ns = int(cfg.time.epoch_ns * cfg.time.num_epochs + 1)
    trace = build_traceset_from_workload(cfg.workload, horizon_ns, n_b, hasher,
                                         rng, domain=name)

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
    telem = DomainTelemetry(
        H=cfg.time.H_bins_per_epoch, num_buckets=n_b, num_queues=n_q, D_q=D,
        w1=cfg.telemetry.w1, w2=cfg.telemetry.w2, w3=cfg.telemetry.w3,
        eps=cfg.telemetry.eps,
    )
    log = DomainMetricsLog(name=name)
    return DomainState(name=name, num_queues=n_q, num_buckets=n_b, D_q=D,
                       trace=trace, rss=rss, pipeline=pipeline,
                       telem=telem, log=log)


# ----------------------------------------------------------------------
# Simulator
# ----------------------------------------------------------------------
class Simulator:
    def __init__(self, cfg: Config) -> None:
        self.cfg = cfg
        self.rng = np.random.default_rng(cfg.experiment.rng_seed)
        self.hasher = Toeplitz()

        self.stateless = _init_domain(cfg, "stateless", self.rng, self.hasher)
        self.stateful = _init_domain(cfg, "stateful", self.rng, self.hasher)

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
        self.stateful_sched = make_stateful_scheduler(
            cfg.stateful_scheduler, self.handoff_mgr, self.rng,
            cfg.time.epoch_ns,
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

    def _process_bin_merged(self, bin_idx: int, t_start: int, t_end: int
                            ) -> None:
        sl, sf = self.stateless, self.stateful

        # 1. pull per-packet streams from each domain
        ts_sl, sz_sl, bk_sl = sl.trace.generate_bin_packets(t_start, t_end)
        ts_sf, sz_sf, bk_sf = sf.trace.generate_bin_packets(t_start, t_end)

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

        # 7. push per-domain telemetry for this bin
        self._record_domain_bin(sl, bin_idx, bb_sl, bp_sl)
        self._record_domain_bin(sf, bin_idx, bb_sf, bp_sf)

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
        for q, qp in enumerate(dom.pipeline.queues):
            adm_pkts[q] = qp.stats.adm_pkts
            adm_bytes[q] = qp.stats.adm_bytes
            drop_pkts[q] = qp.stats.drop_pkts
            drop_bytes[q] = qp.stats.drop_bytes
            credits_ret[q] = qp.stats.credits_returned
            V_end[q] = qp.snapshot_V()
        dom.telem.record_bin(
            k=bin_idx,
            bucket_bytes=bucket_bytes_gen,
            bucket_pkts=bucket_pkts_gen,
            rss_table=dom.rss.table,
            adm_pkts=adm_pkts, adm_bytes=adm_bytes,
            drop_pkts=drop_pkts, drop_bytes=drop_bytes,
            K_q_this_bin=credits_ret,
            V_q_end=V_end,
        )

    # ------------------------------------------------------------------
    # Epoch boundary
    # ------------------------------------------------------------------
    def _on_epoch_boundary(self, epoch: int) -> None:
        sl, sf = self.stateless, self.stateful
        sl_telem = sl.telem.finalize_epoch(self.cfg.time.delta_bin_ns)
        sf_telem = sf.telem.finalize_epoch(self.cfg.time.delta_bin_ns)

        # Stateless: predict + schedule.
        feat = np.stack([
            sl_telem["B_q_gen"], sl_telem["R_q_peak"],
            sl_telem["K_q"].astype(np.float64), sl_telem["P_q"],
        ], axis=1)
        self.predictor.observe(feat)
        pred_risk = self.predictor.predict()
        new_sl_table = self.stateless_sched.step(
            epoch=epoch, telem=sl_telem,
            current_table=sl.rss.table, pred_risk=pred_risk)
        sl.rss.table = new_sl_table
        sl_moves = int(getattr(self.stateless_sched, "moves_this_epoch", 0))
        sl.log.reassign_counts.append(sl_moves)

        # Stateful: advance handoffs, evaluate new moves.
        self.stateful_sched.tick_handoffs(epoch)
        new_sf_table = self.stateful_sched.step(
            epoch=epoch, telem=sf_telem, current_table=sf.rss.table)
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
        epoch_end_ns = float((epoch + 1) * self.cfg.time.epoch_ns) - 1.0
        for h in finished:
            if h.buffered_pkts <= 0:
                continue
            qp_dst = sf.pipeline.queues[h.q_dst]
            avg_size = max(1, int(h.buffered_bytes // max(1, h.buffered_pkts)))
            for _ in range(h.buffered_pkts):
                if not qp_dst.try_admit(epoch_end_ns, avg_size):
                    released_dropped_p += 1

        # Workload drift.
        shift = self.cfg.workload.pattern_shift_period_epochs
        if shift > 0 and epoch > 0 and epoch % shift == 0:
            self._shift_pattern(sl)
            self._shift_pattern(sf)

        # Sum PCIe-drop / credit-drop stats across queues for the epoch.
        def _per_domain_drop_breakdown(dom: DomainState):
            credit_p = sum(q.stats.credit_drop_pkts for q in dom.pipeline.queues)
            credit_b = sum(q.stats.credit_drop_bytes for q in dom.pipeline.queues)
            pcie_p = sum(q.stats.pcie_drop_pkts for q in dom.pipeline.queues)
            pcie_b = sum(q.stats.pcie_drop_bytes for q in dom.pipeline.queues)
            return credit_p, credit_b, pcie_p, pcie_b

        # (The per-queue stats got reset at each bin; to log PER-EPOCH totals
        # we now read the per-domain running counters that sim.py maintained.)
        # We derive them from the epoch-level telemetry dict: N_q_drop is
        # total; credit-vs-pcie breakdown comes from the just-ended bin only,
        # which is not ideal. For accurate per-epoch breakdown we would need
        # to accumulate across bins. Implemented below via _epoch_drop_accum.

        sl.log.log_epoch(epoch, sl_telem, extra={
            "reassignments": sl_moves,
            "pred_risk_max": float(pred_risk.max()) if pred_risk.size else 0.0,
            "pcie_drop_pkts": int(
                self._pcie_epoch_drop_by_domain.get("stateless", 0)),
        })
        sf.log.log_epoch(epoch, sf_telem, extra={
            "handoffs_committed": len(finished),
            "handoffs_pending": len(self.handoff_mgr.pending),
            "released_buffered_bytes": float(released_b),
            "released_buffered_pkts": int(released_p),
            "released_buffered_pkts_dropped_at_dst": int(released_dropped_p),
            "handoff_buffer_overflow_pkts": int(
                sum(h.buffer_overflow_pkts for h in finished)),
            "pcie_drop_pkts": int(
                self._pcie_epoch_drop_by_domain.get("stateful", 0)),
            "pcie_accept_pkts": int(self._pcie_epoch_accept_pkts),
            "pcie_accept_bytes": float(self._pcie_epoch_accept_bytes),
            "pcie_fifo_peak_bytes": float(self._pcie_epoch_fifo_peak),
        })

        # Reset epoch-level PCIe accumulators.
        self._pcie_epoch_accept_pkts = 0
        self._pcie_epoch_accept_bytes = 0.0
        self._pcie_epoch_drop_pkts = 0
        self._pcie_epoch_drop_bytes = 0.0
        self._pcie_epoch_drop_by_domain = {"stateless": 0, "stateful": 0}
        self._pcie_epoch_fifo_peak = 0.0
        if self.cfg.experiment.log_per_bucket_trace:
            sl.log.log_bucket_trace(epoch, sl.rss.table, sl_telem["B_b"])
            sf.log.log_bucket_trace(epoch, sf.rss.table, sf_telem["B_b"])

        # Reset per-epoch telemetry, snapshot V as start-of-next-epoch.
        sl.telem.reset_epoch()
        sf.telem.reset_epoch()
        sl.telem.set_V_start(sl.pipeline.snapshot_V())
        sf.telem.set_V_start(sf.pipeline.snapshot_V())

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
        H = cfg.time.H_bins_per_epoch
        dt = int(cfg.time.delta_bin_ns)

        self.stateless.telem.set_V_start(self.stateless.pipeline.snapshot_V())
        self.stateful.telem.set_V_start(self.stateful.pipeline.snapshot_V())

        # Per-epoch PCIe accumulators (reset at epoch boundary).
        self._pcie_epoch_accept_pkts = 0
        self._pcie_epoch_accept_bytes = 0.0
        self._pcie_epoch_drop_pkts = 0
        self._pcie_epoch_drop_bytes = 0.0
        self._pcie_epoch_drop_by_domain = {"stateless": 0, "stateful": 0}
        self._pcie_epoch_fifo_peak = 0.0

        t = 0
        for e in range(cfg.time.num_epochs):
            for k in range(H):
                t_start, t_end = t, t + dt
                self._process_bin_merged(k, t_start, t_end)
                self._accumulate_pcie_epoch()
                t = t_end
            self._on_epoch_boundary(e)

    def _accumulate_pcie_epoch(self) -> None:
        if self.pcie is None:
            return
        s = self.pcie.stats
        self._pcie_epoch_accept_pkts += s.accepted_pkts
        self._pcie_epoch_accept_bytes += s.accepted_bytes
        self._pcie_epoch_drop_pkts += s.dropped_pkts
        self._pcie_epoch_drop_bytes += s.dropped_bytes
        for d, n in s.dropped_pkts_by_domain.items():
            self._pcie_epoch_drop_by_domain[d] = (
                self._pcie_epoch_drop_by_domain.get(d, 0) + n
            )
        if s.fifo_peak_bytes > self._pcie_epoch_fifo_peak:
            self._pcie_epoch_fifo_peak = s.fifo_peak_bytes

    def save_all(self, outdir: str) -> None:
        from .metrics import write_summary
        self.stateless.log.save(outdir)
        self.stateful.log.save(outdir)

        # PCIe totals: sum the per-epoch "pcie_drop_pkts" / "pcie_accept_pkts"
        # columns we logged. These capture cross-bin totals for the whole
        # simulation.
        def _sum_col(log, col):
            return int(sum(r.get(col, 0) for r in log.rows))

        summary = {
            "stateless": self.stateless.log.summary(),
            "stateful": self.stateful.log.summary(),
            "handoffs_total": len(self.handoff_mgr.completed_history),
            "handoffs_mean_latency_ns": (
                float(np.mean([h.latency_ns for h in self.handoff_mgr.completed_history]))
                if self.handoff_mgr.completed_history else 0.0
            ),
            "pcie": {
                "bandwidth_gbps": (self.cfg.host.pcie_bandwidth_gbps
                                   if self.pcie is not None else None),
                "fifo_bytes": (self.cfg.host.fpga_egress_fifo_bytes
                               if self.pcie is not None else None),
                "total_accept_pkts": _sum_col(self.stateful.log,
                                              "pcie_accept_pkts"),
                "total_pcie_drops_stateless": _sum_col(self.stateless.log,
                                                       "pcie_drop_pkts"),
                "total_pcie_drops_stateful": _sum_col(self.stateful.log,
                                                      "pcie_drop_pkts"),
            },
        }
        write_summary(outdir, summary)
