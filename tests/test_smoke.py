"""Basic smoke tests."""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

import numpy as np

from src.config import Config
from src.hashing import Toeplitz
from src.rss import RSSIndirectionTable
from src.host_pipeline import (QueuePipeline, build_t_dma_one_way_fn,
                                build_queue_to_core)
from src.pcie import PCIeLink


def test_hashing_deterministic():
    h = Toeplitz()
    a = h.hash_5tuple(np.array([0x0A000001], np.uint32),
                      np.array([0xC0A80001], np.uint32),
                      np.array([1234], np.uint16),
                      np.array([80], np.uint16),
                      np.array([6], np.uint8))
    b = h.hash_5tuple(np.array([0x0A000001], np.uint32),
                      np.array([0xC0A80001], np.uint32),
                      np.array([1234], np.uint16),
                      np.array([80], np.uint16),
                      np.array([6], np.uint8))
    assert np.array_equal(a, b)


def test_rss_modulo():
    r = RSSIndirectionTable(num_buckets=16, num_queues=4, init="modulo")
    assert r.table.tolist() == [0, 1, 2, 3] * 4


def test_queue_pipeline_admits_under_credit():
    rtt = [[64, 3800], [1024, 4100], [4096, 4500]]
    t_dma = build_t_dma_one_way_fn(rtt, 300.0, 500.0)
    rng = np.random.default_rng(0)
    qp = QueuePipeline(queue_id=0, D_q=4,
                       t_app_mean_ns=300.0, t_app_jitter_ns=0.0,
                       t_wb_ns=500.0, per_conn_lookup_ns=0.0,
                       core_share_factor=1.0, t_dma_fn=t_dma, rng=rng)
    for t in (0, 100, 200, 300):
        assert qp.try_admit(float(t), 64)
    assert qp.credits_at_fpga == 0
    # Next admit without advance -> drop
    assert not qp.try_admit(500.0, 64)
    # Advance past RTT -> credits flow back
    qp.advance_to(5000.0)
    assert qp.credits_at_fpga == 4


def test_config_loads():
    # The canonical greedy+blend config now lives under configs/stateless/.
    cfg_path = os.path.join(os.path.dirname(HERE),
                            "configs", "stateless", "pred_qp_greedy.yaml")
    cfg = Config.load(cfg_path)
    assert cfg.time.H_bins_per_epoch == 10
    assert cfg.time.stateless_epoch_bins > 0
    assert cfg.time.stateful_epoch_bins > 0
    assert cfg.workload.source == "poisson_flow"


def test_imc17_cdf_sampler_is_monotonic():
    from src.imc17_cdf import make_sampler
    rng = np.random.default_rng(42)
    for quantity in ("flow_size_bytes", "pkt_size_bytes",
                     "iat_ns", "on_ns", "off_ns"):
        for kind in ("web", "cache", "hadoop"):
            s = make_sampler(quantity, kind)
            xs = s.sample(4096, rng)
            assert np.all(xs > 0), (quantity, kind)
            assert xs.mean() > 0


def test_stateless_scheduler_matrix_parse():
    from src.schedulers.stateless_proposed import (
        parse_scheduler_type, canonicalize)
    # Canonical parsing
    assert parse_scheduler_type("qp_oneshot") == ("qp", "oneshot")
    assert parse_scheduler_type("qp_greedy") == ("qp", "greedy")
    assert parse_scheduler_type("pred_oneshot") == ("pred", "oneshot")
    assert parse_scheduler_type("pred_greedy") == ("pred", "greedy")
    assert parse_scheduler_type("pred_qp_oneshot") == ("pred_qp", "oneshot")
    assert parse_scheduler_type("pred_qp_greedy") == ("pred_qp", "greedy")
    # Legacy aliases
    assert canonicalize("ewma_greedy") == "pred_qp_greedy"
    assert canonicalize("reactive_greedy") == "qp_greedy"
    assert canonicalize("reactive_oneshot") == "qp_oneshot"
    assert canonicalize("proposed") == "pred_qp_greedy"


def test_queue_to_core_policy():
    q2c = build_queue_to_core(8, 4, "one_to_one")
    assert q2c.tolist() == [0, 1, 2, 3, 0, 1, 2, 3]


def test_pcie_link_serializes_and_drops_on_fifo_overflow():
    # 64 Gbps => 8 ns / byte. 1024-byte packet => 128 ns serialization.
    link = PCIeLink(bandwidth_gbps=64.0, fifo_bytes=1024, setup_ns=0.0)
    t0 = link.try_transmit(0.0, 1024)
    assert t0 == 128.0, f"expected 128 ns, got {t0}"
    # Immediately send another 1024B packet at same t=0; FIFO holds
    # 1024 bytes pending, so a 1024-byte packet exactly fills it ->
    # 1024 + 1024 > fifo_bytes, drop expected.
    res = link.try_transmit(0.0, 1024)
    assert res is None, "second packet should be dropped (FIFO full)"
    assert link.stats.dropped_pkts == 1
    # After link drains, next packet should succeed.
    res = link.try_transmit(200.0, 1024)
    assert res is not None and res > 200.0


def test_flow_size_sampler_anchors_and_monotonic():
    """Canonical published flow-size CDFs must sample monotonically
    over uniform u, and the analytic mean must be positive and match
    the order-of-magnitude of the published distribution."""
    from src.workload_cdfs import CLASSES, make_flow_size_sampler
    rng = np.random.default_rng(123)
    for cls in CLASSES:
        s = make_flow_size_sampler(cls)
        # mean sane (bytes in a plausible range for datacenter flows)
        assert 100 < s.mean < 1e10, (cls, s.mean)
        xs = s.sample(10000, rng)
        # all positive, no NaN
        assert np.all(xs > 0), cls
        # monotone inverse: sorted u -> sorted x
        u = np.sort(rng.random(4096))
        log_x = np.interp(u, s.ps, s.log_xs)
        assert np.all(np.diff(log_x) >= -1e-9), cls


def test_poisson_flow_generator_hits_target_load():
    """Poisson arrivals calibrated to target Gbps should produce a
    realized rate within ~20% of target for a reasonable horizon."""
    from src.traces import _gen_poisson_flows
    rng = np.random.default_rng(7)
    horizon_ns = 200_000_000  # 200 ms
    pairs, diag = _gen_poisson_flows(
        "web_search", target_gbps=10.0, horizon_ns=horizon_ns,
        rng=rng, per_flow_rate_gbps=10.0)
    assert diag["n_flows"] > 0
    assert 8.0 < diag["realized_gbps"] < 12.5, diag


def test_rpc_generator_hits_target_load():
    """RPC model over fixed connections should match target aggregate
    Gbps after think-time calibration."""
    from src.traces import _gen_rpc_connections
    rng = np.random.default_rng(11)
    horizon_ns = 200_000_000
    pairs, diag = _gen_rpc_connections(
        "cache_follower", n_connections=64,
        target_gbps=4.0, horizon_ns=horizon_ns,
        rng=rng, per_flow_rate_gbps=10.0)
    assert len(pairs) == 64
    assert 3.0 < diag["realized_gbps"] < 5.0, diag


def test_isolated_domain_runs_produce_only_enabled_side():
    """Phase-1 isolation: when one domain is disabled, the simulator
    must not generate packets, record telemetry, or save CSV for that
    domain; the enabled side must still log ``num_epochs`` rows."""
    import yaml
    from src.sim import Simulator
    root = os.path.dirname(HERE)
    for path, want_sl, want_sf in (
        (os.path.join(root, "configs", "stateless_only",
                      "pred_qp_greedy.yaml"), True, False),
        (os.path.join(root, "configs", "stateful_only", "proposed.yaml"),
         False, True),
    ):
        d = yaml.safe_load(open(path))
        d["time"]["num_epochs"] = 10
        d["experiment"]["make_plots"] = False
        d["experiment"]["log_per_bucket_trace"] = False
        cfg = Config.from_dict(d)
        sim = Simulator(cfg)
        sim.run()
        assert sim.enable_sl is want_sl
        assert sim.enable_sf is want_sf
        sl_rows = len(sim.stateless.log.rows)
        sf_rows = len(sim.stateful.log.rows)
        assert (sl_rows > 0) == want_sl, (path, sl_rows, want_sl)
        assert (sf_rows > 0) == want_sf, (path, sf_rows, want_sf)


def test_queue_pipeline_with_external_ring_arrive():
    rtt = [[64, 3800], [1024, 4100]]
    t_dma = build_t_dma_one_way_fn(rtt, 300.0, 500.0)
    rng = np.random.default_rng(0)
    qp = QueuePipeline(queue_id=0, D_q=4,
                       t_app_mean_ns=300.0, t_app_jitter_ns=0.0,
                       t_wb_ns=500.0, per_conn_lookup_ns=0.0,
                       core_share_factor=1.0, t_dma_fn=t_dma, rng=rng)
    # Caller provides ring-arrival time (as if from PCIe link).
    assert qp.try_admit(0.0, 64, t_ring_arrive_ns=100.0)
    assert qp.credits_at_fpga == 3


if __name__ == "__main__":
    test_hashing_deterministic()
    test_rss_modulo()
    test_queue_pipeline_admits_under_credit()
    test_queue_to_core_policy()
    test_pcie_link_serializes_and_drops_on_fifo_overflow()
    test_queue_pipeline_with_external_ring_arrive()
    test_config_loads()
    test_imc17_cdf_sampler_is_monotonic()
    test_stateless_scheduler_matrix_parse()
    test_flow_size_sampler_anchors_and_monotonic()
    test_poisson_flow_generator_hits_target_load()
    test_rpc_generator_hits_target_load()
    test_isolated_domain_runs_produce_only_enabled_side()
    print("smoke tests passed")
