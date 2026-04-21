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
    cfg_path = os.path.join(os.path.dirname(HERE), "configs", "ewma_greedy.yaml")
    cfg = Config.load(cfg_path)
    assert cfg.time.H_bins_per_epoch == 10


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
    print("smoke tests passed")
