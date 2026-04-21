# Software Emulation: Telemetry-Driven RSS Flow Scheduling

Software emulator for the SmartNIC-emulated platform (VCK190 + QDMA) that
models the proposed telemetry + prediction + scheduling framework for
dynamic RSS flow steering. Tracks the design document faithfully for
everything the scheduling algorithm depends on; skips the parts that do
not affect scheduling dynamics (see "Out of scope" below).

## Emulated system

```
              +------------------ FPGA (VCK190) --------------------+
              | stateless gen           stateful gen (TCP stack)    |
              |   |                        |                        |
              |   +---> Toeplitz hash --->   (shared HW module)     |
              |          |                  |                       |
              |   +---> RSS indir table --> RSS indir table         |
              |   |                                                 |
              |   +---> telemetry collector (per-bucket, per-queue) |
              |   +---> stateless scheduler (ML predictor + greedy) |
              |   +---> stateful  scheduler (reactive + handoff)    |
              +-----------------+-------+---------------------------+
                                | QDMA  |
                                v       v        <--- PCIe Gen4 x8 --->
              +--- Host (SR-IOV, two VFs) --------------------------+
              | stateless VF (DPDK PMD)  | stateful VF (kernel TCP)|
              | cores, queues pinned     | cores, queues pinned    |
              +----------------------+---+------+-------------------+
                                     |          |
                                     v          v
                               application threads (pinned)
```

Two disjoint **partitions** per domain. A partition per the design doc is:
RX queue + TX queue + processing core + (stateful) interrupt core +
application thread. The emulator models this as:

* one `QueuePipeline` per queue (one CPU core worth of processing) in
  `src/host_pipeline.py`;
* the pipeline's `T_app` + per-conn lookup + writeback captures both kernel
  and app processing cost;
* queue-to-core mapping is fixed (`queue_to_core_map_*`) — scheduling only
  changes the RSS indirection table (bucket → queue), never the queue → core
  binding.

Because the partition is fixed, **queue pressure `P_q` is a direct proxy
for core availability** — if the core is slow or overloaded, descriptors
queue up in the host ring, `U_q` grows, drops increase, and `P_q` rises.

## End-to-end per-packet model (three layers)

Packets from both stateless and stateful domains contend for a **shared PCIe
link**. Each packet passes through three layers, and drops can occur at two
independent stages:

```
┌─ FPGA layer ────────────────────────────────────────────────────┐
│ generator emits (t_gen, size, bucket)                           │
│ → RSS bucket → queue q                                          │
│ → V_q credit check                                              │
│   if V_q == 0  → CREDIT DROP  (ring full at host)              │
│   else consume one credit and forward to PCIe                   │
└─────────────────────┬───────────────────────────────────────────┘
                      │
┌─ PCIe layer (shared between both domains) ─────────────────────┐
│ FPGA egress staging FIFO (fifo_bytes, default 256 KB)           │
│ Link bandwidth B (default 64 Gbps)                              │
│   if pending_bytes + size > fifo_bytes  → PCIE DROP             │
│   (no credit consumed; packet never reached the host)           │
│ else link busy for size*8/B ns; serialize both domains in time  │
└─────────────────────┬───────────────────────────────────────────┘
                      │  t_ring_arrive = t_pcie_complete
                      v
┌─ Host layer (one per queue) ───────────────────────────────────┐
│ QDMA C2H ring (depth D_q, FIFO)                                 │
│   packet waits if core busy                                     │
│ Core processing T_app                                           │
│   stateless: DPDK poller, ~300 ns/pkt                           │
│   stateful : kernel TCP RX, ~2000 ns/pkt + 80 ns conn lookup    │
│ Descriptor writeback TLP (T_wb)                                 │
│ → FPGA observes credit (V_q += 1)                               │
└─────────────────────────────────────────────────────────────────┘
```

Under zero congestion this reproduces the measured stateless RTT:
`RTT(size) ≈ size*8/B + T_app + T_wb`.
Under congestion two separate drop mechanisms are active:

* **Credit drop** — host core can't drain its ring fast enough; descriptor
  occupancy saturates; new packets find `V_q = 0`. (The original pressure
  source `P_q` in the scheduling algorithm.)
* **PCIe drop** — even if the host could accept more, the shared PCIe link
  is saturated; the FPGA egress FIFO fills; new packets are dropped before
  DMA. Splits by domain so telemetry can tell who is causing the congestion.

Both are counted in total `drop_bytes` / `drop_pkts` feeding the `L_q` term
of `P_q`. Per-epoch logs additionally expose `pcie_drop_pkts` split by
domain so you can distinguish which layer is the bottleneck.

`T_dma(size)` in the earlier single-layer model is now replaced by the PCIe
layer. The legacy uncongested `T_dma(size) = RTT(size) - T_app_base - T_wb`
formula from the RTT table still exists for the `use_pcie_link: false`
fallback (useful as an uncongested baseline). Your measured RTT table
(`host.rtt_table`):

| size | measured RTT |
|---:|---:|
| 64 B | 3.80 µs |
| 128 B | 3.95 µs |
| 256 B | 3.98 µs |
| 512 B | 4.00 µs |
| 1024 B | 4.10 µs |
| 2048 B | 4.20 µs |
| 4096 B | 4.50 µs |

## Workload (per-flow packet traces)

Each flow has a 5-tuple and a pre-computed timeline
`(timestamps_ns, sizes_bytes)` spanning the sim horizon. Sources:

* **`trace_mix`** — compose `web` / `cache` / `hadoop` / `synthetic_rates`
  kinds (Meta microburst style, IMC '17).
* **`trace_csv`** — load real per-packet traces from CSV
  (`flow_id, timestamp_ns, size_bytes [, src_ip, dst_ip, src_port, dst_port, proto]`).
* **`synthetic_rates`** — parametric: `flow_rate_distribution`
  (`uniform` / `zipf` / `heavy_hitter`) × `packet_size_distribution`
  (`fixed` / `imix`) × `burstiness_model` (`cbr` / `onoff`).

Flow's RSS bucket is the real Toeplitz hash of its 5-tuple.

## Schedulers

### Stateless (Algorithm 2, committed directly in HW)

| scheduler_type | behaviour |
|---|---|
| `static` | baseline; RSS never changes |
| `ewma_greedy` | `H_q = α·P_q + (1-α)·R̂_q`; greedy reassignment gated by the paper's fit condition (tolerance set via `fit_condition_tolerance`) |
| `reactive_greedy` | ablation: `α = 1` (no predictor) |
| `reactive_oneshot` | ablation: threshold-only, move heaviest bucket of each hot queue to coldest queue (no fit-condition check, no iterative refresh) |

### Stateful (Algorithm 3, host-coordinated handoff)

Only `proposed` or `static`. The proposed path:

1. HW evaluates `Gain_t(b, q_dst) = Benefit - λ_t · T̂_hand / Δ`; if
   `> ε_t`, issues a handoff request.
2. HW notifies source core, redirects new packets of the bucket into an
   internal buffer (cap: `handoff_buffer_bytes`; overflow → drop).
3. Software performs three phases in sequence:
   * **drain**     — source core finishes in-flight packets of the bucket
   * **migration** — kernel moves per-flow state, app-thread ownership, and
     TX-side ownership to the destination partition
   * **ack**       — kernel ACKs the FPGA
4. FPGA commits the new RSS entry and releases buffered packets to the
   destination queue.

Each phase is modeled as an independent Gaussian sample; total handoff
latency is the sum. EWMA of realised totals feeds the penalty term.

## Run

```bash
pip install -r requirements.txt

python3 scripts/run_single.py --config configs/ewma_greedy.yaml
python3 scripts/run_comparison.py --configs-dir configs --out results/comparison
```

## Key knobs for the three layers

### FPGA layer
* `topology.descriptor_ring_depth_*` — `D_q`; smaller → credit drops sooner.
* `host.t_writeback_ns` — how quickly freed descriptors become credits.

### PCIe layer
* `host.use_pcie_link` — if `false`, falls back to uncongested `T_dma(size)`
  (no bandwidth contention; useful as a baseline).
* `host.pcie_bandwidth_gbps` — link throughput (default 64 Gbps).
* `host.fpga_egress_fifo_bytes` — FIFO in front of the PCIe link; when
  pending bytes exceed this, packets are dropped.
* `host.pcie_setup_ns` — additional per-packet fixed setup beyond what the
  RTT table already captures (typically 0 if the RTT table is used).
* `workload.max_link_gbps` (arbiter) — coarse byte-level pre-throttle;
  usually set ≥ `pcie_bandwidth_gbps` so it is effectively a no-op and the
  PCIe layer is the binding constraint.

### Host layer
* `host.stateless_t_app_*`, `host.stateful_t_app_*`,
  `host.stateful_per_conn_lookup_ns` — per-packet service time.
* `topology.num_cores_*`, `topology.queue_to_core_map_*` — if cores < queues
  on a domain, pipelines multiply their service time by the share factor.

### Output: where drops come from
Per-epoch CSV columns per domain include `pcie_drop_pkts`. In
`summary.json`, the top-level `pcie` block aggregates total PCIe-layer
accepts and drops split by domain. Compare that with per-epoch
`total_N_drop` (all drops) to see the credit vs. PCIe breakdown:
`credit_drops = total_N_drop - pcie_drop_pkts`.

## Out of scope (by design)

These appear in the hardware design doc but are not modeled because they
do not affect the scheduling algorithm's behaviour:

* **Internal FPGA bus-width serialization** (`generation_cycles = size / bus_width`):
  per-flow peak rate caps are already enforced by the trace's inter-arrival
  times; aggregate FPGA-to-host bandwidth is capped by `max_link_gbps`.
* **Hardware timestamp insertion** for RTT measurement — intra-host latency
  experiments, not scheduling.
* **TCP state machine** (TCB, RTO, retransmission) — scheduling sees
  packets as bytes/pkts per bucket; protocol correctness is assumed.
* **Active-open src_port reverse-hash search** — flows are pre-generated
  with 5-tuples; their first-time partition assignment is implicit in the
  hash. The port-selection step is a one-off setup detail that is
  preserved across subsequent migrations (migrations only change the RSS
  indirection table), so the runtime behaviour is the same as the real
  system.
* **SR-IOV VF separation** — functionally equivalent to disjoint queue
  sets, which the emulator models directly.
* **MMIO control path** — the host programs flow characteristics via MMIO
  on the real device; here the same characteristics come straight from the
  YAML config.

## Layout

```
src/
  config.py          # dataclass + YAML loader (all user-tunable params)
  hashing.py         # Toeplitz 5-tuple hash
  rss.py             # RSS indirection table
  traces.py          # per-flow packet traces + workload kinds + CSV loader
  host_pipeline.py   # per-queue host pipeline: ring + core + writeback
  pcie.py            # shared PCIe link: bandwidth + egress FIFO + drops
  arbiter.py         # HW stream arbiter (WRR / strict priority / DRR / random)
  telemetry.py       # per-bin / per-epoch accumulators, P_q computation
  handoff.py         # stateful software handoff (3-phase latency)
  predictors/        # ewma (default), linear, oracle, tcn, none
  schedulers/        # static, ewma_greedy, reactive_greedy, reactive_oneshot, + stateful
  sim.py             # top-level driver
  metrics.py, plotting.py
configs/             # one YAML per experiment
scripts/             # run_single / run_comparison
tests/               # unit tests
```

## Extending

* New predictor: subclass `src.predictors.base.BasePredictor`, drop in
  `src/predictors/`, reference by name in YAML `predictor.predictor_type`.
* New scheduler: subclass `src.schedulers.base.BaseStatelessScheduler` (or
  `BaseStatefulScheduler`), drop in `src/schedulers/`, reference by name
  in YAML `*_scheduler.scheduler_type`.
* New workload kind: add a `_gen_xxx` in `src/traces.py` and route it in
  `_dispatch_kind`.
* New arbiter policy: extend `src/arbiter.py::Arbiter.scale_factors`.
