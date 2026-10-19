# JEPASafe: A 1D CNN + LSTM FPGA Accelerator for Real-Time Activity & Fall Recognition

## Project Documentation & Build Guide

*NIT Trichy Electron Devices Project · Updated 2026-09-27*

> **On the name.** "JEPASafe" predates the architecture. The project began as a
> Transformer JEPA design; that was dropped in favour of a 1D CNN + LSTM, which
> converts cleanly through hls4ml. The name survives in the package
> (`src/jepasafe/`), the HLS project and `jepasafe_top.v` because renaming
> ripples through the RTL for no functional gain.

> **Target board: ZCU104 (XCZU7EV).** Decided 2026-10-01, replacing the Tang Nano
> 9K. The Tang Nano analysis is kept in Resource Budget because it is what forced
> the model down to 16/16/16 — a constraint that **no longer binds**. Read that
> section before trusting any "shrink it to fit" reasoning elsewhere in this file.

---

## Table of Contents

1. [Current State](#current-state)
2. [The Pipeline](#the-pipeline)
3. [Model Architecture](#model-architecture)
4. [Repo Layout](#repo-layout)
5. [Hardware](#hardware)
6. [Toolchain](#toolchain)
7. [Resource Budget](#resource-budget)
8. [Verified hls4ml Constraints](#verified-hls4ml-constraints)
9. [Open Decisions & Integration Gaps](#open-decisions--integration-gaps)
10. [Datasets](#datasets)
11. [Budget](#budget)
12. [Glossary](#glossary)
13. [References](#references)

---

## Current State

| Stage | Status |
|---|---|
| Dataset cache (UCI HAR) | Done — `data/processed/uci_har/*.npy` |
| PyTorch baseline | Done — `models/checkpoints/cnnlstm.pt`, 9 channels, 61.7k params |
| Deployable Keras model | **Done — 90.9% test accuracy, 4,374 params, 6 channels** |
| hls4ml conversion | **Done — `hls/jepasafe/`, 109 files, from trained weights** |
| Vitis HLS C synthesis | **Done. 35,745 LUT / 54 DSP / 159 BRAM, 3.21 ms latency** |
| Board | **ZCU104 (XCZU7EV) — decided. ~16% LUT, ~3% DSP: 6x headroom** |
| RTL integration | Stubs, and the plan needs rethinking for an MPSoC — see Gaps |
| Bitstream | Not started |

The model is exported and synthesises, and the board is settled. Resources are no
longer the constraint — the measured design uses ~16% of the ZU7EV's LUTs. The
two open items are now **(a)** whether to go back to the full-size 64/64/64 model
the Tang Nano budget had forced out, and **(b)** how to integrate on an MPSoC,
where the standalone-RTL plan in `rtl/` is probably the wrong shape.

---

## The Pipeline

```
Train (Keras)  →  hls4ml  →  Vitis HLS  →  Vivado  →  ZCU104
   90.9%          C++ RTL     csynth        IP + block design + bitstream
```

Every command below is verified working on this machine.

```bash
# Set PYTHONPATH=src, or run as python -m src.jepasafe.<module>

# 1. Dataset cache + PyTorch baseline (once; writes data/processed/uci_har/)
python -m jepasafe.demopart --epochs 30

# 2. Train the deployable model
python -m jepasafe.cnnlstm_keras --epochs 40
python -m jepasafe.cnnlstm_keras --selfcheck     # shapes + hls4ml sanity, no data needed

# 3. Generate the HLS project
python -m jepasafe.export_hls                    # --precision, --reuse-factor, --part

# 4. C synthesis — by hand, see Toolchain for why
export PATH="/c/Xilinx/Vitis_HLS/2023.2/bin:$PATH"
cd hls/jepasafe && vitis_hls -f build_prj.tcl
# report: hls/jepasafe/jepasafe_prj/solution1/syn/report/csynth.rpt

# 5. RTL simulation (board-agnostic)
bash scripts/sim.sh rtl/tb/tb_led_blinker.v

# 6. Package as a Vivado IP. csynth alone does NOT produce this.
cd hls/jepasafe && vitis_hls -f ../../scripts/export_ip.tcl
# -> jepasafe_prj/solution1/impl/ip/nitt_hls_jepasafe_1_0.zip

# 7. Bitstream: Vivado. Add that zip to the IP repository, instantiate in a
#    block design with the Zynq PS, connect AXI, generate.
#    scripts/build.sh and program.sh are the old Gowin path and do not apply.
```

### What exactly goes to Vivado

**One artefact: the packaged IP.** Vivado cannot import the raw RTL that csynth
leaves in `jepasafe_prj/solution1/syn/verilog/` — it needs an IP-XACT package.

| Path | What it is |
|---|---|
| `jepasafe_prj/solution1/impl/ip/nitt_hls_jepasafe_1_0.zip` | the IP, self-contained |
| `jepasafe_prj/solution1/impl/ip/` | same thing unzipped (`component.xml` + `hdl/`) |

Add either to Vivado via **Settings → IP → Repository**. It appears in the catalog
as `nitt:hls:jepasafe:1.0`, named "JEPASafe CNN+LSTM HAR inference".

hls4ml's generated `build_opt.tcl` ships with `export 0`, and it is **regenerated
on every `export_hls.py` run**, so editing it does not stick. That is why
`scripts/export_ip.tcl` exists.

### The IP's interface (measured, not assumed)

```
module jepasafe (
    imu_window_TDATA,   imu_window_TVALID,   imu_window_TREADY,    // AXI4-Stream in
    layer14_out_TDATA,  layer14_out_TVALID,  layer14_out_TREADY,   // AXI4-Stream out
    ap_clk, ap_rst_n,
    ap_start, ap_done, ap_ready, ap_idle                           // block-level handshake
);
```

| Port | Width | Meaning |
|---|---|---|
| `imu_window_TDATA` | **96 bit** | 6 channels x `ap_fixed<16,6>`, **128 beats** per inference |
| `layer14_out_TDATA` | **384 bit** | 6 logits, one beat. Values are `ap_fixed<37,17>`, padded to 64 bit each |

Three integration facts that follow from this, all worth knowing before opening
Vivado:

1. **No AXI-Lite.** Control is `ap_ctrl_hs` (`ap_start`/`ap_done`/`ap_idle`/
   `ap_ready`), not a register map, because hls4ml does not add
   `#pragma HLS INTERFACE s_axilite port=return`. The PS therefore cannot start it
   by writing a register. Standard fix for a streaming kernel: tie `ap_start` high
   with a Constant block so it runs free, consuming whatever arrives.
2. **No `TLAST`.** The ports are TDATA/TVALID/TREADY only. AXI DMA's S2MM side
   normally uses `TLAST` to mark the end of a packet, so expect to either add it
   in a thin wrapper or run the DMA in simple mode with an exact transfer length.
3. **96 and 384 are not AXI DMA stream widths.** The DMA offers 32/64/128/…, so
   these need an AXI4-Stream Data Width Converter. Both happen to divide cleanly
   to 32 bit (96 = 3x32, 384 = 12x32), which is the obvious route — verify in
   Vivado rather than taking this on faith.

The output logits are `ap_fixed<37,17>`, not the `<16,6>` used internally — the
final Dense emits its wide accumulator type. Irrelevant for argmax, which is
monotonic, but the host must use 17 integer bits to recover real values.

---

## Model Architecture

Defined in `src/jepasafe/cnnlstm_keras.py`. Shapes below are what hls4ml
actually reports, not what was intended.

```
INPUT  (128, 6)          2.56 s window @ 50 Hz, 6 channels
   |
   |  Conv1D(16, k=5, no bias) + BatchNorm + Activation('relu')
   v      (124, 16)
   |  Conv1D(16, k=5, no bias) + BatchNorm + Activation('relu')
   v      (120, 16)
   |  MaxPooling1D(2)
   v       (60, 16)
   |  Dropout(0.3)          — training only, dropped on export
   |  LSTM(16)              — final hidden state only
   v       (16,)
   |  Dropout(0.3)
   |  Dense(16) + ReLU
   v       (16,)
   |  Dense(6)
   v        (6,)            class logits — NOT an anomaly score
```

**Classes** (UCI HAR): `WALKING`, `WALKING_UPSTAIRS`, `WALKING_DOWNSTAIRS`,
`SITTING`, `STANDING`, `LAYING`.

hls4ml folds each BatchNorm into the preceding Conv1D and drops both Dropouts,
so the synthesised graph is:

```
Input → Conv1D → Activation → Conv1D → Activation → Pooling1D → LSTM → Dense → Activation → Dense
```

### Two deliberate deviations from the PyTorch baseline

**6 channels, not 9.** The cache holds `body_acc` (0-2), `body_gyro` (3-5),
`total_acc` (6-8). The model trains on `total_acc + body_gyro` — exactly what an
MPU6050 outputs. `body_acc` is `total_acc` with gravity filtered out, so using it
would mean reimplementing UCI HAR's Butterworth filter in RTL before inference.

**No normalisation.** Training on raw physical units (g, rad/s) keeps on-device
preprocessing to a shift instead of a per-channel affine. MPU6050 at ±4g gives
8192 LSB/g, so accelerometer scaling is a right-shift by 13. Gyro is messier:
±500°/s gives 65.5 LSB/°/s and UCI HAR is in rad/s, so that path needs one
multiply-by-constant.

### The PyTorch version cannot be exported

`src/jepasafe/demopart.py` holds the same architecture in PyTorch and trains
fine, but hls4ml 1.3.0 **has no `transpose` handler**, and
`CNNLSTMClassifier.forward` needs two — `nn.Conv1d` is channels-first while
`nn.LSTM` wants `(batch, time, feature)`. Three layout variants were tried; all
fail with `Unsupported function transpose`. Keras `Conv1D` is channels-last
already, so the identical architecture converts with no transposes at all.

hls4ml's PyTorch `LSTM` handler itself is fine (single-layer, unidirectional,
`batch_first=True`, no dropout, final-state-only). The transposes are the blocker,
not the recurrence.

Keep `demopart.py`: it builds the `.npy` cache and gives a 9-channel PyTorch
number to compare against the 6-channel Keras result in the report.

---

## Repo Layout

```
.
├── CLAUDE.md                    # this file — the source of truth
├── src/jepasafe/
│   ├── cnnlstm_keras.py         # the deployable model: build + train + selfcheck
│   ├── export_hls.py            # Keras → hls4ml → Vitis HLS project
│   ├── config.py                # shared paths + TrainConfig/HLSConfig
│   ├── demopart.py              # PyTorch baseline; builds the .npy cache
│   └── model2.ipynb             # baseline driver + confusion-matrix plots
├── rtl/
│   ├── jepasafe_top.v           # sensor → window → inference → alert  (STUBBED)
│   ├── mpu6050_i2c.v            # I2C master, 6-axis read
│   ├── window_buffer.v          # sliding window → streaming output
│   ├── uart_tx.v                # for HC-08 / debug
│   ├── led_blinker.v            # toolchain hello-world
│   └── tb/                      # Icarus Verilog testbenches
├── constraints/tang_nano_9k.cst # pin assignments (Gowin)
├── scripts/                     # sim.sh / build.sh / program.sh
├── hls/jepasafe/                # generated (gitignored)
├── models/checkpoints/          # cnnlstm_keras.keras, cnnlstm.pt
├── data/processed/uci_har/      # X_{train,test}.npy, y_{train,test}.npy
├── UCI-HAR Dataset/             # raw inertial signals
└── docs/build-log.md            # weekly log — fill this in as you go
```

---

## Hardware

### Components

| Component | Purpose | Cost |
|---|---|---|
| Sipeed Tang Nano 9K | FPGA board (Gowin GW1NR-9) | ₹1,200 – ₹2,149 |
| MPU6050 (GY-521) | 6-axis accel + gyro | ₹150 |
| HC-08 | BLE module (optional) | ₹250 |
| Jumper wires, breadboard | | ₹100 |

### ZCU104 (XCZU7EV-2FFVC1156E) — the target

| Spec | Value |
|---|---|
| CLB LUTs | 230,400 |
| CLB flip-flops | 460,800 |
| Block RAM | 11.0 Mb (312 × 36 Kb) |
| UltraRAM | 27.0 Mb (96 × 288 Kb) |
| DSP slices | 1,728 |
| Processing System | quad Cortex-A53 + dual Cortex-R5F |
| Vitis part string | `xczu7ev-ffvc1156-2-e` (verified present in this install) |

The PS is the important difference from the Tang Nano: this is an MPSoC, not a
bare FPGA, so the sensor and alert paths do not have to be Verilog. See Gaps.

### Tang Nano 9K (GW1NR-9) — previous target, for reference

8,640 LUT4 · 20 DSP · 26 × 18 Kbit BSRAM (≈58.5 KB) · 27 MHz · 6 LEDs. The model
was sized for this part; see Resource Budget for why it could not work.

### Wiring

`constraints/tang_nano_9k.cst` is the authority; this table mirrors it.

| Signal | Pin | Notes |
|---|---|---|
| `sys_clk` | 52 | 27 MHz |
| `sys_rst_n` | 4 | active low |
| `led[0..5]` | 10, 11, 13, 14, 15, 16 | active low |
| `mpu_scl` | 38 | open-drain, pull-up |
| `mpu_sda` | 27 | open-drain, pull-up |
| `ble_tx` / `ble_rx` | 28 / 29 | HC-08 UART |

MPU6050 and HC-08 both run from **3.3 V — not 5 V**.

---

## Toolchain

| Tool | Version here | Role |
|---|---|---|
| Python | 3.11.9 | |
| TensorFlow / Keras | 2.21.0 / 3.14.1 | training the deployable model |
| PyTorch | 2.12.1+cpu | baseline only |
| hls4ml | 1.3.0 | Keras → HLS C++ |
| Vitis HLS | 2023.2 (`C:\Xilinx`) | C++ → RTL |
| Vivado | 2023.2 (`C:\Xilinx`) | IP packaging, block design, synth + P&R, bitstream |
| PYNQ | not set up | optional PS-side driver (see Gap 0) |
| Icarus Verilog + GTKWave | | RTL simulation |
| msys64 g++, WSL Ubuntu | present | needed for hls4ml `compile()` |

`qkeras` is **not** used. There is no separate quantization step: precision is an
hls4ml config value (`export_hls.py --precision`), so the fixed-point model only
ever exists inside the generated HLS project.

### Windows potholes (both hit, both worked around)

**`hls_model.compile()` fails.** It shells out to `build_lib.sh`, which cmd.exe
cannot run. This is the bit-accurate emulator — the thing that tells you how much
accuracy fixed-point costs *before* synthesis. Recoverable via WSL or by running
Python from an msys64 shell with g++ on PATH. Worth doing before trusting any
`--precision` choice.

**`hls_model.build()` fails.** hls4ml's Vitis backend invokes `vitis-run`, which
ships with full Vitis, not standalone Vitis_HLS 2023.2. Use the classic
invocation instead, which `export_hls.py` prints for you:

```bash
export PATH="/c/Xilinx/Vitis_HLS/2023.2/bin:$PATH"
cd hls/jepasafe && vitis_hls -f build_prj.tcl
```

Synthesis time depends entirely on `io_type`: **~4 minutes** for `io_stream`,
well over an hour for `io_parallel` (where "Standard Transforms" alone ran 838 s
and peak memory hit 16 GB). If a run is grinding, check which one you exported.

`build_prj.tcl` also runs **cosim** after csynth, which takes much longer than the
synthesis and is not needed for resource numbers. The `csynth.rpt` is written
before cosim starts, so it is safe to read the report and stop the process. Cosim
is worth running deliberately though — it is the RTL-vs-C check.

---

## Resource Budget

The arithmetic that forced the shrink, against GW1NR-9's ≈58.5 KB of BSRAM:

| Config | Params | Weights @ 8-bit | Fits BSRAM? |
|---|---|---|---|
| 64/64/64 (PyTorch baseline) | 61,700 | ≈62 KB | **No** |
| 16/16/16 (deployed) | 4,374 | ≈4.4 KB | Yes, comfortably |

Reuse factor trades DSPs for latency; it does **not** shrink weight storage. A
model whose weights exceed BSRAM cannot be fixed with `--reuse-factor` — it has
to get smaller, or stream weights from the 64 Mbit PSRAM.

`--reuse-factor` must divide the layer's MAC count. Valid values for `conv1d` in
the current model: 1, 2, 3, 5, 6, 10, 15, 30, 60, 120, 240, 480. Passing 16
silently becomes 15.

`export_hls.py` sets `static: True` on the LSTM — one reused cell stepped 60
times rather than 60 unrolled copies. Without it the recurrent layer dominates
the design.

### Measured: the first configuration did not fit at all

`io_parallel`, `Strategy: Resource`, `reuse_factor=15`, `ap_fixed<16,6>`,
per-module Vitis HLS estimates against a Zynq-7020 (53,200 LUT, 220 DSP):

| Module | BRAM_18K | DSP | FF | LUT | % Zynq LUT |
|---|---|---|---|---|---|
| `conv1d` | 16 | 32 | 34,332 | 21,729 | 40% |
| `conv1d_1` | 40 | 80 | 66,778 | 53,751 | **101%** |
| `max_pooling1d` | 0 | 0 | 0 | 35 | ~0% |
| `thresholdedrelu` #1 | 0 | 0 | 29,762 | **113,099** | **212%** |
| `thresholdedrelu` #2 | 0 | 0 | 28,802 | **109,451** | **205%** |

≈298,000 LUT for the conv front end alone — **5.6x a Zynq-7020, ~34x the Tang
Nano 9K's 8,640 LUT4** — before the LSTM and Dense layers were even reported.
Two lessons:

1. **Those `thresholdedrelu` blocks are the single biggest line item** and they
   were pure waste. Fixed by using `Activation('relu')` (see Verified hls4ml
   Constraints). This is why that one-word change matters.
2. **`reuse_factor` alone does not save a conv layer under `io_parallel`.** The
   strategy reuses multipliers *within* a dot product but still unrolls across
   all 124 output positions. The structural fix is `io_type='io_stream'`, where
   Conv1D uses a line buffer and processes one time step at a time, so area
   scales with filter count instead of window length.

### Measured: io_stream + maximum reuse

`io_stream`, `Activation('relu')`, per-layer `ReuseFactor` at each layer's full
MAC count (one multiplier, fully sequential: `conv1d` 480, `conv1d_1` 1280,
`lstm` 1024, `dense` 256, `logits` 96), `ap_fixed<16,6>`, 50 MHz. This run
completes csynth in ~4 minutes, versus well over an hour for `io_parallel`.

| Block | BRAM_18K | DSP | LUT |
|---|---|---|---|
| `conv1d` | 2 | 1 | 2,911 |
| `relu` #1 | 0 | 0 | 1,075 |
| `conv1d_1` | 3 | 1 | 4,993 |
| `relu` #2 | 0 | 0 | 1,107 |
| `max_pooling1d` | 0 | 0 | 1,141 |
| **`lstm`** | **36** | **50** | **20,128** |
| ↳ `sigmoid` table | 24 | 0 | 6,338 |
| ↳ `tanh` table | 8 | 0 | 2,274 |
| `dense` | 2 | 1 | 1,797 |
| `relu` #3 | 0 | 0 | 998 |
| `logits` | 1 | 1 | 1,017 |
| **TOTAL** | **159** | **54** | **35,745** |

Latency 160,463 cycles ≈ **3.21 ms**, against a 2.56 s window — three orders of
magnitude of slack. Reuse is already maximal, so further area cuts have to come
from the model or the precision, not the schedule.

**Against the chosen board and the rejected one:**

| | LUT | DSP | BRAM_18K | Verdict |
|---|---|---|---|---|
| **XCZU7EV (ZCU104)** | 35,745 / 230,400 = **16%** | 54 / 1,728 = **3%** | 159 / 624 = **25%** | **Fits, 6x headroom** |
| Zynq-7020 | 67% | 24% | 56% | Would fit |
| GW1NR-9 (Tang Nano 9K) | **414%** | **270%** | **612%** | Rejected |

### Why the Tang Nano was rejected (kept for the report)

The LSTM is 56% of all LUTs. Inside it, the `sigmoid` and `tanh` lookup tables
alone take **32 BRAM blocks — more than the GW1NR-9 has in total (26)**, before a
single weight or FIFO is stored. That was structural, not a tuning problem: gated
recurrence needs those two nonlinearities, and on that part they did not fit even
in isolation. On the ZU7EV those same 32 blocks are 5% of 624.

This is the strongest single result for the write-up: a concrete, measured reason
why an LSTM does not belong on a 9k-LUT part, with the numbers to back it.

### Measured: what precision and table size actually buy

Same io_stream + max-reuse config, varying only precision and the LSTM's
activation `table_size`. Latency is 3.21 ms in all three — unaffected.

| Config | LUT | DSP | BRAM | sigmoid+tanh BRAM |
|---|---|---|---|---|
| `ap_fixed<16,6>`, table 1024 | 35,745 | 54 | 159 | 32 |
| `ap_fixed<8,4>`, table 1024 | 26,526 | **16** | 109 | 32 |
| `ap_fixed<8,4>`, table 256 | 26,526 | 16 | 109 | 32 |

Two results worth keeping:

**8-bit is a real win, mostly on DSP.** LUT −26%, BRAM −31%, **DSP −70% (54 → 16,
which fits the GW1NR-9's 20)**. Config-only, no retrain. Measure the accuracy cost
before relying on it — that needs `compile()` (see Toolchain).

**`table_size` does nothing.** Setting it to 256 on the LSTM propagates correctly
into `parameters.h` (verified: the constant *is* 256 in the synthesised project),
and the resources come out byte-identical — sigmoid stays at 24 BRAM, tanh at 8.
So those 32 blocks are **not** table depth. A 256-entry 8-bit table is 256 bytes,
a fraction of one 18 Kbit block, so 24 blocks can only mean the ROM is replicated
for parallel access across the 4 x 16 gate values. If that is the mechanism it
would scale with `hidden`, not with table depth — untested, and the way to test it
is a `hidden=8` retrain.

### What the ZCU104 changes

Every area optimisation above was fighting an 8,640-LUT budget. On the ZU7EV none
of it is forced any more:

| Decision | Made because of | Still justified? |
|---|---|---|
| `Activation('relu')` over `layers.ReLU()` | 222,550 LUT of `thresholdedrelu` | **Yes** — free win, keep it always |
| `io_stream` over `io_parallel` | conv unrolling blew past 298k LUT | **Yes** — 298k still exceeds 230k, and it maps to AXI-Stream |
| `static: True` on the LSTM | 60 unrolled cells dominated the design | **Yes** — latency is a non-issue |
| max reuse (1 multiplier/layer) | DSP and LUT budget | **No longer needed.** 3.2 ms vs a 2.56 s window; keep it anyway unless latency matters, since it costs nothing |
| `ap_fixed<8,4>` | DSP 54 → 16 to fit 20 | **No longer needed.** 54 of 1,728 is 3%. Stay at 16-bit and skip the accuracy risk |
| **16/16/16 instead of 64/64/64** | 62 KB of weights vs 58.5 KB BSRAM | **No longer needed** — this is the one worth revisiting, see below |

**The model is 14x smaller than it needs to be.** 4,374 params against 61.7k was
a Tang Nano decision. 62 KB of weights is 4% of the ZU7EV's 11 Mb of BRAM. The
full-size model should be retrained and its accuracy compared against the
16/16/16's 90.9% — if it is meaningfully better, there is no longer a reason to
ship the small one.

Run `python -m jepasafe.cnnlstm_keras --filters 64 --hidden 64 --dense 64`.

### Measured: on the ZCU104

`io_stream`, max reuse, `ap_fixed<16,6>`, 100 MHz, `xczu7ev-ffvc1156-2-e`:

| | LUT | DSP | BRAM_18K | FF | Latency |
|---|---|---|---|---|---|
| 16/16/16 (4,374 params) | 33,500 (14%) | 54 (3%) | 159 (25%) | 21,724 (4%) | **1.60 ms** |

Timing met at 100 MHz with 0.37 ns slack. Latency halved from the 50 MHz run, as
expected — against a 2.56 s window it is irrelevant either way.

### Accuracy vs size

| Model | Params | Test accuracy |
|---|---|---|
| 16/16/16 (sized for Tang Nano) | 4,374 | 90.91% |
| 64/64/64 (full size) | 60,486 | **92.09%** |

+1.18 points for 13.8x the parameters.

**But it is not free in area — it overflows BRAM.** Synthesised the same way
(io_stream, max reuse, `ap_fixed<16,6>`, 100 MHz):

| Model | LUT | DSP | BRAM_18K | FF | Latency |
|---|---|---|---|---|---|
| 16/16/16 | 33,500 (14%) | 54 (3%) | 159 (25%) | 21,724 (4%) | 1.60 ms |
| 64/64/64 | 126,167 (54%) | 198 (11%) | **654 (104%)** | 76,421 (16%) | 25.4 ms |

Both meet 100 MHz. The 64/64/64 fails on block RAM alone, by 5%.

**Mechanism confirmed.** The open question above — whether the activation-table
BRAM scales with `hidden` rather than table depth — is now answered. It scales
exactly linearly:

| | `hidden=16` | `hidden=64` |
|---|---|---|
| `sigmoid` table | 24 BRAM | **96 BRAM** |
| `tanh` table | 8 BRAM | **32 BRAM** |

4x the hidden units, 4x the table BRAM, because the ROM is replicated for
parallel access across the `4 x hidden` gate values. Table *depth* is irrelevant
(see the `table_size` result above). Those 128 blocks are 20% of the device on
their own.

**The fix is URAM, which is completely unused.** 0 of 96 blocks (27 Mb) in every
run so far. Directing the large weight arrays there would relieve BRAM
immediately. Options, cheapest first:

1. Push weights to URAM — a Vitis `bind_storage` directive or hls4ml's
   `store_weights_in_bram` / resource pragmas.
2. An intermediate width, e.g. `hidden=32` (expect ~64 BRAM of tables).
3. `ap_fixed<8,4>`, which cut BRAM 31% on the small model.

Until one of those is done, **16/16/16 is the only configuration measured to fit**,
and the 1.18-point accuracy gap is the cost of not doing the work.

---

## Verified hls4ml Constraints

Established by running hls4ml 1.3.0, not from documentation:

- **No `transpose` handler in the PyTorch parser.** Kills any Conv1d→LSTM torch model.
- **PyTorch `LSTM` is supported**, but only: `num_layers == 1`, not bidirectional,
  `dropout == 0`, `batch_first == True`, no initial-state passing under `io_stream`.
- **BatchNorm folds into the preceding Conv1D.** Do not count it as a separate layer.
- **Dropout is dropped** on conversion, in both parsers.
- **`layers.ReLU()` is not free.** hls4ml maps it to `ParametrizedActivation`
  with `thresholdedrelu`, carrying a live threshold compare per element, while
  `layers.Activation('relu')` maps to plain `relu`. The math is identical
  (`ReLU`'s default threshold is 0), so always use the latter — it saves logic
  across all 3,904 conv activations. Accuracy is unchanged at 90.9%.
- **Keras CNN+LSTM converts under `io_parallel` and `io_stream`**, with and without BatchNorm.
- **`convert_from_keras_model` writes nothing to disk.** It builds the graph in
  memory; `hls_model.write()` is what emits the C++ and tcl. `compile()` and
  `build()` call it implicitly, which is easy to mistake for the conversion
  having done it. Miss it and your output directory stays empty.
- `hls4ml.utils.print_dict` no longer exists in 1.3.0.
- `InputShape` goes **inside** `hls_config`, not as a kwarg to
  `convert_from_pytorch_model`.
- The Vitis backend needs *some* Xilinx part to estimate against even if the RTL
  is ultimately carried into Gowin. `export_hls.py` defaults to `xc7z020clg400-1`.

`python -m jepasafe.cnnlstm_keras --selfcheck` asserts the shape chain, the BSRAM
weight budget, and that BN folding + Dropout removal still happen. Run it after
any hls4ml upgrade.

---

## Open Decisions & Integration Gaps

### Decided: ZCU104

Resolved 2026-10-01. This removes the worst risk in the old plan — the Vitis HLS
→ yosys → Gowin hand-off, where HLS output can instantiate Xilinx primitives that
have no Gowin equivalent. Everything now stays inside Vivado 2023.2, already
installed.

Consequences worth being explicit about:

- `constraints/tang_nano_9k.cst` and `scripts/build.sh` / `scripts/program.sh`
  (yosys + nextpnr-himbaechel + gowin_pack) **no longer apply to the build**. Kept
  for now; they are the only record of the Gowin attempt.
- `scripts/sim.sh` (Icarus Verilog) is board-agnostic and still useful.
- Most of `rtl/` was shaped by the Tang Nano — see the gaps below.

### Gap 0 (new): the whole `rtl/` approach may be the wrong shape

`rtl/jepasafe_top.v` implements sensor → window → inference → LED entirely in
Verilog, because a Tang Nano has no processor. The ZU7EV has four Cortex-A53s.
The idiomatic MPSoC flow is instead:

    HLS output → packaged IP → Vivado block design with the Zynq PS
              → AXI-Stream via DMA → driven from PYNQ (Python on the PS)

Under that flow the PS reads the MPU6050 over its own I2C, assembles windows in
DDR, DMAs them to the accelerator, and takes the argmax and the alert action in
software. That deletes the need for `mpu6050_i2c.v`, `window_buffer.v`,
`uart_tx.v` and the alert logic in `jepasafe_top.v` — four of the five RTL
modules — and makes the alert path capable of something better than an LED.

The counter-argument is pedagogical: hand-written I2C and a window buffer in
Verilog demonstrate hardware design skill that a PYNQ notebook does not, which
may matter for an Electron Devices project. **This is a real decision, not an
oversight.** Keeping the standalone-PL design on a ZCU104 is entirely possible,
just unusual.

The three gaps below apply to the standalone-PL route. Under the PYNQ route,
Gaps 2 and 3 disappear and Gap 1 becomes a few lines of Python.

### Gap 1: the alert logic expects an anomaly score

`rtl/jepasafe_top.v` compares one byte against `ALERT_THRESHOLD = 8'd205` (≈0.8).
The model emits **6 class logits**, not a score. This needs an argmax plus an
explicit decision about which classes constitute an alert. Since UCI HAR contains
no fall or assault class, a genuine safety trigger needs EdgeFall (or a collected
fall set) and a retrain — the current model can only answer "which of six ordinary
activities is this". **This gap survives either route**: it is a dataset problem,
not a hardware one.

### Gap 2: `window_buffer` parameters do not match the model

Defaults are `WINDOW_SIZE=64, N_CHANNELS=6, DATA_WIDTH=8`, and
`jepasafe_top.v` instantiates it with **no overrides**. The model needs
`WINDOW_SIZE=128` and, at `ap_fixed<16,6>`, `DATA_WIDTH=16`. (8-bit precision is
no longer worth taking just to save width — see Resource Budget.)

### Gap 3: the inference core is not instantiated

`jepasafe_top.v` hardwires `sample_q` and `anomaly_score` to zero. The hls4ml core
in `hls/jepasafe/` has to be packaged and wired in, and the int16 sensor counts
scaled into the model's input format.

### Gap 4 (new): no ZCU104 pin constraints

`constraints/` holds only the Gowin `.cst`. Any PL-side I/O — a PMOD-mounted
MPU6050, for instance — needs an `.xdc`, and the PMOD bank I/O voltage has to be
checked against the MPU6050's 3.3 V before wiring anything.

---

## Datasets

| Dataset | Why | Link |
|---|---|---|
| **UCI HAR** | In use. 30 participants, 50 Hz, published baselines | [archive.ics.uci.edu](https://archive.ics.uci.edu/ml/datasets/human+activity+recognition+using+smartphones) |
| **EdgeFall** | 1,500 fall / daily-activity trials — the actual safety class | [ieee-dataport.org](https://ieee-dataport.org/open-access/edgefall) |
| **BandX-Activity** | Same MPU6050 as the deployment target, 33 volunteers, 20 Hz | [zenodo.org](https://zenodo.org/record/8400931) |

UCI HAR is a 6-class *activity* problem. It proves the pipeline end to end, but it
cannot produce a fall detector — see Gap 1.

---

## Budget

**This table is stale — it costed the Tang Nano build.** A ZCU104 is a
development board in a completely different bracket (roughly two orders of
magnitude more than a Tang Nano at list price, though university programmes and
lab access change that entirely). Fill in the real figure, or note "institute
lab equipment", before this goes in a report.

| Item | Cost (₹) |
|---|---|
| **ZCU104 (XCZU7EV)** | **TBD — lab asset or purchase?** |
| MPU6050 (GY-521) | 150 |
| HC-08 Bluetooth | 250 |
| Jumper wires & misc | 100 |
| ~~Sipeed Tang Nano 9K~~ | ~~1,200 – 2,149~~ (no longer the target) |

Software: TensorFlow, hls4ml and Icarus Verilog are open-source. **Note the
licence question** — Vivado's free WebPACK edition does *not* cover every
UltraScale+ device, and the ZU7EV may require a paid or academic licence. Confirm
this before planning around it; it is the one toolchain cost that could bite.

---

## Glossary

| Term | Definition |
|---|---|
| **FPGA** | Field-Programmable Gate Array — reconfigurable logic chip |
| **LUT4** | 4-input look-up table — the GW1NR-9's basic logic unit |
| **DSP slice** | Hardened multiply-accumulate block |
| **BSRAM** | Block RAM — on-chip memory; holds the weights |
| **HLS** | High-Level Synthesis — C++ to RTL |
| **RTL** | Register Transfer Level — the Verilog hls4ml emits |
| **P&R** | Place & Route — mapping synthesised logic onto the real fabric |
| **Bitstream** | Binary configuration file that programs the FPGA (`.fs` on Gowin) |
| **ap_fixed<W,I>** | Fixed-point type: W total bits, I integer bits |
| **Reuse factor** | How many times one multiplier is reused; higher = fewer DSPs, more latency |
| **io_parallel / io_stream** | hls4ml layer interface: whole tensors vs streamed elements |
| **1D CNN** | 1-D convolutional network — slides filters along the time axis |
| **LSTM** | Long Short-Term Memory — recurrent cell with gated state |
| **BatchNorm folding** | Merging BN's scale/shift into the preceding conv's weights |
| **I2C / UART** | 2-wire sensor bus / asynchronous serial (HC-08) |

---

## References

### Recurrent & CNN models for HAR on embedded hardware

1. **Anguita, D., et al. (2013).** "A Public Domain Dataset for Human Activity
   Recognition Using Smartphones." ESANN. — the UCI HAR dataset paper.
2. **STORM** — 19.7k-parameter Transformer for Human Activity Recognition.
3. **StrikeWatch** — Transformer inference on an iCE40UP5K FPGA; useful for how far
   a genuinely tiny part can be pushed.

### hls4ml & FPGA deployment

4. **Duarte, J., et al. (2018).** "Fast inference of deep neural networks in FPGAs
   for particle physics." JINST 13 P07027. — the hls4ml paper.
5. **hls4ml documentation** — [fastmachinelearning.org/hls4ml](https://fastmachinelearning.org/hls4ml/)
6. **hls4ml tutorials** — [github.com/fastmachinelearning/hls4ml-tutorial](https://github.com/fastmachinelearning/hls4ml-tutorial)

### Board & toolchain

7. **Sipeed Tang Nano 9K wiki** — [wiki.sipeed.com](https://wiki.sipeed.com/hardware/en/tang/Tang-Nano-9K/Nano-9K.html)
8. **Lushay Labs Tang Nano series** — [learn.lushaylabs.com](https://learn.lushaylabs.com/tang-nano-series/)
9. **Project F / FPGA fundamentals** — for the Verilog side.

### Application context

10. Fall detection and assault-detection wearables — survey literature, for the
    report's motivation section. Note the honest framing: this project
    demonstrates the *deployment pipeline*, and the current model classifies
    ordinary activities rather than detecting assault.

---

## Build Log

Keep it in `docs/build-log.md`, not here. Challenges and solutions are the part
the report needs, so write them down while they are fresh. Record, every time
`--precision` or the layer widths change:

- float vs fixed-point accuracy (needs `compile()` — see Toolchain)
- LUT / DSP / BSRAM from `csynth.rpt`, against 8,640 LUT4 · 20 DSP · 58.5 KB

---

## Final Checklist

- [x] Trained model in Python (90.91% at 16/16/16; 92.09% at 64/64/64)
- [x] hls4ml conversion from trained weights
- [x] C synthesis read against the budget (33,500 LUT = 14% of ZU7EV)
- [x] Board chosen — ZCU104
- [ ] Decide 16/16/16 vs 64/64/64 for the shipped model
- [ ] Decide standalone-PL vs PYNQ integration (Gap 0) — gates everything below
- [ ] Confirm Vivado licence covers XCZU7EV
- [ ] Fixed-point accuracy measured against float (needs `compile()` via WSL)
- [ ] Cosim pass (RTL vs C) on the chosen model
- [ ] HLS output packaged as IP, block design built
- [ ] Alert logic redefined for class logits
- [ ] Bitstream with inference running on the board
- [ ] Sensor integration (or testbench feeding recorded data)
- [ ] Demo video (2–3 min) and presentation
- [ ] Code pushed to GitHub

---

**End of Document**
