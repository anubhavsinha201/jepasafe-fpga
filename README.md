# JEPASafe — a 1D CNN + LSTM accelerator on FPGA

Human activity recognition from a 6-axis IMU, trained in Keras and deployed to
programmable logic through hls4ml → Vitis HLS → Vivado. Target part is the
**ZCU104 (XCZU7EV)**.

The interesting content of this project is not the model — it is small and
ordinary on purpose. It is the **measured** record of what it costs to put a
recurrent network on an FPGA: which hls4ml settings matter, which ones do
nothing, and exactly where an LSTM stops fitting on a small part.

```
Train (Keras)  →  hls4ml  →  Vitis HLS  →  Vivado  →  ZCU104
   90.91%         C++ / RTL    csynth       IP + block design
```

> **On the name.** "JEPASafe" predates the architecture. The project started as
> a Transformer JEPA design; that was dropped for a 1D CNN + LSTM, which
> converts cleanly through hls4ml. The name survives in the package and the RTL
> because renaming ripples for no functional gain.

---

## Status, honestly

| Stage | State |
|---|---|
| Keras model trained | **Done** — 90.91% test accuracy, 4,374 params, 6 channels |
| hls4ml conversion from trained weights | **Done** — 109 files |
| Vitis HLS C synthesis | **Done** — 1.60 ms, timing met at 100 MHz |
| Fixed-point accuracy vs float | **Done** — `ap_fixed<16,6>` costs **0.136 points** (90.770% vs 90.906%) over all 2,947 test windows |
| RTL integration (sensor → window → inference → alert) | **Done** — all modules implemented, 7 Icarus testbenches pass |
| Vivado place & route + bitstream | **Done** — 15,173 LUT (6.6%), 57 DSP (3.3%), 96 BRAM_18K (15.4%), WNS +2.887 ns, DRC clean |
| Running on hardware | **Not done — no board available.** The bitstream exists; only the download step is outstanding |
| RTL-vs-C co-simulation | **Blocked** — `xelab.exe` is blocked by this machine's Device Guard policy |

Two things a reader should know before trusting any number here:

1. **No fall detection.** UCI HAR is six ordinary activities — walking,
   stairs, sitting, standing, laying. It contains no fall, impact or assault
   class, so despite the name this model cannot detect the event it is named
   for. Closing that needs a different dataset ([EdgeFall](https://ieee-dataport.org/open-access/edgefall))
   and a retrain. It is a dataset gap, not a hardware one.
2. **The existing cosim pass is not a numerical result.** `tb_data/` was never
   populated, so the testbench fell through to its all-zero branch. The run
   proves the RTL does not deadlock and nothing more. Fixing this is Stage 0 of
   [docs/simulation-plan.md](docs/simulation-plan.md).

Since the board is unavailable, verification proceeds in simulation —
**[docs/simulation-plan.md](docs/simulation-plan.md)** records what was proven
that way, what is blocked, and what stays unprovable without hardware.

**What is verified without a board:** the fixed-point model's accuracy over the
full test set (csim); the preprocessing path against the core's own expected
AXI-Stream payload, bit-for-bit within sensor quantisation; every hand-written
RTL module by unit testbench; and that the whole design fits the real part and
closes timing, with a real bitstream to show for it.

**What is not:** the generated core's *RTL* against its C model. That needs
co-simulation, and `xelab.exe` is blocked by Device Guard on this machine.

---

## Model

```
INPUT  (128, 6)          2.56 s window @ 50 Hz, 6 channels
   │  Conv1D(16, k=5, no bias) + BatchNorm + Activation('relu')   → (124, 16)
   │  Conv1D(16, k=5, no bias) + BatchNorm + Activation('relu')   → (120, 16)
   │  MaxPooling1D(2)                                             →  (60, 16)
   │  Dropout(0.3)          — training only
   │  LSTM(16)              — final hidden state only              →  (16,)
   │  Dense(16) + ReLU                                             →  (16,)
   │  Dense(6)                                                     →   (6,)
   ▼  class logits (not an anomaly score)
```

hls4ml folds each BatchNorm into the preceding Conv1D and drops the Dropouts,
so the synthesised graph is `Conv1D → relu → Conv1D → relu → Pooling1D → LSTM →
Dense → relu → Dense`.

**6 channels, not 9.** UCI HAR ships `body_acc`, `body_gyro` and `total_acc`.
This trains on `total_acc + body_gyro` — exactly what an MPU6050 outputs.
`body_acc` is `total_acc` with gravity filtered out, so using it would mean
reimplementing a Butterworth filter in RTL before inference.

**No input normalisation.** Training on raw physical units (g, rad/s) keeps
on-device preprocessing to a shift and one multiply instead of a per-channel
affine transform.

---

## Measured results

### Accuracy: float vs fixed point

Measured by csim over all 2,947 UCI HAR test windows, against the float Keras
logits as reference:

| | Accuracy | |
|---|---|---|
| Float (Keras) | 90.906% | baseline |
| `ap_fixed<16,6>` | **90.770%** | **0.136-point cost** |
| Argmax agreement | 99.729% | 8 of 2,947 flipped, 2 of them favourably |

16-bit needs no retrain and no per-layer precision tuning. The flips land where
the float model is already weak (STANDING↔SITTING, WALKING↔DOWNSTAIRS), so
quantisation is not this model's limiting factor.

### Accuracy vs size

| Model | Params | Test accuracy |
|---|---|---|
| 16/16/16 | 4,374 | 90.91% |
| 64/64/64 | 60,486 | 92.09% |

+1.18 points for 13.8x the parameters — and the bigger model does not fit:

| Model | LUT | DSP | BRAM_18K | FF | Latency |
|---|---|---|---|---|---|
| 16/16/16 | 33,500 (14%) | 54 (3%) | 159 (25%) | 21,724 (4%) | 1.60 ms |
| 64/64/64 | 126,167 (54%) | 198 (11%) | **654 (104%)** | 76,421 (16%) | 25.4 ms |

Those are Vitis HLS *estimates*. Vivado place & route on the shipped 16/16/16
design says otherwise, and the comparison is itself a result:

| | csynth estimate | post-route | |
|---|---|---|---|
| LUT | 33,500 | **15,173** (6.6%) | over by 2.2x |
| DSP | 54 | **57** (3.3%) | essentially exact |
| BRAM_18K | 159 | **96** (15.4%) | over by 1.7x |
| URAM | 0 | 0 | unused |

So hls4ml/Vitis estimates run conservative on logic and memory but land exactly
on DSP. Every "will it fit" decision in this project was made against LUT and
BRAM figures roughly twice the truth — which means the 64/64/64 rejection at
104% BRAM is worth re-measuring rather than trusting.

The full standalone-PL top — I2C master, scaling, window buffer, argmax, MMCM
and I/O on top of the core — adds only **660 LUTs and 3 DSPs**, and closes
timing with WNS +2.887 ns (critical path 7.11 ns, so ~140 MHz available).

Both meet 100 MHz. 64/64/64 fails on block RAM alone, by 5%. URAM is completely
unused (0 of 96 blocks) in every run, so binding the weights there is the
obvious fix — not yet done.

### Three results worth reusing elsewhere

**`layers.ReLU()` is not free — use `Activation('relu')`.** hls4ml maps
`layers.ReLU` to a *parametrized* `thresholdedrelu` carrying a live threshold
compare per element. The math is identical (ReLU's default threshold is 0), but
the two activation blocks alone measured **222,550 LUT**, more than the whole
device. One word changed; accuracy unaffected.

**`reuse_factor` cannot save a conv layer under `io_parallel`.** It reuses
multipliers *within* a dot product but still unrolls across all 124 output
positions — ~298,000 LUT for the conv front end alone. The structural fix is
`io_type='io_stream'`, where Conv1D uses a line buffer and area scales with
filter count instead of window length.

**Activation-table BRAM scales with `hidden`, not with table depth.** Setting
the LSTM's `table_size` to 256 propagates correctly into `parameters.h` and
changes the resource usage by **zero bytes**. What does change it is the hidden
width, exactly linearly:

| | `hidden=16` | `hidden=64` |
|---|---|---|
| `sigmoid` table | 24 BRAM | 96 BRAM |
| `tanh` table | 8 BRAM | 32 BRAM |

The ROM is replicated for parallel access across the `4 x hidden` gate values.
At `hidden=64` those tables are 20% of the device on their own.

### Why an LSTM does not belong on a 9k-LUT part

The original target was a Tang Nano 9K (GW1NR-9: 8,640 LUT4, 20 DSP, 26 BRAM).
The LSTM is 56% of all LUTs, and its `sigmoid` + `tanh` lookup tables alone take
**32 BRAM blocks — more than that part has in total** before a single weight is
stored. That is structural, not a tuning problem: gated recurrence needs both
nonlinearities. Against the GW1NR-9 the design measures 414% LUT / 270% DSP /
612% BRAM. On the ZU7EV the same 32 blocks are 5% of 624.

---

## Repo layout

```
├── CLAUDE.md                   # full design rationale and measurement log — the source of truth
├── docs/
│   ├── simulation-plan.md      # how to verify without a board  ← start here
│   └── build-log.md            # weekly log
├── src/jepasafe/
│   ├── cnnlstm_keras.py        # the deployable model: build + train + selfcheck
│   ├── export_hls.py           # Keras → hls4ml → Vitis HLS project
│   ├── config.py               # shared paths and dataclass configs
│   └── demopart.py             # PyTorch baseline; builds the .npy dataset cache
├── vitis_src/                  # self-contained HLS C++ sources (open directly in Vitis HLS)
├── rtl/                        # Verilog: top, I2C, window buffer, UART, AXIS TLAST adapter
│   └── tb/                     # testbenches
├── models/checkpoints/         # trained weights (.keras, .pt)
├── scripts/                    # sim / build / export_ip helpers
├── constraints/                # Gowin .cst (historical — ZCU104 needs an .xdc)
└── report/                     # LaTeX write-up + PDF
```

**Not in this repo:** the UCI HAR dataset (270 MB) and the generated
`hls/` projects (2.4 GB). Both regenerate from the commands below.

---

## Quickstart

```bash
python -m venv .venv && .venv/Scripts/activate      # Windows
pip install -r requirements.txt
export PYTHONPATH=src                               # or: pip install -e .
```

```bash
# 1. Download UCI HAR to the repo root, then build the .npy cache
#    (also trains the 9-channel PyTorch baseline for comparison)
python -m jepasafe.demopart --epochs 30

# 2. Train the deployable Keras model
python -m jepasafe.cnnlstm_keras --epochs 40
python -m jepasafe.cnnlstm_keras --selfcheck     # shapes + hls4ml sanity, no data needed

# 3. Generate the HLS project       (--precision, --parallelism, --part, --io-type)
python -m jepasafe.export_hls

# 4. C synthesis. Run by hand — hls_model.build() needs `vitis-run`, which
#    ships with full Vitis, not standalone Vitis_HLS.
export PATH="/c/Xilinx/Vitis_HLS/2023.2/bin:$PATH"
cd hls/jepasafe && vitis_hls -f build_prj.tcl
# report: hls/jepasafe/jepasafe_prj/solution1/syn/report/csynth.rpt

# 5. Package as a Vivado IP. csynth alone does NOT produce this.
cd hls/jepasafe && vitis_hls -f ../../scripts/export_ip.tcl
# -> jepasafe_prj/solution1/impl/ip/nitt_hls_jepasafe_1_0.zip
```

Prefer to skip hls4ml entirely? `vitis_src/` is a self-contained source set —
open it in Vitis HLS and synthesise. See `vitis_src/README.txt`.

### The IP's interface (measured, not assumed)

```
imu_window_TDATA    96 bit    6 channels x ap_fixed<16,6>, 128 beats per inference
layer14_out_TDATA  384 bit    6 logits, ap_fixed<37,17> each, padded to 64 bit, one beat
ap_clk, ap_rst_n, ap_start, ap_done, ap_ready, ap_idle
```

Three integration facts that follow, all worth knowing before opening Vivado:

- **No AXI-Lite.** Control is `ap_ctrl_hs`, not a register map, because hls4ml
  does not emit `#pragma HLS INTERFACE s_axilite port=return`. The PS cannot
  start it by writing a register; tie `ap_start` high for a free-running
  streaming kernel.
- **No `TLAST`.** AXI DMA's S2MM channel needs it to mark packet end.
  [`rtl/axis_tlast_gen.v`](rtl/axis_tlast_gen.v) is the thin wrapper that adds it.
- **96 and 384 are not AXI DMA stream widths.** Both divide cleanly to 32 bit
  (96 = 3x32, 384 = 12x32), so an AXI4-Stream Data Width Converter handles it.

The output logits are `ap_fixed<37,17>`, not the `<16,6>` used internally — the
final Dense emits its wide accumulator type. Irrelevant for argmax, which is
monotonic, but a host decoding real values must use 17 integer bits.

---

## Toolchain

| Tool | Version | Role |
|---|---|---|
| Python | 3.11.9 | |
| TensorFlow / Keras | 2.21.0 / 3.14.1 | training |
| PyTorch | 2.12.1+cpu | baseline only |
| hls4ml | 1.3.0 | Keras → HLS C++ |
| Vitis HLS | 2023.2 | C++ → RTL, csim, cosim |
| Vivado | 2023.2 | IP packaging, block design, synth, P&R, `xsim` |

`qkeras` is deliberately not used: precision is an hls4ml config value
(`export_hls.py --precision`), so the fixed-point model only ever exists inside
the generated HLS project.

### Verified hls4ml 1.3.0 constraints

Established by running it, not from documentation:

- **No `transpose` handler in the PyTorch parser.** This kills any
  Conv1d → LSTM torch model, since `nn.Conv1d` is channels-first and `nn.LSTM`
  wants `(batch, time, feature)`. Keras `Conv1D` is channels-last already, so
  the identical architecture converts with no transposes at all. This, not the
  recurrence, is why the deployable model is in Keras.
- **PyTorch `LSTM` itself is supported**, but only `num_layers == 1`, not
  bidirectional, `dropout == 0`, `batch_first == True`.
- **BatchNorm folds into the preceding Conv1D**; **Dropout is dropped**. Do not
  count either as a layer in the synthesised graph.
- **`convert_from_keras_model` writes nothing to disk.** It builds the graph in
  memory — `hls_model.write()` is what emits the C++ and tcl. `compile()` and
  `build()` call it implicitly, which is easy to mistake for conversion having
  done it. Miss it and your output directory stays empty.
- `hls4ml.utils.print_dict` no longer exists in 1.3.0.
- `--reuse-factor` must divide the layer's MAC count or hls4ml silently rounds
  it down. Passing 16 to this model's `conv1d` quietly becomes 15.

### Windows potholes

`hls_model.compile()` shells out to `build_lib.sh`, which cmd.exe cannot run —
use WSL or an msys64 shell with g++ on PATH. `hls_model.build()` invokes
`vitis-run`, which ships only with full Vitis; use the `vitis_hls -f
build_prj.tcl` invocation above instead.

Synthesis time depends entirely on `io_type`: **~4 minutes** for `io_stream`,
well over an hour for `io_parallel` (where peak memory hit 16 GB). If a run is
grinding, check which one you exported.

---

## Dataset

[UCI HAR](https://archive.ics.uci.edu/ml/datasets/human+activity+recognition+using+smartphones)
— 30 participants, 50 Hz, 7,352 train / 2,947 test windows. Not redistributed
here; download it to the repo root as `UCI-HAR Dataset/` and run
`jepasafe.demopart` once to build the cache.

Cite Anguita et al. (2013), *A Public Domain Dataset for Human Activity
Recognition Using Smartphones*, ESANN.

---

## References

- Duarte, J. et al. (2018). *Fast inference of deep neural networks in FPGAs for
  particle physics.* JINST 13 P07027 — the hls4ml paper.
- [hls4ml documentation](https://fastmachinelearning.org/hls4ml/) ·
  [tutorials](https://github.com/fastmachinelearning/hls4ml-tutorial)
- Anguita, D. et al. (2013). ESANN — the UCI HAR dataset paper.

---

NIT Trichy Electron Devices project. Full design rationale, every measurement
and the reasoning behind each decision are in [CLAUDE.md](CLAUDE.md).
