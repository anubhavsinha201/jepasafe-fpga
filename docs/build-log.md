# JEPASafe Build Log

Template from CLAUDE.md. Fill in as you go — challenges and solutions are the
part that matters for the report, so write them down while they're fresh.

---

## 2026-10-19 (later) — Bitstream built, and the HLS estimate audited

**`vivado/top_impl/jepasafe_top.bit` exists.** Full standalone-PL design,
synthesised, placed, routed and written for `xczu7ev-ffvc1156-2-e` against the
installed ZCU104 board file. Timing met, DRC clean, 0 critical warnings. The
only step a board would add is downloading it.

| | Core alone (OOC) | Full top | % of ZU7EV |
|---|---|---|---|
| CLB LUTs | 14,513 | **15,173** | 6.6% |
| CLB Registers | 19,260 | **19,565** | 4.3% |
| DSPs | 54 | **57** | 3.3% |
| Block RAM tiles | 48 | **48** (= 96 BRAM_18K) | 15.4% |
| URAM | 0 | **0** | 0% |
| Bonded IOB | — | 9 | 2.5% |
| MMCM | — | 1 | 12.5% |
| WNS | +2.984 ns | **+2.887 ns** | met |
| WHS | +0.023 ns | **+0.010 ns** | met, tight |

The whole preprocessing chain — I2C master, scaling, 128x96 window buffer,
argmax, MMCM and I/O — costs only **660 LUTs and 3 DSPs** on top of the core.
Critical path 7.11 ns, so the design would close around 140 MHz if the latency
ever mattered. It does not: 1.6 ms against a 2.56 s window.

**How the Vitis HLS estimate held up:**

| | csynth estimate | Vivado post-route | Verdict |
|---|---|---|---|
| LUT | 33,500 | 15,173 | over-estimated 2.2x |
| FF | 21,724 | 19,565 | close |
| DSP | 54 | 54 (core) | **exact** |
| BRAM_18K | 159 | 96 | over-estimated 1.7x |

Worth putting in the report: hls4ml/Vitis HLS estimates are *conservative on
logic and memory but exact on DSP*. Every "will it fit" decision earlier in this
project was made against LUT and BRAM numbers that turned out to be roughly
twice the truth — including the Tang Nano rejection, which was correct anyway by
a factor of four, and the 64/64/64 rejection at 654 BRAM_18K, which may deserve
a re-measure now that the estimate is known to run ~1.7x high.

**Correction, recorded because it nearly went into the report.** A first pass
counted primitives with `get_cells -hier -filter {REF_NAME =~ DSP*}` and
reported **486 DSPs**, a seeming 9x blowout against the estimate. That was a
measurement error: on UltraScale+ a `DSP48E2` is modelled as nine leaf cells
(`DSP_ALU`, `DSP_MULTIPLIER`, `DSP_PREADD_DATA`, ...), and 54 x 9 = 486 exactly.
Raw `LUT*` counting is wrong the same way, since it ignores LUT6_2 pairing
(20,608 counted vs 15,173 actual). `report_utilization` is the only authority;
both Vivado scripts now parse it instead of counting cells.

**Three ZCU104 facts, read from the board file rather than assumed.** All three
contradict something written down earlier:

| Assumed | Actual |
|---|---|
| 125 MHz single-ended user clock | **300 MHz differential pair** (AH18/AH17, DIFF_SSTL12) — the top divides it with an MMCM |
| `led[5:0]`, six LEDs | **four** (D5 D6 A5 B5, LVCMOS33) |
| PMOD voltage needs checking | **PL I2C is LVCMOS33** (N12/P12) — an MPU6050 attaches with no level shifting, so Gap 4 resolves favourably |

The ZCU104 board file is also *installed* (`xilinx.com:zcu104:part0:1.1`),
contradicting `vitis_src/README.txt`. And the Vivado licence covers XCZU7EV —
`synth_design` ran to completion, which settles that open question.

One caveat to carry forward: the PL UART is **LVCMOS18**, so an HC-08 at 3.3 V
on that path *would* need level shifting.

---

## Measurements

CLAUDE.md asks for float-vs-fixed accuracy and the resource numbers every time
precision or the layer widths change. That table:

| Model | Precision | Float acc | Fixed acc | Cost | LUT | DSP | BRAM_18K | Latency |
|---|---|---|---|---|---|---|---|---|
| 16/16/16 | `ap_fixed<16,6>` | 90.906% | **90.770%** | **0.136 pt** | 33,500 (14%) | 54 (3%) | 159 (25%) | 1.60 ms |
| 16/16/16 | `ap_fixed<8,4>` | 90.906% | not measured | — | 26,526 | 16 | 109 | 1.60 ms |
| 64/64/64 | `ap_fixed<16,6>` | 92.09% | not measured | — | 126,167 (54%) | 198 (11%) | **654 (104%)** | 25.4 ms |

All at 100 MHz on `xczu7ev-ffvc1156-2-e`, `io_stream`, max reuse, static LSTM.
64/64/64 overflows block RAM by 4%, so 16/16/16 is the only measured fit.

---

## 2026-10-19 — Fixed-point accuracy, finally measured

**The number:** `ap_fixed<16,6>` costs **0.136 accuracy points** — 90.770% vs
90.906% float, over all 2,947 UCI HAR test windows. 8 of 2,947 argmax decisions
flipped (99.73% agreement), and 2 of those flipped *toward* the correct label,
so the net damage is 4 windows.

No retrain, no per-layer precision tuning, no QAT. 16-bit is vindicated and the
open question in CLAUDE.md's precision table is closed.

Where the 8 flips went: 4 STANDING→SITTING, 2 UPSTAIRS→DOWNSTAIRS, and 2 that
improved. The confusion matrix shows the model's real weakness is unchanged by
quantisation — SITTING/STANDING confusion (94 + 54 errors) and
WALKING/DOWNSTAIRS confusion (49 + 51) dominate, and those are the same pairs
the float model struggles with. Quantisation is not what limits this model.

**Challenge: every prior simulation had been running on a window of zeros.**

`hls/jepasafe/tb_data/` was empty. `jepasafe_test.cpp` tries to open
`tb_data/tb_input_features.dat`, fails, and falls through to an else-branch that
calls `nnet::fill_zero`. So csim and the 2026-10-01 cosim both ran five
all-zero windows. The cosim report says `Verilog | Pass`, which is true and
means only that the RTL does not deadlock — all five result lines are
byte-identical. It is a liveness check wearing the costume of a correctness
check.

This is worth writing up for the report as a verification lesson: the testbench
failed *safely* and *silently*, and a green result was sitting in the repo for
18 days meaning nothing. Nothing in the hls4ml flow warns you.

**Solution:** `src/jepasafe/make_tb_data.py` writes the vectors from the test
set plus the float Keras logits as reference, and `export_hls.py` now sets
`InputData`/`OutputPredictions` so hls4ml's writer copies them in — and prints a
warning when they are missing instead of producing an empty directory.

**Subtlety worth recording:** the flatten order is load-bearing. The testbench
hands each line to `copy_data<float, input_t, 0, 128*6>`, which fills the
`hls::stream` element by element, so a line must be time-major
(`t0c0 t0c1 ... t0c5 t1c0 ...`). That is C-order `ravel()` of a `(128, 6)`
window. Channel-major would simulate perfectly happily and produce confident
nonsense, so `make_tb_data.py --selfcheck` asserts it.

**Timing, for anyone budgeting a rerun:** csim over 2,947 windows took **659 s**
(~4.5 windows/s). The reuse-factor loops run their full MAC count in C++ even
though csim does not model cycles, so it is not instant.

**Two toolchain corrections to CLAUDE.md**, both verified on this machine:

1. **Icarus Verilog is not installed.** `iverilog` is not on PATH, so
   `scripts/sim.sh` cannot run. CLAUDE.md's Toolchain table lists it as present.
   Vivado 2023.2's `xsim`/`xvlog`/`xelab` are installed and are what cosim uses.
2. The generated core is **66 files of plain Verilog-2001 with no vendor
   primitives** — no `DSP48E2`, no `RAMB36E2`, no `xpm_*`, no SystemVerilog. So
   Icarus could plausibly simulate the whole core if installed. The catch is 13
   `.dat` ROM files read via `$readmemh("./name.dat", ...)` on relative paths,
   so the sim must run with its cwd set to the `verilog/` directory or those
   files copied next to the testbench. Silent zero-filled ROMs are the failure
   mode — and they look exactly like a badly trained model.

---

## Week 1: Toolchain Setup

**Date:** _______________

**Tasks Completed:**
- [ ] Obtained ZCU104 (lab or purchase)
- [ ] Installed Vivado / Vitis HLS 2023.2
- [ ] Installed Python environment
- [ ] Ran LED Blinker tutorial

**Challenges Faced:**
1.
2.

**Solutions:**
1.
2.

---

## Week 2: Verilog Learning

**Date:** _______________

**Tasks Completed:**
- [ ] Completed Verilog basics course
- [ ] Understood always blocks, registers, combinational logic
- [ ] Simulated counter module

**Challenges Faced:**
1.
2.

**Solutions:**
1.
2.

---

## Week 3-4: Model Training

**Date:** _______________

**Tasks Completed:**
- [ ] Downloaded dataset (UCI HAR / EdgeFall / BandX)
- [ ] Trained model in Python
- [ ] Chose fixed-point precision (hls4ml config, not QKeras)
- [ ] Tested anomaly detection threshold

**Metrics:** float AUC ____ / quantized AUC ____ / threshold ____

**Challenges Faced:**
1.
2.

**Solutions:**
1.
2.

---

## Week 5-6: FPGA Deployment

**Date:** _______________

**Tasks Completed:**
- [ ] Generated HLS C++ code with hls4ml
- [ ] Packaged HLS IP and built the Vivado block design
- [ ] Generated bitstream
- [ ] Debugged resource issues

**Resource usage:** ____ / 230,400 LUT · ____ / 1,728 DSP · ____ / 624 BRAM_18K  (XCZU7EV)

**Challenges Faced:**
1.
2.

**Solutions:**
1.
2.

---

## Week 7-8: Integration & Testing

**Date:** _______________

**Tasks Completed:**
- [ ] Wrote I2C controller for MPU6050
- [ ] Integrated sensor with AI model
- [ ] Implemented LED alert
- [ ] Tested fall detection

**Challenges Faced:**
1.
2.

**Solutions:**
1.
2.

---

## Final Checklist

- [ ] Hello-world design running on the ZCU104
- [ ] Trained model + fixed-point accuracy measured
- [ ] Synthesized bitstream for the model
- [ ] Functional inference on the FPGA
- [ ] Sensor integration (or testbench simulation)
- [ ] Working alert mechanism (LED/Bluetooth)
- [ ] Documentation and presentation materials
- [ ] All code on GitHub (public or private)
- [ ] A demo video (2-3 minutes)
- [ ] A clear explanation of the flow: Train → hls4ml → Vitis HLS → Vivado → ZCU104
