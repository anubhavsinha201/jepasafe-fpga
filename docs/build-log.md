# JEPASafe Build Log

Template from CLAUDE.md. Fill in as you go — challenges and solutions are the
part that matters for the report, so write them down while they're fresh.

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
