# Board-Free Verification Plan

*How to finish JEPASafe without a ZCU104 on the desk. Written 2026-10-19.*

The board is unavailable. This plan replaces every claim the hardware would
have supported with a claim simulation can support, and states plainly which
claims cannot be made at all.

**Route decision:** this plan targets the **standalone-PL** design — sensor →
window → core → alert entirely in the programmable logic (CLAUDE.md "Gap 0").
That choice is partly *because* the board is gone: the PYNQ route's whole value
is Python running on the Cortex-A53s, which cannot be simulated in any
meaningful way, whereas every stage of the all-RTL design can be.

---

## What the board would have proven, and what replaces it

| Hardware claim | Simulation substitute | Stage |
|---|---|---|
| Fixed-point model is as accurate as the float one | csim over all 2,947 test windows | 2 |
| The RTL computes what the C says | cosim, same vectors, bit-compare | 3 |
| It fits the part and meets timing | Vivado synth + implementation reports | 6 |
| Sensor → window → core → alert works | xsim testbench, recorded samples from ROM | 5 |
| Live demo | waveform walkthrough + three-way numeric agreement table | 7 |
| Real MPU6050 works | **nothing — unprovable without hardware** | — |

---

## Stage 0 — Unblock: generate `tb_data/` (do this first)

**`hls/jepasafe/tb_data/` is empty.** This is the most important finding behind
this plan. The testbench `vitis_src/jepasafe_test.cpp` opens
`tb_data/tb_input_features.dat`, fails, and silently falls through to its
`fill_zero` branch:

```cpp
} else {
    std::cout << "INFO: Unable to open input/predictions file, using default input." << std::endl;
    const unsigned NUM_TEST_SAMPLES = 5;
    ...  nnet::fill_zero<input_t, 128*6>(imu_window);
```

So the cosim run of 2026-10-01 that reports `Verilog | Pass` proved only that
the RTL does not deadlock. Its `rtl_cosim_results.log` holds five *identical*
lines — the logits for an all-zero window:

```
-4.04395 -7.22949 -3.30566 -0.71875 -5.66602 11.6904
```

Nothing numerical has been verified yet. Every stage below depends on fixing
this, and nothing below is worth doing until it is.

**Task.** Add `src/jepasafe/make_tb_data.py`:

- Load `data/processed/uci_har/X_test.npy` (2,947 x 128 x 9) and narrow to the
  6 deployable channels with the existing `cnnlstm_keras.CHANNEL_IDX` — do not
  re-derive that index list, it is already correct.
- Write `tb_input_features.dat`: one window per line, 768 space-separated
  floats, time-major to match `nnet::copy_data<float, input_t, 0, 128*6>`.
- Write `tb_output_predictions.dat`: one line per window, the 6 **float Keras
  logits** from `models/checkpoints/cnnlstm_keras.keras`. These are the
  reference, not the labels.
- `--num-vectors N` to emit a stratified subset (Stage 3 needs this).
- Also write `y_subset.npy` alongside, so the Python comparison in Stage 2 can
  line vectors up with ground-truth labels.

**Wire it into the export.** hls4ml's writer only copies these into `tb_data/`
if the config names them. In `export_hls.py:build_config`, add:

```python
config["InputData"]         = str(tb_dir / "tb_input_features.dat")
config["OutputPredictions"] = str(tb_dir / "tb_output_predictions.dat")
```

**Verify:** after re-running `python -m jepasafe.export_hls`,
`wc -l hls/jepasafe/tb_data/*.dat` must print 2947 twice. If `tb_data/` is
still empty the rest of this document is void.

---

## Stage 1 — Optional: the bit-accurate C model via `compile()`

`hls_model.compile()` builds a shared library you can call from Python, which
makes precision sweeps a one-liner. It needs bash + g++, so it fails under
cmd.exe; run it from WSL Ubuntu (installed, currently stopped) or an msys64
shell with g++ on PATH.

**This stage is genuinely optional.** Stage 2 produces the same fixed-point
logits straight out of csim, with no WSL dependency at all. Do Stage 1 only if
you intend to sweep several precisions — then it pays for itself. Otherwise
skip it; the WSL environment needs its own TensorFlow + hls4ml install, which
is a long detour for one number.

---

## Stage 2 — csim: fixed-point accuracy over the full test set

This is the stage that closes the longest-standing open item in CLAUDE.md's
checklist: *"Fixed-point accuracy measured against float"*.

csim runs the C++ model, not the RTL, so all 2,947 windows are cheap — minutes,
not hours. `build_prj.tcl` already runs csim before csynth.

```bash
export PATH="/c/Xilinx/Vitis_HLS/2023.2/bin:$PATH"
cd hls/jepasafe && vitis_hls -f build_prj.tcl
# -> hls/jepasafe/tb_data/csim_results.log   (2947 lines x 6 logits)
```

Then compare in Python — argmax of the fixed-point logits against `y_test`, and
separately against argmax of the Keras float logits:

| Metric | Expected | Meaning |
|---|---|---|
| Float accuracy | 90.91% (known) | the baseline |
| `ap_fixed<16,6>` accuracy | measure | **the number the report needs** |
| Argmax agreement, fixed vs float | measure | how often quantisation flips a decision |

Produce the 6x6 confusion matrix for the fixed-point run too. If the
fixed-point accuracy comes out within a few tenths of a point, that is a clean
result and `ap_fixed<16,6>` is vindicated. If it drops noticeably, the likely
culprit is *not* the input format: 6 integer bits give a range of +/-32 at a
resolution of 2^-10 ~ 0.001, which is ample for accelerometer g and gyro rad/s.
A large drop would point at an intermediate layer instead — re-export with
per-layer precision rather than reaching for a retrain.

Worth repeating the whole stage at `ap_fixed<8,4>` once `<16,6>` is recorded.
CLAUDE.md measures 8-bit at -26% LUT and -70% DSP; the only thing missing has
always been its accuracy cost. One extra csim run supplies it.

---

## Stage 3 — cosim: does the RTL match the C?

Same vectors, now through the actual generated Verilog, driven by xsim.

**Use a subset.** cosim simulates ~160k clock cycles per window; 2,947 windows
is not viable. Generate 60 vectors, 10 per class, with
`make_tb_data.py --num-vectors 60`. Budget roughly 5-15 s per window and
measure the real rate on the first run before walking away from it.

**Pass criterion:** `rtl_cosim_results.log` is *identical* to
`csim_results.log` for those 60 windows. Not close — identical. Both are the
same fixed-point arithmetic, so any difference is a bug in the generated RTL or
in the stream handshaking, and it is worth chasing rather than tolerating.

This is the real "RTL vs C" check, and the `Verilog | Pass` already sitting in
`sim/report/jepasafe_cosim.rpt` is not it.

---

## Stage 4 — Build the standalone-PL RTL (the bulk of the work)

Four of the five modules in `rtl/` are stubs that hardwire their outputs to
zero. They also encode Tang Nano assumptions that no longer hold. The measured
core interface is the contract to build against:

```
imu_window_TDATA   96 bit   = 6 channels x ap_fixed<16,6>, 128 beats per inference
layer14_out_TDATA  384 bit  = 6 logits, ap_fixed<37,17> each, padded to 64 bit, one beat
ap_start / ap_done / ap_ready / ap_idle     (ap_ctrl_hs — no AXI-Lite register map)
```

### 4a. `window_buffer.v` — implement it

Current defaults are `WINDOW_SIZE=64, N_CHANNELS=6, DATA_WIDTH=8`, and
`jepasafe_top.v` instantiates it with no overrides at all. The model needs
`WINDOW_SIZE=128` and `DATA_WIDTH=16`.

**There is a second mismatch that CLAUDE.md's Gap 2 does not mention:** the
output port is declared `output wire [DATA_WIDTH-1:0] stream_data` — one
channel wide. The core takes **all six channels in a single 96-bit beat**, so
this must become `[N_CHANNELS*DATA_WIDTH-1:0]`. Getting this wrong produces a
design that simulates happily and gives confident nonsense, so fix the port
width before writing the logic behind it.

Implementation: circular buffer of 128 x 96 bits in block RAM, streaming the
full window out on each `STRIDE` boundary, with real TVALID/TREADY
backpressure — the core *will* stall, because its interval is 160k cycles while
samples arrive at 50 Hz.

### 4b. `sample_scale.v` — new, small

int16 MPU6050 counts to `ap_fixed<16,6>` (10 fractional bits). No normalisation
was baked into training precisely so this stays cheap:

| Channel | Sensor scale | Conversion | Cost |
|---|---|---|---|
| Accel (+/-4 g) | 8192 LSB/g | `count >> 3` | a shift |
| Gyro (+/-500 deg/s) | 65.5 LSB per deg/s, target rad/s | `(count * 279) >> 10` | one multiply |

The gyro constant: 1/65.5 x pi/180 x 2^10 = 0.2727, and round(0.2727 x 1024) =
279, an error of 0.09%.

Note that CLAUDE.md's "right-shift by 13" for the accelerometer was derived for
a narrower fixed-point format and is **wrong for `ap_fixed<16,6>`** — it would
scale the input down by 1024x. Recompute per format: the shift is
log2(LSB_per_unit) - fractional_bits.

Keep both scale factors as module parameters. A real MPU6050 has offset and
sensitivity error; the calibration knob has to exist even though simulation
will never need it.

### 4c. `argmax6.v` — new, small

Replaces the broken alert comparator. Slices six 64-bit lanes out of the
384-bit bus, sign-extends the `ap_fixed<37,17>` fields, compares them as signed
integers, emits a 3-bit class index plus `valid`. Argmax is monotonic so no
rescaling is needed — the padded integer comparison is exact.

### 4d. `jepasafe_top.v` — rewire

- Instantiate the real core from `hls/jepasafe/jepasafe_prj/solution1/syn/verilog/`.
- Tie `ap_start` high — there is no AXI-Lite port to start it from.
- Replace `anomaly_score >= ALERT_THRESHOLD` with a class-mask parameter:
  `ALERT_CLASSES = 6'b000000`, i.e. **nothing triggers an alert by default**,
  because UCI HAR contains no fall class. See the honesty note at the end.
- Drop `DATA_WIDTH = 8` and the `8'd205` threshold; both are dead.

### 4e. `mpu6050_i2c.v` — defer, or do it deliberately

Simulation does not need it: drive the chain from a ROM of recorded samples.
Implement the I2C master only if the pedagogical argument for hand-written
hardware matters for an Electron Devices submission — and if you do, it needs a
behavioral slave to verify against (`rtl/tb/mpu6050_model.v`): ACK the device
address, serve the PWR_MGMT_1 / ACCEL_CONFIG / GYRO_CONFIG writes, then return
ROM bytes for the 14-byte burst read from 0x3B. A simulated slave tests your
master's logic, not the bus, but it does catch real bugs — timing, NACK
handling, the big-endian byte pairing.

---

## Stage 5 — End-to-end testbench

`rtl/tb/tb_jepasafe_top.v`: `$readmemh` a ROM of N recorded windows as **int16
sensor counts** (inverse-scale the UCI HAR floats in `make_tb_data.py`, so the
testbench exercises `sample_scale.v` rather than bypassing it), clock the whole
chain, and check the emitted class index per window.

**Pass criterion:** the class the RTL chain predicts equals the class csim
predicted for the same window in Stage 2. That links this stage back to a
number already trusted, instead of introducing a new reference.

### Simulator: xsim, not Icarus

Two corrections to CLAUDE.md's Toolchain table, both verified on this machine:

- **Icarus Verilog is not installed.** `iverilog` is not on PATH. CLAUDE.md
  lists it as present. `scripts/sim.sh` cannot run today.
- **Vivado 2023.2's `xsim` / `xvlog` / `xelab` are installed** and are what
  cosim already uses.

The good news, also verified: the generated core is **66 files of plain
Verilog-2001 with no vendor primitives** — no `DSP48E2`, no `RAMB36E2`, no
`xpm_*`, no SystemVerilog. So Icarus could plausibly elaborate the whole core
if you install it, which would make the fast iteration loop free. The one catch
is 13 `.dat` ROM files loaded as `$readmemh("./name.dat", rom0)` with relative
paths — the simulation must run with its working directory set to the
`verilog/` folder, or those files copied next to the testbench. Silent
zero-filled ROMs are the failure mode, and they look exactly like a model that
trained badly.

Recommended split:

| Path | Tool | Scope |
|---|---|---|
| `scripts/sim.sh` | Icarus (install it) | unit tests: `axis_tlast_gen`, `window_buffer`, `argmax6`, `sample_scale`, `uart_tx` |
| `scripts/sim_xsim.tcl` | Vivado xsim | full chain including the hls4ml core |

Cheap first check before committing to either:
`iverilog -g2012 -o /dev/null -y <verilog-dir> jepasafe.v`. One minute, and it
decides the tooling.

---

## Stage 6 — Vivado implementation: "it would have fit"

A full synth + place & route produces the utilization and timing numbers the
board would have confirmed, and writes a real bitstream — you simply never
download it. For a report this is nearly as strong as running it.

**Do this check first, because it can block everything:** confirm the Vivado
licence covers XCZU7EV. The free WebPACK edition does not cover every
UltraScale+ device. If it does not, Stages 0-5 all still work (Vitis HLS csim
and cosim are not gated the same way) but this stage stops dead, so find out
before planning around it.

Then write `constraints/zcu104.xdc` (Gap 4 — `constraints/` holds only the
Gowin `.cst`). Check all of the following against the ZCU104 user guide
**UG1267** rather than assuming:

- **Clock.** Either the on-board differential user clock through an MMCM, or a
  PS `PL_CLK` output. The design closed timing at 100 MHz with 0.37 ns slack;
  keep 100 MHz.
- **LEDs.** `jepasafe_top` drives `led[5:0]`. The ZCU104 does not have six PL
  user LEDs — count them in UG1267 and shrink the port. A missing pin
  assignment is a hard implementation error, which is a cheap way to find out.
- **PMOD I/O voltage.** The MPU6050 is a 3.3 V part. Verify the PMOD bank's
  VCCO before assigning the I2C pins, and before wiring anything to a real
  sensor later.

Deliverables: `utilization_placed.rpt`, `timing_summary.rpt`, and the `.bit`.
Expect them near the csynth estimate of 33,500 LUT / 54 DSP / 159 BRAM_18K —
though post-route numbers always move, and the standalone-PL wrapper adds logic
the HLS report never saw.

**Ship 16/16/16.** The 64/64/64 model is +1.18 points (92.09% vs 90.91%) but
measures **654 BRAM_18K = 104% of the device** and fails on block RAM alone.
Without the URAM work sketched in CLAUDE.md, 16/16/16 is the only configuration
measured to fit, and a plan with no board is not the moment to start a
storage-binding experiment.

---

## Stage 7 — The demo, without hardware

1. **Waveform walkthrough.** Record the Stage 5 simulation: samples entering,
   the 128-beat window streaming across TVALID/TREADY, `ap_done` firing, the
   class index appearing, the LED asserting. Annotate the 1.6 ms inference
   against the 2.56 s window to make the timing slack visible.
2. **Three-way agreement table.** One window, three columns: Keras float
   logits, csim fixed-point logits, cosim RTL logits. If Stages 2 and 3 pass,
   columns 2 and 3 are bit-identical and column 1 is within quantisation error.
   That single table is the whole deployment pipeline verified in one view.
3. **Accuracy table.** Float vs `ap_fixed<16,6>` vs `ap_fixed<8,4>` over 2,947
   windows, with the resource numbers beside each.

Present this as what it is — a verified pipeline with the final download step
missing — and it is more rigorous than a video of a blinking LED.

---

## What stays unproven

State this explicitly in the report rather than letting a reader assume
otherwise:

- **No real MPU6050.** I2C timing against physical silicon, 3.3 V levels, bus
  capacitance, sensor noise and offset calibration are all untested.
- **No real-world accuracy.** 90.91% is UCI HAR's 30 subjects with a
  waist-mounted phone. Nothing here measures accuracy at a different body
  position or with a different sensor.
- **No power measurement.** Requires the board's rails.
- **No fall detection, at all.** The model classifies six ordinary activities.
  UCI HAR has no fall, impact, or assault class, so JEPASafe cannot currently
  detect the event it is named for. This is a dataset gap (CLAUDE.md Gap 1),
  not a hardware one — the board's absence does not cause it and getting the
  board would not fix it. Closing it needs EdgeFall or a collected fall set
  plus a retrain. Say so in the motivation section instead of implying a
  working safety device.

---

## Order of work

Stages 0 → 2 → 3 are by far the best value: no new RTL, they close two
checklist items, and they produce the report's central result table. Stage 0 is
half a day and everything else depends on it.

| Priority | Stages | Needs | Closes |
|---|---|---|---|
| 1 | 0, 2, 3 | Python + Vitis HLS (both installed) | fixed-point accuracy; RTL-vs-C |
| 2 | 6 | Vivado + licence check + `.xdc` | "it fits and meets timing" |
| 3 | 4, 5 | the real RTL work | end-to-end chain |
| 4 | 7 | the above | the deliverable |
| optional | 1 | WSL env | faster precision sweeps |

Stage 6 is independent of Stages 4 and 5 for the core alone, so the licence
check and the `.xdc` can proceed in parallel with the RTL work.

Log results in `docs/build-log.md` as they land — per CLAUDE.md, every time
precision or layer widths change, record float vs fixed accuracy and the
LUT / DSP / BRAM numbers from `csynth.rpt`.
