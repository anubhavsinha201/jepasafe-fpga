# JEPASafe Build Log

Template from CLAUDE.md. Fill in as you go — challenges and solutions are the
part that matters for the report, so write them down while they're fresh.

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
