# JEPASafe

A Transformer JEPA-based FPGA accelerator for real-time women's safety.
Full design rationale, hardware notes, timeline and references live in [CLAUDE.md](CLAUDE.md).

**Pipeline:** Train → Quantize → hls4ml → HLS → Gowin → Tang Nano 9K

## Layout

```
.
├── CLAUDE.md            # Project documentation & research guide (the source of truth)
├── data/
│   ├── raw/             # Downloaded datasets, untouched (gitignored)
│   └── processed/       # Windowed / normalised arrays ready for training
├── models/
│   ├── checkpoints/     # Float training checkpoints
│   ├── quantized/       # QKeras-quantized models
│   └── exported/        # Weights dumped for HLS / RTL
├── notebooks/           # Exploration, dataset EDA, threshold tuning
├── src/jepasafe/        # Python: data, model, train, quantize, hls export
├── hls/                 # hls4ml output projects (generated, gitignored)
├── rtl/                 # Verilog sources
│   └── tb/              # Testbenches (Icarus Verilog)
├── constraints/         # Tang Nano 9K pin constraints (.cst)
├── sim/                 # Simulation outputs, VCD waveforms (gitignored)
├── scripts/             # sim / build / program helpers
└── docs/build-log.md    # Weekly build log
```

## Quickstart

```bash
# 1. Python environment
python -m venv fpga_project
fpga_project\Scripts\activate        # Windows
# source fpga_project/bin/activate   # Linux/macOS
pip install -r requirements.txt

# 2. Train the float model
python -m jepasafe.train --data data/processed --epochs 50

# 3. Quantize
python -m jepasafe.quantize --model models/checkpoints/jepasafe.keras

# 4. Convert to HLS
python -m jepasafe.export_hls --model models/quantized/jepasafe_q.keras

# 5. Simulate the RTL
bash scripts/sim.sh rtl/tb/tb_led_blinker.v

# 6. Build + program the board
bash scripts/build.sh
bash scripts/program.sh
```

Set `PYTHONPATH=src` (or `pip install -e .`) so `python -m jepasafe.*` resolves.

## Status

Scaffold only. Modules marked `TODO` in `rtl/` and `src/` are stubs with settled
interfaces but no implementation yet — see [docs/build-log.md](docs/build-log.md)
for what has actually been built and verified.
