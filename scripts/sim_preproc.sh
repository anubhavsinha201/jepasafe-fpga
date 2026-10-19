#!/usr/bin/env bash
# Preprocessing-chain check: sensor counts -> sample_scale -> window_buffer,
# against the exact beats the hls4ml core expects.
#
#   bash scripts/sim_preproc.sh
#
# Regenerates both vector files first, so it is safe to run from a clean tree.

set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH="${PYTHONPATH:-}:src"

python -m jepasafe.make_tb_data --out-dir sim/jepasafe_rtl counts --window "${1:-0}"
python scripts/tv_to_hex.py
bash scripts/sim.sh rtl/tb/tb_preproc_chain.v
