#!/usr/bin/env bash
# RTL-vs-C check on the hls4ml core, with Icarus Verilog.
#
#   bash scripts/sim_rtl.sh        # all 60 transactions
#   bash scripts/sim_rtl.sh 3      # first 3 only, to prove the flow quickly
#
# This replaces Vitis cosim, which cannot run here — Device Guard blocks
# xelab.exe. See docs/simulation-plan.md.
#
# Prerequisite: python scripts/tv_to_hex.py
#
# Two path details that bite:
#   - The core's weight ROMs are loaded with $readmemh("./name.dat", ...), a
#     relative path, so vvp must run with its cwd set to the directory holding
#     those 13 .dat files. A wrong cwd gives silently zero-filled ROMs, which
#     look exactly like a badly trained model rather than a setup error.
#   - The testbench reads its vectors as sim/jepasafe_rtl/*.hex, relative to the
#     repo root. Both cannot be satisfied by one cwd, so the .dat ROMs are
#     copied next to the vectors and everything runs from the repo root.

set -euo pipefail
cd "$(dirname "$0")/.."

NTX="${1:-60}"
CORE_RTL="hls/jepasafe/jepasafe_prj/solution1/syn/verilog"
WORK="sim/jepasafe_rtl"
export PATH="/c/iverilog/bin:$PATH"

if ! command -v iverilog >/dev/null; then
  echo "iverilog not found. Install it:  winget install --id Icarus.Verilog" >&2
  exit 1
fi
if [ ! -d "$CORE_RTL" ]; then
  echo "No generated RTL at $CORE_RTL — run csynth first." >&2
  exit 1
fi
if [ ! -f "$WORK/stim_imu_window.hex" ]; then
  echo "No vectors. Run: python scripts/tv_to_hex.py" >&2
  exit 1
fi

# The weight ROMs must sit in vvp's cwd, which is the repo root.
cp "$CORE_RTL"/*.dat . 2>/dev/null || true
trap 'cd "$(dirname "$0")/.."; rm -f ./*.dat' EXIT

echo "=== compiling core + testbench ==="
iverilog -g2012 -o "$WORK/tb_jepasafe_core.vvp" \
  -y "$CORE_RTL" -I "$CORE_RTL" \
  rtl/tb/tb_jepasafe_core.v "$CORE_RTL/jepasafe.v"

echo "=== running ($NTX transaction(s)) ==="
vvp "$WORK/tb_jepasafe_core.vvp" "+ntx=$NTX" | tee "$WORK/sim_rtl.log"

grep -q "RTL-vs-C: PASS" "$WORK/sim_rtl.log"
