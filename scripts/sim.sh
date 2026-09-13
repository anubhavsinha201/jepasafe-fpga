#!/usr/bin/env bash
# Compile and run a testbench with Icarus Verilog.
#
#   bash scripts/sim.sh rtl/tb/tb_led_blinker.v
#   bash scripts/sim.sh                          # runs every tb in rtl/tb/
#
# Waveforms land in sim/*.vcd — open with: gtkwave sim/tb_led_blinker.vcd

set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p sim

run_one() {
  local tb="$1"
  local name
  name="$(basename "${tb%.v}")"
  echo "=== $name ==="
  iverilog -g2012 -o "sim/$name.vvp" -y rtl -I rtl "$tb"
  vvp "sim/$name.vvp"
  echo
}

if [ $# -gt 0 ]; then
  run_one "$1"
else
  for tb in rtl/tb/tb_*.v; do
    run_one "$tb"
  done
fi
