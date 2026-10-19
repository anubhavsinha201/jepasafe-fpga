#!/usr/bin/env bash
# Run the Icarus Verilog testbenches.
#
#   bash scripts/sim.sh                       # every runnable testbench
#   bash scripts/sim.sh rtl/tb/tb_argmax6.v   # just one
#
# Waveforms, when a testbench dumps them, land in sim/*.vcd.
#
# tb_jepasafe_core is excluded from the default sweep: it drives the generated
# hls4ml core, which needs the zero-init copy and still does not simulate
# correctly under Icarus (Vitis RTL assumes FPGA power-up-zero). Run it
# deliberately with scripts/sim_rtl.sh. See docs/simulation-plan.md.

set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p sim
export PATH="/c/iverilog/bin:$PATH"

if ! command -v iverilog >/dev/null; then
  echo "iverilog not found. Install it:  winget install --id Icarus.Verilog" >&2
  exit 1
fi

EXCLUDE="tb_jepasafe_core"
fails=0

run_one() {
  local tb="$1"
  local name
  name="$(basename "${tb%.v}")"
  echo "=== $name ==="
  # -y rtl/tb as well as -y rtl: the I2C test needs the behavioural slave model.
  iverilog -g2012 -o "sim/$name.vvp" -y rtl -y rtl/tb -I rtl "$tb"
  # tee to the log AND the terminal, then grade from the log. Piping straight
  # into grep -q hides the testbench's own output, which is the part worth
  # reading when something fails.
  vvp "sim/$name.vvp" | tee "sim/$name.log"
  if grep -qE ': *FAIL' "sim/$name.log"; then
    fails=$((fails + 1))
  fi
  echo
}

if [ $# -gt 0 ]; then
  run_one "$1"
else
  for tb in rtl/tb/tb_*.v; do
    case "$(basename "$tb")" in
      $EXCLUDE.v) echo "=== skipping $(basename "$tb") (see sim_rtl.sh) ==="; echo; continue ;;
    esac
    run_one "$tb"
  done
fi

if [ "$fails" -gt 0 ]; then
  echo "$fails testbench(es) FAILED"
  exit 1
fi
echo "all testbenches passed"
