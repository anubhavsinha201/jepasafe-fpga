#!/usr/bin/env bash
# Synthesise + place & route -> bitstream (.fs).
#
# Two supported toolchains:
#
#   Gowin EDA (recommended first)
#     GUI: open the project, add rtl/*.v + hls/jepasafe/**/*.v, set
#     constraints/tang_nano_9k.cst, then Synthesize -> Place & Route.
#     CLI: gw_sh scripts/build.tcl   (write build.tcl when you switch to CLI)
#
#   Yosys + nextpnr-himbaechel (open source) — what this script drives.
#     Install: yosys, nextpnr-himbaechel with the gowin uarch, gowin_pack
#
# Override the top module:  TOP=led_blinker bash scripts/build.sh

set -euo pipefail
cd "$(dirname "$0")/.."

TOP="${TOP:-jepasafe_top}"
DEVICE="GW1NR-LV9QN88PC6/I5"
FAMILY="GW1N-9C"
BUILD_DIR="impl"

mkdir -p "$BUILD_DIR"

SOURCES=(rtl/*.v)
# Pull in hls4ml RTL once it exists
if compgen -G "hls/jepasafe/**/*.v" > /dev/null; then
  SOURCES+=(hls/jepasafe/**/*.v)
fi

echo "=== yosys: synthesis (top=$TOP) ==="
yosys -p "read_verilog ${SOURCES[*]}; synth_gowin -top $TOP -json $BUILD_DIR/$TOP.json"

echo "=== nextpnr: place & route ==="
nextpnr-himbaechel \
  --json "$BUILD_DIR/$TOP.json" \
  --write "$BUILD_DIR/$TOP.pnr.json" \
  --device "$DEVICE" \
  --vopt family="$FAMILY" \
  --vopt cst=constraints/tang_nano_9k.cst

echo "=== gowin_pack: bitstream ==="
gowin_pack -d "$FAMILY" -o "$BUILD_DIR/$TOP.fs" "$BUILD_DIR/$TOP.pnr.json"

echo "bitstream -> $BUILD_DIR/$TOP.fs"
echo "Check the nextpnr utilisation report against the budget: 8,640 LUT4 / 20 DSP."
