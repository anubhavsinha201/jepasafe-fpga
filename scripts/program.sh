#!/usr/bin/env bash
# Load a bitstream onto the Tang Nano 9K over USB-C (onboard BL702 USB-JTAG).
#
#   bash scripts/program.sh              # SRAM, volatile - lost on power cycle
#   bash scripts/program.sh --flash      # SPI flash, survives reboot
#
# Linux: needs the openFPGALoader udev rules, or run with sudo.
# Windows: install the WinUSB driver for the board with Zadig first.

set -euo pipefail
cd "$(dirname "$0")/.."

TOP="${TOP:-jepasafe_top}"
BITSTREAM="${BITSTREAM:-impl/$TOP.fs}"

if [ ! -f "$BITSTREAM" ]; then
  echo "no bitstream at $BITSTREAM — run scripts/build.sh first" >&2
  exit 1
fi

if [ "${1:-}" = "--flash" ]; then
  echo "writing $BITSTREAM to SPI flash..."
  openFPGALoader -b tangnano9k -f "$BITSTREAM"
else
  echo "writing $BITSTREAM to SRAM..."
  openFPGALoader -b tangnano9k "$BITSTREAM"
fi
