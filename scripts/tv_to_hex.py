"""Convert hls4ml/Vitis transaction vectors into flat $readmemh files.

Device Guard blocks `xelab.exe` on this machine, so xsim cosim cannot run (see
docs/simulation-plan.md). Icarus Verilog can elaborate the generated core
instead — it is plain Verilog-2001 with no vendor primitives — but hls4ml's own
cosim harness cannot port to it, because that harness is SystemVerilog and
depends on Xilinx simulation libraries.

What *does* port is the data. During the aborted cosim, the instrumented C
testbench wrote the exact AXI-Stream payloads it would have driven, and the
exact outputs the C model produced, to:

    sim/tv/cdatafile/c.jepasafe.autotvin_imu_window.dat
    sim/tv/cdatafile/c.jepasafe.autotvout_layer14_out.dat

Those are ground truth straight from the C model, in the RTL's own bit layout,
so using them sidesteps every question about how 6 channels pack into 96 bits.
Feed the inputs to the core under Icarus, compare the outputs against the
expected file, and you have the RTL-vs-C check that cosim would have given.

    python scripts/tv_to_hex.py            # writes into sim/jepasafe_rtl/

Format of the source files:

    [[[runtime]]]
    [[transaction]]  <n>
    0x<hex>                 <- 128 of these for the input, 1 for the output
    [[/transaction]]

Output: two files with no markers, one hex word per line, for $readmemh.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
TV = REPO / "hls/jepasafe/jepasafe_prj/solution1/sim/tv/cdatafile"
OUT = REPO / "sim/jepasafe_rtl"

BEATS_PER_WINDOW = 128
IN_NIBBLES = 96 // 4  # 96-bit TDATA  = 6 channels x ap_fixed<16,6>
OUT_NIBBLES = 384 // 4  # 384-bit TDATA = 6 logits x ap_fixed<37,17> in 64-bit lanes


def parse(path: Path) -> list[list[str]]:
    """Return one list of hex words per transaction, markers stripped."""
    transactions: list[list[str]] = []
    current: list[str] | None = None
    for line in path.read_text().splitlines():
        line = line.strip()
        if re.match(r"^\[\[transaction\]\]", line):
            current = []
        elif line == "[[/transaction]]":
            if current is not None:
                transactions.append(current)
            current = None
        elif line.startswith("0x") and current is not None:
            current.append(line[2:])
    return transactions


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--tv-dir", type=Path, default=TV)
    p.add_argument("--out-dir", type=Path, default=OUT)
    args = p.parse_args()

    src_in = args.tv_dir / "c.jepasafe.autotvin_imu_window.dat"
    src_out = args.tv_dir / "c.jepasafe.autotvout_layer14_out.dat"
    for f in (src_in, src_out):
        if not f.exists():
            raise SystemExit(
                f"{f} not found.\n"
                f"These are written by the C testbench during a cosim attempt. Run:\n"
                f"  cd hls/jepasafe && vitis_hls -f ../../scripts/cosim.tcl\n"
                f"It will fail at the xelab step, which is expected and fine — the\n"
                f"vectors are written before that."
            )

    stim, expected = parse(src_in), parse(src_out)

    if len(stim) != len(expected):
        raise SystemExit(f"{len(stim)} input transactions vs {len(expected)} output")
    for i, beats in enumerate(stim):
        if len(beats) != BEATS_PER_WINDOW:
            raise SystemExit(f"input transaction {i}: {len(beats)} beats, expected {BEATS_PER_WINDOW}")
    for i, words in enumerate(expected):
        if len(words) != 1:
            raise SystemExit(f"output transaction {i}: {len(words)} words, expected 1")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    # Zero-pad so every line is the full width: $readmemh takes the value either
    # way, but a ragged file hides a truncation bug behind plausible numbers.
    stim_path = args.out_dir / "stim_imu_window.hex"
    exp_path = args.out_dir / "expected_logits.hex"
    stim_path.write_text(
        "".join(f"{w.zfill(IN_NIBBLES)}\n" for beats in stim for w in beats)
    )
    exp_path.write_text(
        "".join(f"{words[0].zfill(OUT_NIBBLES)}\n" for words in expected)
    )

    print(f"{len(stim)} transactions")
    print(f"  {stim_path.relative_to(REPO)}  {len(stim) * BEATS_PER_WINDOW} beats x {IN_NIBBLES} nibbles")
    print(f"  {exp_path.relative_to(REPO)}  {len(expected)} words x {OUT_NIBBLES} nibbles")

    # Decode logit 0 of transaction 0 as a sanity print. Element 0 sits in the
    # LSB lane (the last 16 nibbles) and carries 20 fractional bits.
    lane0 = int(expected[0][0].zfill(OUT_NIBBLES)[-16:], 16)
    if lane0 >= 1 << 63:
        lane0 -= 1 << 64
    print(f"  sanity: transaction 0, logit 0 = {lane0 / (1 << 20):.5f}")
    print(f"\nNext: bash scripts/sim_rtl.sh")


if __name__ == "__main__":
    main()
