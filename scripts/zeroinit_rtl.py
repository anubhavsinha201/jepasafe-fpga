"""Make a simulation-only copy of the generated RTL with registers zero-initialised.

Why this exists: Vitis HLS RTL relies on FPGA power-up-zero semantics. Only 24
of the 66 generated modules carry a "power-on initialization" block; the rest
leave registers and memories undriven at time 0. Vivado's xsim models the
power-up-zero behaviour, so cosim works there. Icarus Verilog follows strict
Verilog X semantics instead, so that uninitialised state propagates X through
the datapath from the very first conv output and every logit comes out as X.

On this machine xelab.exe is blocked by Device Guard, so xsim is unavailable and
Icarus is the only simulator. This script bridges the gap by injecting an
`initial` block into a COPY of the RTL. Zero-init is what the hardware actually
does at power-up, so the patched copy is closer to silicon than X propagation —
but it is still a simulation aid, which is why it never touches the synthesis
source.

    python scripts/zeroinit_rtl.py
    bash scripts/sim_rtl.sh

Two things it must not break, both of which would fail quietly:

  1. Registers an existing power-on block sets to a NON-zero value. The clearest
     example is `ap_CS_fsm = 2'd1`: zeroing that starts the state machine in the
     wrong state. Any name already assigned in an existing initial block is left
     alone.
  2. The 13 weight/activation ROMs, which load via $readmemh in their own
     initial block. Verilog does not order initial blocks, so zeroing those
     arrays could race with the load and silently blank the weights — which
     looks exactly like a badly trained model.
"""

from __future__ import annotations

import argparse
import re
import shutil
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
SRC = REPO / "hls/jepasafe/jepasafe_prj/solution1/syn/verilog"
DST = REPO / "sim/jepasafe_rtl/rtl_zeroinit"

# reg [signed] [range] name [array_range] ;
# `reg\b` rather than `reg[ \t]+`: Vitis emits both "reg [15:0] x;" and
# "reg[15:0] x;" with no space. Requiring whitespace silently skipped the conv
# line-buffer registers, which is precisely where X entered the datapath. \b
# still matches before '[' while refusing to match a name like "regfoo".
REG_RE = re.compile(
    r"^[ \t]*reg\b[ \t]*(?:signed\b[ \t]*)?"   # reg, optionally signed
    r"(?:\[[^\]]+\][ \t]*)?"                   # optional bit range
    r"([A-Za-z_][A-Za-z0-9_$]*)"               # name
    r"[ \t]*(\[[^\]]+\])?"                     # optional array range
    r"[ \t]*;",
    re.MULTILINE,
)
READMEMH_RE = re.compile(r"\$readmem[hb]\s*\([^,]+,\s*([A-Za-z_][A-Za-z0-9_$]*)")
# Names assigned inside any initial block, e.g. "#0 ap_CS_fsm = 2'd1;"
INIT_ASSIGN_RE = re.compile(
    r"^[ \t]*(?:#\d+[ \t]*)?([A-Za-z_][A-Za-z0-9_$]*)[ \t]*(?:\[[^\]]*\])?[ \t]*(?:<?=)",
    re.MULTILINE,
)
ARRAY_RANGE_RE = re.compile(r"^\[\s*0\s*:\s*(.+?)\s*\]$")


def initial_block_names(text: str) -> set[str]:
    """Names assigned in any `initial begin ... end` block."""
    names: set[str] = set()
    for m in re.finditer(r"\binitial\b\s*begin(.*?)\bend\b", text, re.DOTALL):
        names.update(INIT_ASSIGN_RE.findall(m.group(1)))
    # Also catch single-statement `initial x = ...;`
    for m in re.finditer(r"\binitial\b\s+([A-Za-z_][A-Za-z0-9_$]*)\s*=", text):
        names.add(m.group(1))
    return names


MODULE_RE = re.compile(r"\bmodule\b.*?\bendmodule\b", re.DOTALL)


def patch(text: str) -> tuple[str, int, int, list[str]]:
    """Patch every module in the file independently.

    Several generated files hold more than one module — the FIFOs each pair a
    wrapper with its `_ShiftReg`. Collecting names across the whole file and
    injecting once before the last `endmodule` puts module A's registers inside
    module B, which Icarus rejects with "Could not find variable".
    """
    out: list[str] = []
    pos = 0
    tot_s = tot_a = 0
    skipped: list[str] = []

    for m in MODULE_RE.finditer(text):
        out.append(text[pos:m.start()])
        body, n_s, n_a, sk = patch_module(m.group(0))
        out.append(body)
        tot_s += n_s
        tot_a += n_a
        skipped += sk
        pos = m.end()
    out.append(text[pos:])
    return "".join(out), tot_s, tot_a, skipped


def patch_module(text: str) -> tuple[str, int, int, list[str]]:
    """Return (patched module text, n_scalars, n_arrays, skipped array names)."""
    protected = initial_block_names(text) | set(READMEMH_RE.findall(text))

    scalars: list[str] = []
    arrays: list[tuple[str, str]] = []
    skipped: list[str] = []

    for name, array_range in REG_RE.findall(text):
        if name in protected:
            continue
        if not array_range:
            scalars.append(name)
            continue
        m = ARRAY_RANGE_RE.match(array_range.strip())
        if m:
            arrays.append((name, m.group(1)))
        else:
            # Not a [0:N] memory. Rather than guess the iteration order, leave
            # it and say so — a wrong loop bound is worse than no init.
            skipped.append(f"{name}{array_range}")

    if not scalars and not arrays:
        return text, 0, 0, skipped

    lines = [
        "",
        "// ---- simulation-only zero-init, injected by scripts/zeroinit_rtl.py ----",
        "// Vitis RTL assumes FPGA power-up-zero. Icarus propagates X instead.",
        "// Not present in the synthesised design; do not edit this copy by hand.",
    ]
    if arrays:
        lines.append("integer zeroinit_idx;")
    lines.append("initial begin")
    for name in scalars:
        lines.append(f"    {name} = 0;")
    for name, hi in arrays:
        lines.append(
            f"    for (zeroinit_idx = 0; zeroinit_idx <= ({hi}); "
            f"zeroinit_idx = zeroinit_idx + 1) {name}[zeroinit_idx] = 0;"
        )
    lines.append("end")
    lines.append("// ---- end zero-init ----")
    lines.append("")

    block = "\n".join(lines)
    # Append before the module's final endmodule.
    idx = text.rstrip().rfind("endmodule")
    if idx < 0:
        return text, 0, 0, skipped
    return text[:idx] + block + "\n" + text[idx:], len(scalars), len(arrays), skipped


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--src", type=Path, default=SRC)
    p.add_argument("--dst", type=Path, default=DST)
    args = p.parse_args()

    if not args.src.is_dir():
        raise SystemExit(f"{args.src} not found — run csynth first.")

    if args.dst.exists():
        shutil.rmtree(args.dst)
    args.dst.mkdir(parents=True)

    # The weight ROMs are read with relative paths, so they must sit beside the
    # patched sources as well as in vvp's cwd.
    n_dat = 0
    for f in args.src.glob("*.dat"):
        shutil.copy2(f, args.dst / f.name)
        n_dat += 1

    total_s = total_a = 0
    patched_files = 0
    all_skipped: list[str] = []
    for f in sorted(args.src.glob("*.v")):
        text = f.read_text()
        out, n_s, n_a, skipped = patch(text)
        (args.dst / f.name).write_text(out)
        if n_s or n_a:
            patched_files += 1
        total_s += n_s
        total_a += n_a
        all_skipped += [f"{f.name}: {s}" for s in skipped]

    print(f"{args.dst.relative_to(REPO)}")
    print(f"  {patched_files} of {len(list(args.src.glob('*.v')))} modules patched")
    print(f"  {total_s} scalar regs, {total_a} memories zero-initialised")
    print(f"  {n_dat} weight ROM .dat files copied")
    if all_skipped:
        print(f"  {len(all_skipped)} array(s) left alone (not [0:N]):")
        for s in all_skipped[:10]:
            print(f"    {s}")
    print("\nNext: bash scripts/sim_rtl.sh")


if __name__ == "__main__":
    main()
