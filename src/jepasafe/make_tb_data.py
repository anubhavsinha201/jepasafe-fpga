"""Generate and score the Vitis HLS testbench vectors.

``hls/jepasafe/tb_data/`` ships empty, so ``jepasafe_test.cpp`` falls through to
its ``fill_zero`` branch and csim/cosim run on an all-zero window. That makes
the "Verilog | Pass" in ``sim/report/jepasafe_cosim.rpt`` a liveness check and
nothing more. This module writes the real vectors, and reads back what the
simulators produce.

    python -m jepasafe.make_tb_data make                  # all 2,947 test windows
    python -m jepasafe.make_tb_data make --num-vectors 60 # stratified, for cosim
    python -m jepasafe.make_tb_data compare hls/jepasafe/tb_data/csim_results.log
    python -m jepasafe.make_tb_data selfcheck             # no HLS tools needed

``make`` writes three files to ``data/processed/tb/``:

    tb_input_features.dat     one window per line, 768 space-separated floats
    tb_output_predictions.dat one line per window, the 6 float Keras logits
    tb_index.npz              the chosen test-set indices + their labels

The first two are what hls4ml's writer copies into the HLS project's
``tb_data/`` — see ``export_hls.py``, which points ``InputData`` and
``OutputPredictions`` at them. The third is not used by HLS at all; it exists so
``compare`` can line a results log back up against ground truth when the
vectors are a subset.

Flattening order matters. The testbench reads each line into a flat vector and
hands it to ``nnet::copy_data<float, input_t, 0, 128*6>``, which fills the
stream element by element, so the line has to be time-major:
``t0c0 t0c1 ... t0c5 t1c0 ...``. That is exactly C-order ``ravel()`` of a
``(128, 6)`` window — ``selfcheck`` asserts it rather than trusting it.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from .cnnlstm_keras import CLASS_NAMES, N_CLASSES, WINDOW, load
from .config import CHECKPOINTS, DATA_PROCESSED

TB_DIR = DATA_PROCESSED / "tb"
N_CHANNELS = 6

INPUT_DAT = "tb_input_features.dat"
OUTPUT_DAT = "tb_output_predictions.dat"
INDEX_NPZ = "tb_index.npz"


def stratified_indices(y: np.ndarray, n: int) -> np.ndarray:
    """``n`` indices spread as evenly as the class counts allow, in test order.

    cosim simulates ~160k cycles per window, so it runs on a subset — and a
    subset that misses a class cannot show that class ever decoding correctly.
    Returned sorted, so a results log stays aligned with the original test
    order and is easy to eyeball against ``y``.
    """
    if n >= len(y):
        return np.arange(len(y))

    per_class, extra = divmod(n, N_CLASSES)
    picked: list[np.ndarray] = []
    for cls in range(N_CLASSES):
        members = np.flatnonzero(y == cls)
        take = per_class + (1 if cls < extra else 0)
        # Spread across the class rather than taking members[:take]. UCI HAR is
        # ordered by subject, so the leading windows of each class come from the
        # same few people and score several points below the full test set —
        # a subset accuracy that looks like a regression when it is just a
        # biased sample. Evenly spaced costs the same and stays deterministic.
        step = max(1, len(members) // take)
        picked.append(members[::step][:take])
    return np.sort(np.concatenate(picked))


def make(num_vectors: int | None, model_path: Path, out_dir: Path) -> None:
    from tensorflow import keras  # local: selfcheck and compare do not need TF

    x, y = load("test")
    assert x.shape[1:] == (WINDOW, N_CHANNELS), x.shape

    idx = np.arange(len(y)) if num_vectors is None else stratified_indices(y, num_vectors)
    x, y = x[idx], y[idx]

    model = keras.models.load_model(model_path)
    logits = model.predict(x, batch_size=256, verbose=0)
    assert logits.shape == (len(idx), N_CLASSES), logits.shape

    out_dir.mkdir(parents=True, exist_ok=True)
    # %.8g round-trips float32 and keeps the file readable. The testbench parses
    # with atof(), so precision beyond float32 would be thrown away anyway.
    np.savetxt(out_dir / INPUT_DAT, x.reshape(len(idx), -1), fmt="%.8g")
    np.savetxt(out_dir / OUTPUT_DAT, logits, fmt="%.8g")
    np.savez(out_dir / INDEX_NPZ, indices=idx, y=y)

    acc = (logits.argmax(1) == y).mean()
    print(f"{len(idx)} vectors -> {out_dir}")
    print(f"  {INPUT_DAT:26s} {len(idx)} x {x.shape[1] * x.shape[2]} floats")
    print(f"  {OUTPUT_DAT:26s} {len(idx)} x {N_CLASSES} float logits")
    print(f"  float accuracy on these vectors: {acc:.4%}")
    print(f"  class counts: {dict(zip(CLASS_NAMES, np.bincount(y, minlength=N_CLASSES)))}")
    print("\nNext: re-export so hls4ml copies these into the project's tb_data/")
    print("  python -m jepasafe.export_hls")


def counts(window: int, out_dir: Path) -> None:
    """Emit one window as raw MPU6050 int16 counts, for the RTL preprocessing test.

    The inverse of sample_scale.v: turn the physical units the model trains on
    back into the register values a real sensor would have produced, so the
    testbench exercises the scaling path instead of bypassing it.

        accel  count = g     * 8192                (+/-4 g)
        gyro   count = rad/s * 65.5 * 180/pi       (+/-500 deg/s)

    Round-tripping counts -> ap_fixed<16,6> rounds twice, so a beat can land one
    LSB (2^-10) away from the directly-converted value. That is a property of
    the sensor's quantisation, not an RTL bug, which is why the testbench allows
    1 LSB and reports how many beats matched exactly.
    """
    x, _ = load("test")
    w = x[window]
    acc = np.clip(np.rint(w[:, 0:3] * 8192.0), -32768, 32767).astype(np.int64)
    gyro = np.clip(np.rint(w[:, 3:6] * 65.5 * 180.0 / np.pi), -32768, 32767).astype(np.int64)

    out_dir.mkdir(parents=True, exist_ok=True)
    # One line per beat: 6 counts as 16-bit hex, channel 0 first, so the
    # testbench can $readmemh straight into a packed 96-bit word.
    lines = []
    for t in range(w.shape[0]):
        vals = list(acc[t]) + list(gyro[t])
        lines.append("".join(f"{v & 0xFFFF:04x}" for v in reversed(vals)))
    (out_dir / "counts_window.hex").write_text("\n".join(lines) + "\n")

    print(f"window {window} -> {out_dir / 'counts_window.hex'}  ({w.shape[0]} beats)")
    print(f"  accel count range {acc.min()}..{acc.max()}  gyro {gyro.min()}..{gyro.max()}")
    print(f"  beat 0 counts: {list(acc[0]) + list(gyro[0])}")
    print(f"  beat 0 physical: {np.round(w[0], 5).tolist()}")


def read_log(path: Path) -> np.ndarray:
    """Parse a csim / cosim results log into (n, 6) logits."""
    rows = [
        [float(v) for v in line.split()]
        for line in path.read_text().splitlines()
        if line.strip()
    ]
    if not rows:
        raise SystemExit(f"{path} is empty — did the simulation actually run?")
    widths = {len(r) for r in rows}
    if widths != {N_CLASSES}:
        raise SystemExit(f"{path}: expected {N_CLASSES} values per line, saw widths {widths}")
    return np.array(rows)


def compare(log_path: Path, out_dir: Path, against: Path | None) -> None:
    fixed = read_log(log_path)

    index_file = out_dir / INDEX_NPZ
    if not index_file.exists():
        raise SystemExit(f"{index_file} missing — run `make` before `compare`.")
    meta = np.load(index_file)
    y = meta["y"]

    if len(fixed) != len(y):
        raise SystemExit(
            f"{log_path} has {len(fixed)} rows but {index_file} describes {len(y)} vectors.\n"
            f"The log is stale, or it was produced from a different --num-vectors."
        )

    float_logits = np.loadtxt(out_dir / OUTPUT_DAT, ndmin=2)
    fixed_pred, float_pred = fixed.argmax(1), float_logits.argmax(1)

    print(f"{log_path}  ({len(y)} vectors)\n")
    print(f"  float accuracy          {(float_pred == y).mean():.4%}")
    print(f"  fixed-point accuracy    {(fixed_pred == y).mean():.4%}")
    print(f"  argmax agreement        {(fixed_pred == float_pred).mean():.4%}")
    print(f"  max abs logit error     {np.abs(fixed - float_logits).max():.4g}")

    flips = np.flatnonzero(fixed_pred != float_pred)
    if flips.size:
        print(f"\n  {flips.size} window(s) where quantisation changed the class:")
        for i in flips[:10]:
            print(
                f"    vector {i:5d}  true {CLASS_NAMES[y[i]]:18s} "
                f"float {CLASS_NAMES[float_pred[i]]:18s} -> fixed {CLASS_NAMES[fixed_pred[i]]}"
            )
        if flips.size > 10:
            print(f"    ... and {flips.size - 10} more")

    print("\n  confusion (rows = true, cols = fixed-point predicted)")
    cm = np.zeros((N_CLASSES, N_CLASSES), dtype=int)
    for t, p in zip(y, fixed_pred):
        cm[t, p] += 1
    print("    " + " ".join(f"{n[:6]:>6s}" for n in CLASS_NAMES))
    for name, row in zip(CLASS_NAMES, cm):
        print(f"    {' '.join(f'{v:6d}' for v in row)}  {name}")

    # The RTL and the C are the same fixed-point arithmetic, so "close" is a bug.
    if against is not None:
        other = read_log(against)
        if other.shape != fixed.shape:
            raise SystemExit(f"shape mismatch: {log_path} {fixed.shape} vs {against} {other.shape}")
        identical = np.array_equal(fixed, other)
        print(f"\n  vs {against}: {'IDENTICAL' if identical else 'DIFFERENT'}")
        if not identical:
            bad = np.flatnonzero(np.any(fixed != other, axis=1))
            print(f"    {bad.size} differing row(s), first at vector {bad[0]}")
            print(f"      {log_path.name}: {fixed[bad[0]]}")
            print(f"      {against.name}: {other[bad[0]]}")
            raise SystemExit(1)


def selfcheck() -> None:
    """Pin the two things that silently corrupt every downstream number."""
    # 1. Flatten order must be time-major to match copy_data's element fill.
    w = np.arange(WINDOW * N_CHANNELS, dtype=np.float32).reshape(WINDOW, N_CHANNELS)
    flat = w.reshape(-1)
    assert flat[0] == w[0, 0] and flat[1] == w[0, 1], "not channel-contiguous"
    assert flat[N_CHANNELS] == w[1, 0], "second beat must start at index 6"
    assert np.array_equal(flat.reshape(WINDOW, N_CHANNELS), w), "round-trip failed"

    # 2. A stratified subset must cover every class, or cosim proves nothing
    #    about the classes it skipped.
    y = np.repeat(np.arange(N_CLASSES), 100)
    idx = stratified_indices(y, 60)
    assert len(idx) == 60, len(idx)
    assert np.array_equal(np.sort(idx), idx), "indices must stay in test order"
    counts = np.bincount(y[idx], minlength=N_CLASSES)
    assert counts.min() >= 1, f"a class was dropped: {counts}"
    assert counts.tolist() == [10] * N_CLASSES, counts.tolist()
    # Uneven request: still every class, still the right total.
    odd = stratified_indices(y, 13)
    assert len(odd) == 13 and np.bincount(y[odd], minlength=N_CLASSES).min() >= 1
    # Asking for more than exists returns everything, not a crash.
    assert len(stratified_indices(y, 10**6)) == len(y)

    # 3. read_log rejects the shapes that would misalign a comparison.
    import tempfile

    tmp = Path(tempfile.mkdtemp()) / "r.log"
    tmp.write_text("1 2 3 4 5 6\n7 8 9 10 11 12\n")
    assert read_log(tmp).shape == (2, N_CLASSES)
    tmp.write_text("1 2 3\n")
    try:
        read_log(tmp)
    except SystemExit:
        pass
    else:
        raise AssertionError("read_log accepted a 3-wide row")

    print("selfcheck ok | time-major flatten, stratified coverage, log parsing")


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out-dir", type=Path, default=TB_DIR)
    sub = p.add_subparsers(dest="cmd", required=True)

    m = sub.add_parser("make", help="write the .dat vectors")
    m.add_argument("--num-vectors", type=int, default=None,
                   help="stratified subset size; default = the whole test set")
    m.add_argument("--model", type=Path, default=CHECKPOINTS / "cnnlstm_keras.keras")

    c = sub.add_parser("compare", help="score a csim/cosim results log")
    c.add_argument("log", type=Path)
    c.add_argument("--against", type=Path, default=None,
                   help="a second log that must be bit-identical (csim vs cosim)")

    n = sub.add_parser("counts", help="one window as raw MPU6050 int16 counts")
    n.add_argument("--window", type=int, default=0)

    sub.add_parser("selfcheck", help="no data, model, or HLS tools needed")

    args = p.parse_args(argv)
    if args.cmd == "make":
        make(args.num_vectors, args.model, args.out_dir)
    elif args.cmd == "compare":
        compare(args.log, args.out_dir, args.against)
    elif args.cmd == "counts":
        counts(args.window, args.out_dir)
    else:
        selfcheck()


if __name__ == "__main__":
    main()
