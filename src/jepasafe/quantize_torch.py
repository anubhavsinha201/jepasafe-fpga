"""Per-channel fake quantization for the trained PyTorch UCI HAR model.

Answers two questions the FPGA path needs answered:

  1. What does 8 / 6 / 4-bit cost in accuracy, when every weight tensor is
     quantized *per output channel* rather than per tensor?
  2. What fixed-point format does each individual layer need? That table is
     what hls4ml's `granularity='name'` config consumes later.

    python -m jepasafe.quantize_torch              # train if needed, sweep 8/6/4
    python -m jepasafe.quantize_torch --bits 8     # one width
    python -m jepasafe.quantize_torch --selfcheck  # asserts only, no dataset

Everything hinges on `torch.nn.utils.parametrize`, which intercepts reads of
`module.weight`, keeps the float master weight as `.original`, and lets a
straight-through estimator pass gradients back to it. One mechanism therefore
serves both modes: PTQ is "register, then eval", QAT is "register, then train".

Scope, so the gaps are visible in the write-up:

  - Weights are quantized per output channel; activations per tensor, against a
    range calibrated on a handful of batches.
  - Biases stay float. Hardware accumulates in a wider format, so quantizing
    them would measure a constraint the FPGA does not have.
  - The LSTM's *output* is not activation-quantized (it returns a tuple, and it
    is not FPGA-bound anyway). Its two weight matrices are quantized.

This model is a PC-side benchmark, not a Tang Nano 9K candidate — see the
docstring in demopart.py. The precision table transfers to the deployable model
in model.py; these trained LSTM weights do not.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score
from torch import nn
from torch.nn.utils import parametrize

from .config import CHECKPOINTS, QUANTIZED
from .demopart import (
    DEFAULT_ROOT,
    MODELS,
    collect_predictions,
    load_split,
    make_loader,
    run,
    run_epoch,
)

BIT_WIDTHS = (8, 6, 4)

# Modules whose output is a plain tensor, so a hook can fake-quant it in place.
ACT_MODULES = (nn.Conv1d, nn.Linear, nn.ReLU)


# ------------------------------------------------------------------ quantizers


def _fake_quant(x: torch.Tensor, scale: torch.Tensor | float, bits: int) -> torch.Tensor:
    """Symmetric round-to-nearest onto `bits` levels, straight-through backward.

    The clamp range is asymmetric ([-2^(b-1), 2^(b-1)-1]) because two's
    complement has one more negative code than positive; `scale` is derived from
    the positive end, so the extra code is simply headroom.
    """
    qmax = 2 ** (bits - 1) - 1
    xq = (x / scale).round().clamp(-qmax - 1, qmax) * scale
    return x + (xq - x).detach()


class FakeQuant(nn.Module):
    """Per-channel symmetric weight fake-quant, as a parametrization.

    `dim` is the output-channel axis, which is 0 for every tensor here:
    Conv1d (out, in, k), Linear (out, in), LSTM weight_ih/hh (4*hidden, ...)
    where each row belongs to one gate of one unit.
    """

    def __init__(self, bits: int, dim: int = 0):
        super().__init__()
        self.bits = bits
        self.dim = dim

    def scale(self, w: torch.Tensor) -> torch.Tensor:
        dims = [d for d in range(w.dim()) if d != self.dim]
        amax = w.detach().abs().amax(dim=dims, keepdim=True)
        return (amax / (2 ** (self.bits - 1) - 1)).clamp(min=1e-12)

    def forward(self, w: torch.Tensor) -> torch.Tensor:
        return _fake_quant(w, self.scale(w), self.bits)


class ActQuant:
    """Forward hook: learn the output's range while calibrating, quantize after.

    The range is a high percentile of |output|, not its maximum. Absolute-max
    calibration lets one outlier activation set the scale for every value in the
    layer, and at 4 bits that was worth tens of accuracy points depending on
    which batches happened to be drawn — the measurement moved more than the
    thing being measured. Clipping the top 0.1% trades a few saturated outliers
    for resolution across the bulk of the distribution. Pass percentile=100 to
    get the old absolute-max behaviour back.

    Ranges are frozen once calibration ends, so QAT cannot chase a moving
    quantization grid.
    """

    def __init__(self, bits: int, percentile: float = 99.9):
        self.bits = bits
        self.percentile = percentile
        self.amax = 0.0
        self.calibrating = True

    def __call__(self, module, inputs, output):
        if self.calibrating:
            values = output.detach().abs().flatten()
            if self.percentile >= 100:
                observed = values.max().item()
            else:
                # torch.quantile refuses tensors past ~16M elements; a strided
                # subsample keeps it deterministic and is plenty for a percentile.
                values = values[:: max(1, values.numel() // 1_000_000)]
                observed = torch.quantile(values, self.percentile / 100).item()
            self.amax = max(self.amax, observed)
            return None  # leave the output alone
        scale = max(self.amax, 1e-12) / (2 ** (self.bits - 1) - 1)
        return _fake_quant(output, scale, self.bits)


# ------------------------------------------------------------- attach / detach


def quantized_tensors(model: nn.Module):
    """(module, attribute_name, qualified_name) for every weight we quantize."""
    for name, module in model.named_modules():
        if isinstance(module, (nn.Conv1d, nn.Linear)):
            yield module, "weight", f"{name}.weight"
        elif isinstance(module, nn.LSTM):
            for attr in ("weight_ih_l0", "weight_hh_l0"):
                yield module, attr, f"{name}.{attr}"


def attach(model: nn.Module, bits: int, percentile: float = 99.9) -> dict[str, ActQuant]:
    """Register weight parametrizations and activation hooks. Returns the hooks."""
    for module, attr, _ in quantized_tensors(model):
        parametrize.register_parametrization(module, attr, FakeQuant(bits))

    hooks: dict[str, ActQuant] = {}
    for name, module in model.named_modules():
        if isinstance(module, ACT_MODULES):
            quant = ActQuant(bits, percentile)
            module.register_forward_hook(quant)
            hooks[f"{name}.out" if name else "out"] = quant
    return hooks


def calibrate(model: nn.Module, loader, device, hooks: dict[str, ActQuant], batches: int) -> None:
    """Run a few batches to fill in the activation ranges, then freeze them."""
    model.eval()
    with torch.no_grad():
        for i, (x_batch, _) in enumerate(loader):
            if i >= batches:
                break
            model(x_batch.to(device))
    for quant in hooks.values():
        quant.calibrating = False


def ap_fixed(amax: float, bits: int) -> str:
    """Narrowest ap_fixed<bits, I> whose range covers [-amax, amax].

    ap_fixed<W, I> spans [-2^(I-1), 2^(I-1)) — I counts the integer bits
    *including* the sign. So I = floor(log2(amax)) + 2, not ceil(...) + 1:
    the two agree everywhere except when amax is an exact power of two, where
    ceil hands back a range whose open upper end clips the very value that
    set it (amax = 1.0 -> ap_fixed<8,1> = [-1, 1)).
    """
    int_bits = (math.floor(math.log2(amax)) + 2) if amax > 0 else 1
    return f"ap_fixed<{bits},{int_bits}>"


def precision_table(model: nn.Module, hooks: dict[str, ActQuant], bits: int) -> dict:
    """Per-layer fixed-point formats from the observed ranges.

    A layer asking for more than 4 integer bits is one that
    HLSConfig.default_precision (ap_fixed<8,4>) would clip.
    """
    fmt = lambda amax: ap_fixed(amax, bits)

    table = {"bits": bits, "weights": {}, "activations": {}}
    for module, attr, qualified in quantized_tensors(model):
        table["weights"][qualified] = fmt(getattr(module, attr).detach().abs().max().item())
    for name, quant in hooks.items():
        table["activations"][name] = fmt(quant.amax)
    return table


# ------------------------------------------------------------------- evaluate


def evaluate(model: nn.Module, loader, device) -> tuple[float, float]:
    y_true, y_pred = collect_predictions(model, loader, device)
    return accuracy_score(y_true, y_pred) * 100, f1_score(y_true, y_pred, average="weighted")


def finetune(model, train_loader, test_loader, device, epochs: int, lr: float, verbose: bool):
    """QAT: same fake-quant, gradients reach the float master weights via STE."""
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    best_acc, best_state = 0.0, None

    for epoch in range(1, epochs + 1):
        train_loss, train_acc = run_epoch(model, train_loader, criterion, device, optimizer)
        _, val_acc = run_epoch(model, test_loader, criterion, device)
        if val_acc > best_acc:
            best_acc, best_state = val_acc, copy.deepcopy(model.state_dict())
        if verbose:
            print(
                f"    QAT epoch {epoch}/{epochs}"
                f" - loss: {train_loss:.4f} - accuracy: {train_acc:.4f}"
                f" - val_accuracy: {val_acc:.4f}"
            )

    if best_state is not None:
        model.load_state_dict(best_state)


# ----------------------------------------------------------------------- main


def load_or_train(model: str, weights: Path, epochs: int, seed: int, device, verbose: bool):
    """Trained weights, training them first if the checkpoint is missing."""
    net = MODELS[model]().to(device)
    if not weights.exists():
        if verbose:
            print(f"no checkpoint at {weights} — training first ({epochs} epochs)")
        run(model=model, epochs=epochs, seed=seed, save=weights, verbose=verbose)
    net.load_state_dict(torch.load(weights, map_location=device))
    return net


def sweep(
    model: str = "cnnlstm",
    weights: Path | str | None = None,
    bits: tuple[int, ...] = BIT_WIDTHS,
    epochs: int = 30,
    batch_size: int = 64,
    calib_batches: int = 10,
    calib_percentile: float = 99.9,
    qat_epochs: int = 5,
    qat_threshold: float = 1.0,
    lr: float = 1e-4,
    seed: int = 42,
    out_dir: Path | str = QUANTIZED,
    verbose: bool = True,
) -> dict:
    """Quantize at each bit width, report accuracy, write the precision tables."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    weights = Path(weights) if weights else CHECKPOINTS / f"{model}.pt"
    out_dir = Path(out_dir)

    x_train, y_train = load_split(DEFAULT_ROOT, "train")
    x_test, y_test = load_split(DEFAULT_ROOT, "test")
    train_loader = make_loader(x_train, y_train, batch_size, shuffle=True)
    test_loader = make_loader(x_test, y_test, batch_size, shuffle=False)

    net = load_or_train(model, weights, epochs, seed, device, verbose)
    float_state = copy.deepcopy(net.state_dict())

    base_acc, base_f1 = evaluate(net, test_loader, device)
    if verbose:
        print(f"\nfloat32 baseline: accuracy {base_acc:.2f}%  F1 {base_f1:.4f}")

    rows = []
    for n_bits in bits:
        # Re-seed per width: train_loader shuffles, and max-based calibration is
        # sensitive enough at 4 bits that different batches move the result by
        # tens of points. Every width must see the same calibration data, and a
        # rerun must reproduce the number.
        torch.manual_seed(seed)

        # Fresh model per width: hooks live on the modules and parametrizations
        # rewrite the state_dict keys, so rebuilding is cleaner than unpicking.
        net = MODELS[model]().to(device)
        net.load_state_dict(float_state)
        hooks = attach(net, n_bits, calib_percentile)
        calibrate(net, train_loader, device, hooks, calib_batches)

        acc, f1 = evaluate(net, test_loader, device)
        drop = base_acc - acc
        if verbose:
            print(f"\n{n_bits}-bit PTQ : accuracy {acc:.2f}%  F1 {f1:.4f}  ({acc - base_acc:+.2f} pp)")

        qat_acc = qat_f1 = None
        if drop > qat_threshold:
            if verbose:
                print(f"  drop exceeds {qat_threshold} pp — fine-tuning {qat_epochs} epochs")
            finetune(net, train_loader, test_loader, device, qat_epochs, lr, verbose)
            qat_acc, qat_f1 = evaluate(net, test_loader, device)
            if verbose:
                print(
                    f"  {n_bits}-bit QAT : accuracy {qat_acc:.2f}%  F1 {qat_f1:.4f}"
                    f"  ({qat_acc - base_acc:+.2f} pp)"
                )

        table = precision_table(net, hooks, n_bits)
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"precision_{n_bits}bit.json"
        out_path.write_text(json.dumps(table, indent=2))

        rows.append(
            {
                "bits": n_bits,
                "ptq_accuracy": acc,
                "ptq_f1": f1,
                "qat_accuracy": qat_acc,
                "qat_f1": qat_f1,
                "precision": table,
                "path": out_path,
            }
        )

    if verbose:
        print(f"\n{'bits':>5}{'PTQ acc':>10}{'PTQ F1':>9}{'QAT acc':>10}{'vs float':>10}")
        print(f"{'32':>5}{base_acc:>9.2f}%{base_f1:>9.4f}{'-':>10}{'-':>10}")
        for row in rows:
            best = row["qat_accuracy"] if row["qat_accuracy"] is not None else row["ptq_accuracy"]
            qat = f"{row['qat_accuracy']:.2f}%" if row["qat_accuracy"] is not None else "-"
            print(
                f"{row['bits']:>5}{row['ptq_accuracy']:>9.2f}%{row['ptq_f1']:>9.4f}"
                f"{qat:>10}{best - base_acc:>+9.2f}pp"
            )
        print(f"\nprecision tables -> {out_dir}")

    return {"float_accuracy": base_acc, "float_f1": base_f1, "rows": rows}


# ------------------------------------------------------------------ selfcheck


def _selfcheck() -> None:
    """Asserts covering the claims this file rests on. No dataset needed."""
    torch.manual_seed(0)

    q8 = FakeQuant(8)

    # Quantizing an already-quantized tensor changes nothing.
    w = torch.randn(16, 8, 3)
    once = q8(w)
    assert torch.allclose(once, q8(once), atol=1e-6), "fake-quant is not idempotent"

    # Round-to-nearest error cannot exceed half a step.
    assert (once - w).abs().max() <= q8.scale(w).max() / 2 + 1e-6, "error exceeds scale/2"

    # The whole point of per-channel. With channel scales orders apart, one
    # shared scale quantizes the quiet channels to nothing; a scale per channel
    # keeps their *relative* precision. Mean absolute error would hide this —
    # it is dominated by the loud channel either way — so measure the quiet one.
    w = torch.randn(4, 32) * torch.tensor([1.0, 1e-2, 1e-3, 1e-4]).unsqueeze(1)
    quiet = w[3]
    rel = lambda wq: ((wq[3] - quiet).abs().mean() / quiet.abs().mean()).item()

    per_channel = rel(FakeQuant(8)(w))
    per_tensor = rel(_fake_quant(w, w.abs().max() / 127, 8))
    assert per_channel < 0.02, f"per-channel lost the quiet channel: {per_channel:.4f}"
    assert per_tensor > 0.9, f"per-tensor should have flattened it, got {per_tensor:.4f}"

    # ap_fixed must cover its own amax, especially at exact powers of two where
    # the open upper end of [-2^(I-1), 2^(I-1)) bites.
    for amax in (0.3, 0.9, 1.0, 2.0, 3.0, 8.0, 1e-4):
        int_bits = int(ap_fixed(amax, 8).split(",")[1].rstrip(">"))
        assert amax < 2 ** (int_bits - 1), f"ap_fixed clips {amax} at I={int_bits}"
        assert amax >= 2 ** (int_bits - 2), f"ap_fixed wastes a bit on {amax}"
    assert ap_fixed(1.0, 8) == "ap_fixed<8,2>", ap_fixed(1.0, 8)
    assert ap_fixed(0.0, 8) == "ap_fixed<8,1>", ap_fixed(0.0, 8)

    # Attaching and removing leaves the float weights bit-exact.
    net = MODELS["cnnlstm"]()
    before = {k: v.clone() for k, v in net.state_dict().items()}
    for module, attr, _ in quantized_tensors(net):
        parametrize.register_parametrization(module, attr, FakeQuant(8))
    for module, attr, _ in quantized_tensors(net):
        parametrize.remove_parametrizations(module, attr, leave_parametrized=False)
    after = net.state_dict()
    assert all(torch.equal(before[k], after[k]) for k in before), "removal changed the weights"

    # STE: gradients must reach the float master weight, through the LSTM too.
    lstm = nn.LSTM(4, 3, batch_first=True)
    parametrize.register_parametrization(lstm, "weight_ih_l0", FakeQuant(4))
    lstm(torch.randn(2, 5, 4))[0].sum().backward()
    grad = lstm.parametrizations.weight_ih_l0.original.grad
    assert grad is not None and grad.abs().sum() > 0, "STE gradient did not reach master weight"

    print("selfcheck: all assertions passed")


def main(argv: list[str] | None = None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", choices=sorted(MODELS), default="cnnlstm")
    p.add_argument("--weights", type=Path, default=None, help="default: models/checkpoints/<model>.pt")
    p.add_argument("--bits", type=int, nargs="+", default=list(BIT_WIDTHS))
    p.add_argument("--epochs", type=int, default=30, help="training epochs, if no checkpoint exists")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--calib-batches", type=int, default=10)
    p.add_argument("--calib-percentile", type=float, default=99.9,
                   help="activation range percentile; 100 = absolute max")
    p.add_argument("--qat-epochs", type=int, default=5)
    p.add_argument("--qat-threshold", type=float, default=1.0, help="pp drop that triggers QAT")
    p.add_argument("--lr", type=float, default=1e-4, help="QAT learning rate")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", type=Path, default=QUANTIZED)
    p.add_argument("--selfcheck", action="store_true", help="run assertions and exit")
    args = p.parse_args(argv)

    if args.selfcheck:
        return _selfcheck()

    return sweep(
        model=args.model,
        weights=args.weights,
        bits=tuple(args.bits),
        epochs=args.epochs,
        batch_size=args.batch_size,
        calib_batches=args.calib_batches,
        calib_percentile=args.calib_percentile,
        qat_epochs=args.qat_epochs,
        qat_threshold=args.qat_threshold,
        lr=args.lr,
        seed=args.seed,
        out_dir=args.out,
    )


if __name__ == "__main__":
    main()
