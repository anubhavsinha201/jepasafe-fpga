"""Keras CNN+LSTM — the version of demopart.py's model that reaches the FPGA.

Why a second copy of the same architecture: hls4ml 1.3.0's PyTorch parser has
no `transpose` handler, and ``CNNLSTMClassifier.forward`` needs two of them —
``nn.Conv1d`` is channels-first, ``nn.LSTM`` wants (batch, time, feature).
Keras ``Conv1D`` is channels-last already, so the identical architecture
converts with no transposes anywhere. hls4ml folds the BatchNorms into the
Conv1Ds and drops the Dropouts, leaving the same shape chain PyTorch produces:

    (128, 6) -> 124 -> 120 -> maxpool 60 -> lstm 16 -> dense 16 -> 6

    python -m jepasafe.cnnlstm_keras                 # train + save
    python -m jepasafe.cnnlstm_keras --selfcheck     # no data needed

The 16/16/16 defaults were shrunk from demopart's 64/64/64 to fit a Tang Nano 9K
(61.7k params = 62 KB of 8-bit weights against 58.5 KB of BSRAM). The target is
now a ZCU104 (XCZU7EV: 11 Mb BRAM + 27 Mb URAM), where that constraint does not
bind at all — pass `--filters 64 --hidden 64 --dense 64` for the full-size model.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from tensorflow import keras
from tensorflow.keras import layers

from .config import CHECKPOINTS, DATA_PROCESSED, TRAIN

CACHE = DATA_PROCESSED / "uci_har"

# The cache holds 9 channels in this order:
#   0-2 body_acc, 3-5 body_gyro, 6-8 total_acc
# Train on total_acc + body_gyro only: that is exactly what an MPU6050 hands
# you. body_acc is total_acc with gravity filtered out, which would mean
# reimplementing UCI HAR's Butterworth filter in RTL before inference.
CHANNEL_IDX = [6, 7, 8, 3, 4, 5]  # ax, ay, az, gx, gy, gz

WINDOW = 128
N_CLASSES = 6
CLASS_NAMES = [
    "WALKING",
    "WALKING_UPSTAIRS",
    "WALKING_DOWNSTAIRS",
    "SITTING",
    "STANDING",
    "LAYING",
]


def load(split: str, cache: Path = CACHE):
    """Load a cached split, narrowed to the 6 deployable channels.

    The cache is written by demopart.py; run that once if it is missing.
    Units are UCI HAR's: g for accelerometer, rad/s for gyro. The int16 ->
    physical-units scaling is a shift in RTL, so no normalisation happens
    here — keeping the model on raw physical units saves a preprocessing block.
    """
    x_path, y_path = cache / f"X_{split}.npy", cache / f"y_{split}.npy"
    if not x_path.exists():
        raise FileNotFoundError(
            f"{x_path} not found. Build the cache first:\n"
            f"  python -m jepasafe.demopart --epochs 1"
        )
    return np.load(x_path)[:, :, CHANNEL_IDX], np.load(y_path)


def build(filters=16, hidden=16, dense=16, dropout=0.3, window=WINDOW,
          channels=len(CHANNEL_IDX), n_classes=N_CLASSES) -> keras.Model:
    """Mirror of demopart.CNNLSTMClassifier, layer for layer.

    Convs carry no bias because the following BatchNorm's shift makes it
    redundant — and hls4ml folds that pair into one Conv1D on conversion.
    Output is logits; softmax stays off the FPGA (argmax gives the same class).
    """
    model = keras.Sequential(name="cnnlstm")
    model.add(keras.Input(shape=(window, channels), name="imu_window"))

    for _ in range(2):
        model.add(layers.Conv1D(filters, 5, use_bias=False))
        model.add(layers.BatchNormalization())
        # Activation('relu'), not layers.ReLU(): identical math (ReLU's default
        # threshold is 0), but hls4ml maps layers.ReLU to a *parametrized*
        # thresholdedrelu that carries a live threshold compare per element.
        # Plain relu is strictly less logic across 3,904 conv activations.
        model.add(layers.Activation("relu"))

    model.add(layers.MaxPooling1D(2))
    model.add(layers.Dropout(dropout))
    model.add(layers.LSTM(hidden))  # final hidden state only
    model.add(layers.Dropout(dropout))
    model.add(layers.Dense(dense, activation="relu"))
    model.add(layers.Dense(n_classes, name="logits"))
    return model


def train(model: keras.Model, epochs: int, out: Path) -> dict:
    x_train, y_train = load("train")
    x_test, y_test = load("test")

    model.compile(
        optimizer=keras.optimizers.Adam(TRAIN.learning_rate),
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=["accuracy"],
    )
    model.fit(
        x_train, y_train,
        epochs=epochs,
        batch_size=TRAIN.batch_size,
        validation_data=(x_test, y_test),
        verbose=2,
    )

    loss, acc = model.evaluate(x_test, y_test, verbose=0)
    out.parent.mkdir(parents=True, exist_ok=True)
    model.save(out)
    print(f"test accuracy {acc:.4f} | params {model.count_params()} | saved -> {out}")
    return {"accuracy": float(acc), "loss": float(loss), "params": model.count_params()}


def selfcheck() -> None:
    """Shapes, weight budget, and what hls4ml actually makes of the model."""
    model = build()
    assert model.input_shape == (None, WINDOW, 6), model.input_shape
    assert model.output_shape == (None, N_CLASSES), model.output_shape

    # Shape chain must match the PyTorch docstring in demopart.py
    shapes = [l.output.shape[1:] for l in model.layers]
    assert (124, 16) in shapes and (120, 16) in shapes and (60, 16) in shapes, shapes

    params = model.count_params()
    # 8-bit weights against the XCZU7EV's 11 Mb of BRAM (URAM not counted).
    # Was 26 x 18 Kbit = 58.5 KB when the target was a Tang Nano 9K, which is
    # what forced 16/16/16; the ZCU104 makes this check effectively a formality.
    bram_bytes = 11 * 1024 * 1024 // 8
    assert params < bram_bytes, f"{params} B of weights > {bram_bytes} B BRAM"

    import tempfile

    import hls4ml  # imported here so training does not need it

    cfg = hls4ml.utils.config_from_keras_model(
        model, granularity="name", default_precision="ap_fixed<16,6>", default_reuse_factor=16
    )
    hls_model = hls4ml.converters.convert_from_keras_model(
        model, hls_config=cfg, output_dir=tempfile.mkdtemp(prefix="jepasafe_selfcheck"),
        project_name="jepasafe", backend="Vitis", io_type="io_parallel",
    )
    seen = [l.class_name for l in hls_model.get_layers()]
    # BatchNorm folded into Conv1D, Dropout dropped -> neither should survive
    assert "BatchNormalization" not in seen, seen
    assert "Dropout" not in seen, seen
    # layers.ReLU would show up as ParametrizedActivation (thresholdedrelu)
    assert "ParametrizedActivation" not in seen, seen
    assert seen.count("Conv1D") == 2 and "LSTM" in seen, seen

    print(f"selfcheck ok | params {params} ({params} B at 8 bit, budget {bram_bytes} B)")
    print(f"hls4ml graph: {' -> '.join(seen)}")


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--filters", type=int, default=16)
    p.add_argument("--hidden", type=int, default=16)
    p.add_argument("--dense", type=int, default=16)
    p.add_argument("--epochs", type=int, default=TRAIN.epochs)
    p.add_argument("--out", type=Path, default=CHECKPOINTS / "cnnlstm_keras.keras")
    p.add_argument("--selfcheck", action="store_true", help="no data or training needed")
    args = p.parse_args(argv)

    if args.selfcheck:
        selfcheck()
        return

    keras.utils.set_random_seed(TRAIN.seed)
    model = build(filters=args.filters, hidden=args.hidden, dense=args.dense)
    model.summary()
    train(model, args.epochs, args.out)


if __name__ == "__main__":
    main()
