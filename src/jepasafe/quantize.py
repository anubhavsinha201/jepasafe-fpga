"""Quantization-aware training with QKeras.

Post-training quantization to 8 bits usually costs a few points of AUC on a
model this small. QAT recovers most of it, so this rebuilds the network with
QKeras layers and fine-tunes from the float checkpoint.

    python -m jepasafe.quantize --model models/checkpoints/jepasafe.keras
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from tensorflow import keras

from .config import CHECKPOINTS, DATA_PROCESSED, QUANTIZED, TRAIN

# Matches HLSConfig.default_precision: ap_fixed<8,4> -> 8 total bits, 4 integer
BITS, INT_BITS = 8, 4
KERNEL_Q = f"quantized_bits({BITS},{INT_BITS},1,alpha=1)"
ACT_Q = f"quantized_relu({BITS},{INT_BITS})"


def build_quantized():
    """TODO: mirror model.build_jepasafe using QConv1D / QDense / QActivation.

    MultiHeadAttention has no QKeras equivalent — either
      (a) hand-roll attention from QDense + softmax, or
      (b) leave attention in float and let hls4ml quantize it via hls_config.
    Option (b) is the faster path to a first bitstream; revisit if resources blow up.
    """
    raise NotImplementedError("QKeras rebuild not written yet")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", type=Path, default=CHECKPOINTS / "jepasafe.keras")
    p.add_argument("--data", type=Path, default=DATA_PROCESSED)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--out", type=Path, default=QUANTIZED / "jepasafe_q.keras")
    args = p.parse_args()

    float_model = keras.models.load_model(args.model)
    qmodel = build_quantized()
    qmodel.set_weights(float_model.get_weights())

    x = np.load(args.data / "X.npy")
    y = np.load(args.data / "y.npy")

    qmodel.compile(
        optimizer=keras.optimizers.Adam(TRAIN.learning_rate / 10),
        loss="binary_crossentropy",
        metrics=["accuracy", keras.metrics.AUC(name="auc")],
    )
    qmodel.fit(x, y, epochs=args.epochs, batch_size=TRAIN.batch_size, validation_split=TRAIN.val_split)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    qmodel.save(args.out)
    print(f"saved -> {args.out}")


if __name__ == "__main__":
    main()
