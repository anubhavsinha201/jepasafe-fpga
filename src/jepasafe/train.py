"""Train the float JEPASafe model.

    python -m jepasafe.train --data data/processed --epochs 50
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from tensorflow import keras

from .config import CHECKPOINTS, DATA_PROCESSED, TRAIN
from .model import build_jepasafe


def load_split(data_dir: Path):
    """Expects X.npy (n, window, 6) and y.npy (n,) written by a data prep step."""
    x = np.load(data_dir / "X.npy")
    y = np.load(data_dir / "y.npy")
    return x, y


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", type=Path, default=DATA_PROCESSED)
    p.add_argument("--epochs", type=int, default=TRAIN.epochs)
    p.add_argument("--batch-size", type=int, default=TRAIN.batch_size)
    p.add_argument("--out", type=Path, default=CHECKPOINTS / "jepasafe.keras")
    args = p.parse_args()

    keras.utils.set_random_seed(TRAIN.seed)

    x, y = load_split(args.data)
    print(f"loaded {x.shape[0]} windows of shape {x.shape[1:]}")

    model = build_jepasafe()
    model.compile(
        optimizer=keras.optimizers.Adam(TRAIN.learning_rate),
        loss="binary_crossentropy",
        metrics=["accuracy", keras.metrics.AUC(name="auc")],
    )
    model.summary()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    model.fit(
        x,
        y,
        epochs=args.epochs,
        batch_size=args.batch_size,
        validation_split=TRAIN.val_split,
        callbacks=[
            keras.callbacks.ModelCheckpoint(args.out, save_best_only=True, monitor="val_auc", mode="max"),
            keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True, monitor="val_auc", mode="max"),
        ],
    )
    print(f"saved -> {args.out}")


if __name__ == "__main__":
    main()
