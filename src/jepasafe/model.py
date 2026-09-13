"""The JEPASafe network: 1D CNN front end + Transformer JEPA predictor.

Mirrors the architecture diagram in CLAUDE.md:

    IMU window (T, 6)
      -> Conv1D(8, k=3, relu) -> MaxPool1D(2)      # feature extractor
      -> TransformerEncoder(2 heads, d=16, ff=64)  # predictor
      -> GlobalAveragePooling1D
      -> Dense(1, sigmoid)                         # anomaly score
"""

from __future__ import annotations

from tensorflow import keras
from tensorflow.keras import layers

from .config import MODEL, ModelConfig


def transformer_encoder(x, cfg: ModelConfig):
    """One pre-norm encoder block. Pre-norm keeps activations in a tighter
    range, which matters once everything is squeezed into ap_fixed<8,4>."""
    h = layers.LayerNormalization(epsilon=1e-6)(x)
    h = layers.MultiHeadAttention(
        num_heads=cfg.n_heads,
        key_dim=cfg.d_model // cfg.n_heads,
    )(h, h)
    x = layers.Add()([x, h])

    h = layers.LayerNormalization(epsilon=1e-6)(x)
    h = layers.Dense(cfg.d_ff, activation="relu")(h)
    h = layers.Dense(cfg.d_model)(h)
    return layers.Add()([x, h])


def build_jepasafe(cfg: ModelConfig = MODEL) -> keras.Model:
    """Supervised anomaly-score model — the thing that ships to the FPGA."""
    inputs = keras.Input(shape=(cfg.window_size, cfg.n_channels), name="imu_window")

    # Stage 1: 1D CNN front end
    x = layers.Conv1D(cfg.conv_filters, cfg.conv_kernel, activation="relu", padding="same")(inputs)
    x = layers.MaxPooling1D(cfg.pool_size)(x)

    # Project to the transformer width
    x = layers.Conv1D(cfg.d_model, 1, name="to_d_model")(x)

    # Stage 2: Transformer JEPA predictor
    x = transformer_encoder(x, cfg)

    # Output: anomaly score in [0, 1]
    x = layers.GlobalAveragePooling1D()(x)
    outputs = layers.Dense(1, activation="sigmoid", name="anomaly_score")(x)

    return keras.Model(inputs, outputs, name="jepasafe")


def build_encoder(cfg: ModelConfig = MODEL) -> keras.Model:
    """Encoder alone — the shared trunk used during JEPA pretraining."""
    inputs = keras.Input(shape=(cfg.window_size, cfg.n_channels), name="imu_window")
    x = layers.Conv1D(cfg.conv_filters, cfg.conv_kernel, activation="relu", padding="same")(inputs)
    x = layers.MaxPooling1D(cfg.pool_size)(x)
    x = layers.Conv1D(cfg.d_model, 1, name="to_d_model")(x)
    return keras.Model(inputs, x, name="jepasafe_encoder")


def build_jepa_pretrainer(cfg: ModelConfig = MODEL):
    """TODO: JEPA self-supervised pretraining.

    Shape of the job:
      - context encoder + EMA target encoder (stop-gradient on the target)
      - mask a span of the window; predict the *target encoder's embeddings*
        of the masked span from the context embeddings
      - loss is L2 in embedding space, never in input space

    Once pretrained, load these weights into ``build_jepasafe`` and fine-tune
    the sigmoid head on labelled fall data.
    """
    raise NotImplementedError("JEPA pretraining loop not written yet")


if __name__ == "__main__":
    build_jepasafe().summary()
