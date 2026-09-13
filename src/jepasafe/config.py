"""Single source of truth for model + deployment hyperparameters.

Numbers here mirror the architecture diagram in CLAUDE.md. Changing one means
retraining, re-quantizing and re-running hls4ml, so keep them in one place.
"""

from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

DATA_RAW = REPO_ROOT / "data" / "raw"
DATA_PROCESSED = REPO_ROOT / "data" / "processed"
CHECKPOINTS = REPO_ROOT / "models" / "checkpoints"
QUANTIZED = REPO_ROOT / "models" / "quantized"
EXPORTED = REPO_ROOT / "models" / "exported"
HLS_DIR = REPO_ROOT / "hls"


@dataclass(frozen=True)
class ModelConfig:
    # Input: a sliding window of 6-axis IMU samples (accel x/y/z + gyro x/y/z)
    window_size: int = 64
    n_channels: int = 6

    # Stage 1 - 1D CNN front end
    conv_filters: int = 8
    conv_kernel: int = 3
    pool_size: int = 2

    # Stage 2 - Transformer JEPA predictor
    n_heads: int = 2
    d_model: int = 16
    d_ff: int = 64

    # Output
    alert_threshold: float = 0.8


@dataclass(frozen=True)
class TrainConfig:
    epochs: int = 50
    batch_size: int = 64
    learning_rate: float = 1e-3
    val_split: float = 0.2
    seed: int = 42


@dataclass(frozen=True)
class HLSConfig:
    """Precision must fit the GW1NR-9 budget: 8,640 LUT4 and 20 DSP slices."""

    default_precision: str = "ap_fixed<8,4>"
    granularity: str = "name"
    reuse_factor: int = 8  # higher = fewer DSPs, more latency
    io_type: str = "io_stream"
    project_name: str = "jepasafe"


MODEL = ModelConfig()
TRAIN = TrainConfig()
HLS = HLSConfig()
