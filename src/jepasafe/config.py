"""Paths and training constants shared by the deployment path.

Model dimensions deliberately live in ``cnnlstm_keras.build()`` defaults rather
than here: they get swept per resource-budget experiment, and a second copy in
this file would only drift out of step with the one that trained the weights.
"""

from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

DATA_PROCESSED = REPO_ROOT / "data" / "processed"
CHECKPOINTS = REPO_ROOT / "models" / "checkpoints"
HLS_DIR = REPO_ROOT / "hls"


@dataclass(frozen=True)
class TrainConfig:
    epochs: int = 50
    batch_size: int = 64
    learning_rate: float = 1e-3
    seed: int = 42


@dataclass(frozen=True)
class HLSConfig:
    """Precision and reuse factor are export_hls.py CLI flags, not constants —
    they are the two knobs you turn against a resource report."""

    granularity: str = "name"
    project_name: str = "jepasafe"


TRAIN = TrainConfig()
HLS = HLSConfig()
