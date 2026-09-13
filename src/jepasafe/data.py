"""Dataset loading and windowing.

Target datasets (see CLAUDE.md "Dataset Resources"):
  - UCI HAR       - start here, easiest to debug against published baselines
  - BandX-Activity - same MPU6050 sensor as the deployment target
  - EdgeFall      - 1500 fall / daily-activity trials

Each loader must return raw samples shaped (n_samples, 6) in a consistent
channel order: [ax, ay, az, gx, gy, gz].
"""

from __future__ import annotations

import numpy as np

from .config import MODEL


def window(series: np.ndarray, size: int | None = None, stride: int | None = None) -> np.ndarray:
    """Slice a continuous (n_samples, n_channels) recording into overlapping windows.

    Returns an array of shape (n_windows, size, n_channels).
    """
    size = size or MODEL.window_size
    stride = stride or size // 2

    if series.ndim != 2:
        raise ValueError(f"expected (n_samples, n_channels), got {series.shape}")
    if len(series) < size:
        raise ValueError(f"recording shorter than window: {len(series)} < {size}")

    starts = range(0, len(series) - size + 1, stride)
    return np.stack([series[s : s + size] for s in starts])


def normalise(windows: np.ndarray, stats: tuple[np.ndarray, np.ndarray] | None = None):
    """Per-channel zero-mean / unit-variance scaling.

    Pass ``stats`` from the training split when normalising val/test, so the
    same constants can be baked into the FPGA preprocessing path.
    """
    if stats is None:
        mean = windows.mean(axis=(0, 1))
        std = windows.std(axis=(0, 1)) + 1e-8
    else:
        mean, std = stats
    return (windows - mean) / std, (mean, std)


def load_uci_har(root):
    """TODO: parse the UCI HAR inertial signals into (n_samples, 6)."""
    raise NotImplementedError("Download UCI HAR into data/raw/ and implement the parser")


def load_bandx(root):
    """TODO: parse BandX-Activity (Zenodo record 8400931) into (n_samples, 6)."""
    raise NotImplementedError("Download BandX-Activity into data/raw/ and implement the parser")


def load_edgefall(root):
    """TODO: parse EdgeFall trials into (n_samples, 6) plus fall/no-fall labels."""
    raise NotImplementedError("Download EdgeFall into data/raw/ and implement the parser")


LOADERS = {
    "uci_har": load_uci_har,
    "bandx": load_bandx,
    "edgefall": load_edgefall,
}
