# Data

Contents of `raw/` and `processed/` are gitignored — datasets are large and
redistributable only from their original sources.

## Download into `raw/`

| Dataset | Why | Link |
|---|---|---|
| **UCI HAR** | Start here — 30 participants, 50 Hz, published baselines to compare against | https://archive.ics.uci.edu/ml/datasets/human+activity+recognition+using+smartphones |
| **BandX-Activity** | Same MPU6050 sensor as the deployment target, 33 volunteers, 20 Hz | https://zenodo.org/record/8400931 |
| **EdgeFall** | 1500 fall / daily-activity trials — the actual anomaly class | https://ieee-dataport.org/open-access/edgefall |

## `processed/`

Training reads `X.npy` and `y.npy` from here:

- `X.npy` — float32, shape `(n_windows, window_size, 6)`, channel order `[ax, ay, az, gx, gy, gz]`
- `y.npy` — uint8, shape `(n_windows,)`, `1` = anomaly (fall / violent motion)

Also save the normalisation constants (`mean.npy`, `std.npy`) — the same values
have to be baked into the FPGA preprocessing path, or inference on-device will
see a different input distribution than training did.

Windowing and normalisation helpers are in `src/jepasafe/data.py`.
