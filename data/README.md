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

`uci_har/` holds the parsed cache, written on first run of
`python -m jepasafe.demopart`:

- `X_{train,test}.npy` — float32 `(n_windows, 128, 9)`, channels
  `[body_acc_xyz, body_gyro_xyz, total_acc_xyz]`
- `y_{train,test}.npy` — int64 `(n_windows,)`, 0-5, see `demopart.CLASS_NAMES`

`cnnlstm_keras.py` narrows that to 6 channels (`total_acc` + `body_gyro`) —
exactly what an MPU6050 outputs. `body_acc` is `total_acc` with gravity filtered
out, which would mean reimplementing UCI HAR's Butterworth filter in RTL.

No normalisation constants: the model trains on raw physical units (g, rad/s),
so on-device preprocessing is a shift rather than a per-channel affine.
