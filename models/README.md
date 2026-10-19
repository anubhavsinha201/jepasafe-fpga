# Models

Weights are gitignored; the directory structure is not.

| Directory | Holds | Written by |
|---|---|---|
| `checkpoints/` | `cnnlstm_keras.keras` (deployed), `cnnlstm.pt` (PyTorch baseline) | `jepasafe.cnnlstm_keras`, `jepasafe.demopart` |

There is no separate quantization step: precision is set in the hls4ml config
(`export_hls.py --precision`), so the fixed-point model only exists inside the
generated HLS project.

Record float vs fixed-point accuracy in [../docs/build-log.md](../docs/build-log.md)
every time you change `--precision` — that gap is the number the report needs.
Measuring it requires hls4ml's `compile()`, which needs bash + g++ (so WSL or
msys64 on Windows, not cmd.exe).
