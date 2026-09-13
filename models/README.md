# Models

Weights are gitignored; the directory structure is not.

| Directory | Holds | Written by |
|---|---|---|
| `checkpoints/` | Float training checkpoints (`jepasafe.keras`) | `python -m jepasafe.train` |
| `quantized/` | QKeras quantization-aware models (`jepasafe_q.keras`) | `python -m jepasafe.quantize` |
| `exported/` | Weights dumped for HLS / RTL, plus normalisation constants | `python -m jepasafe.export_hls` |

Record float vs quantized AUC in [../docs/build-log.md](../docs/build-log.md)
every time you requantize — the gap between them is the number that justifies
the QAT step in the report.
