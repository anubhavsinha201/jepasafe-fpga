"""Convert the quantized Keras model into an HLS project with hls4ml.

    python -m jepasafe.export_hls --model models/quantized/jepasafe_q.keras

Read the printed resource estimate against the GW1NR-9 budget before touching
the Gowin toolchain: 8,640 LUT4 and 20 DSP slices. If DSPs overflow, raise
HLS.reuse_factor (trades latency for area) before shrinking the model.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import hls4ml
from tensorflow import keras

from .config import HLS, HLS_DIR, QUANTIZED


def build_config(model):
    config = hls4ml.utils.config_from_keras_model(
        model,
        granularity=HLS.granularity,
        default_precision=HLS.default_precision,
    )
    config["Model"]["ReuseFactor"] = HLS.reuse_factor
    config["Model"]["Strategy"] = "Resource"  # area over throughput on a 9K part
    return config


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", type=Path, default=QUANTIZED / "jepasafe_q.keras")
    p.add_argument("--out", type=Path, default=HLS_DIR / HLS.project_name)
    p.add_argument("--build", action="store_true", help="run C synthesis after conversion")
    args = p.parse_args()

    model = keras.models.load_model(args.model)
    config = build_config(model)
    print(hls4ml.utils.print_dict(config))

    hls_model = hls4ml.converters.convert_from_keras_model(
        model,
        hls_config=config,
        output_dir=str(args.out),
        project_name=HLS.project_name,
        io_type=HLS.io_type,
    )
    hls_model.compile()
    print(f"HLS project -> {args.out}")

    if args.build:
        # Needs Vivado/Vitis HLS on PATH. The generated RTL then gets pulled
        # into the Gowin project by scripts/build.sh.
        hls_model.build(csim=False, synth=True, export=True)


if __name__ == "__main__":
    main()
