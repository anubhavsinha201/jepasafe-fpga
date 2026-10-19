"""Convert the trained Keras CNN+LSTM into an HLS project with hls4ml.

    python -m jepasafe.export_hls
    python -m jepasafe.export_hls --precision "ap_fixed<8,4>"

Then run C synthesis by hand (see the note at the bottom of main()):

    export PATH="/c/Xilinx/Vitis_HLS/2023.2/bin:$PATH"
    cd hls/jepasafe && vitis_hls -f build_prj.tcl

Target is the ZCU104 (XCZU7EV): 230,400 LUT, 1,728 DSP, 312 BRAM36 (624 x 18 Kb),
96 URAM. The measured design is ~36k LUT / 54 DSP / 159 BRAM_18K, so there is
roughly 6x headroom on logic and 30x on DSP.

Defaults here are the configuration measured to work, not the fastest one:

  io_stream        Conv1D uses a line buffer and processes one time step at a
                   time. Under io_parallel the conv layers unroll across all 124
                   output positions and the design blew past 298k LUT.
  max reuse        One multiplier per layer, fully sequential. Latency comes out
                   ~3.2 ms against a 2.56 s window — three orders of magnitude of
                   slack — so there is nothing to gain by spending area on
                   parallelism. Lower --parallelism if you ever need the latency.
  static LSTM      One reused cell stepped 60 times, not 60 unrolled copies.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from pprint import pprint

import hls4ml
from tensorflow import keras

from .config import CHECKPOINTS, HLS, HLS_DIR
from .make_tb_data import INPUT_DAT, OUTPUT_DAT, TB_DIR

# ZCU104. Verified present in this Vivado/Vitis 2023.2 install.
DEFAULT_PART = "xczu7ev-ffvc1156-2-e"


def build_config(model, precision: str, parallelism: int, tb_dir: Path = TB_DIR):
    """Per-layer ReuseFactor = MACs/parallelism, i.e. `parallelism` multipliers.

    ReuseFactor has to divide the layer's MACs-per-call exactly or hls4ml
    silently rounds it down, so this derives it from the real weight shapes
    rather than taking one number on faith.
    """
    config = hls4ml.utils.config_from_keras_model(
        model,
        granularity=HLS.granularity,
        default_precision=precision,
        default_reuse_factor=1,
    )
    config["Model"]["Strategy"] = "Resource"

    for layer in model.layers:
        weights = layer.get_weights()
        if not weights or layer.__class__.__name__ not in ("Conv1D", "Dense", "LSTM"):
            continue
        macs = int(weights[0].size)  # Conv1D: k*cin*cout; Dense: in*out
        layer_config = config["LayerName"][layer.name]
        layer_config["ReuseFactor"] = max(1, macs // parallelism)
        if layer.__class__.__name__ == "LSTM":
            layer_config["static"] = True

    # Point the writer at the csim/cosim vectors. Without these two keys
    # write_test_bench() creates tb_data/ and leaves it empty, and
    # jepasafe_test.cpp then falls through to its fill_zero branch -- which is
    # how a cosim run can report "Pass" having classified nothing but a window
    # of zeros. Generate them with:  python -m jepasafe.make_tb_data make
    features, predictions = tb_dir / INPUT_DAT, tb_dir / OUTPUT_DAT
    if features.exists() and predictions.exists():
        config["InputData"] = str(features)
        config["OutputPredictions"] = str(predictions)
    else:
        print(
            f"WARNING: {tb_dir} has no testbench vectors, so tb_data/ will be\n"
            f"         empty and csim/cosim will run on an all-zero window.\n"
            f"         Fix with: python -m jepasafe.make_tb_data make"
        )

    return config


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", type=Path, default=CHECKPOINTS / "cnnlstm_keras.keras")
    p.add_argument("--out", type=Path, default=HLS_DIR / HLS.project_name)
    p.add_argument("--part", default=DEFAULT_PART)
    p.add_argument("--precision", default="ap_fixed<16,6>")
    p.add_argument("--parallelism", type=int, default=1,
                   help="multipliers per layer; 1 = minimum area (default)")
    p.add_argument("--io-type", default="io_stream", choices=["io_stream", "io_parallel"])
    p.add_argument("--clock-period", type=int, default=10, help="ns; 10 = 100 MHz")
    args = p.parse_args()

    model = keras.models.load_model(args.model)
    config = build_config(model, args.precision, args.parallelism)
    pprint(config)

    hls_model = hls4ml.converters.convert_from_keras_model(
        model,
        hls_config=config,
        output_dir=str(args.out),
        project_name=HLS.project_name,
        backend="Vitis",
        io_type=args.io_type,
        part=args.part,
        clock_period=args.clock_period,
    )

    # convert_from_keras_model only builds the graph in memory. write() is what
    # actually emits the C++/tcl project — without it hls/ stays empty.
    hls_model.write()

    print(f"\nHLS project -> {args.out}   part={args.part}")
    for layer in hls_model.get_layers():
        print(f"  {layer.class_name:22s} {layer.name:16s} "
              f"{str(layer.get_output_variable().shape):12s} rf={layer.get_attr('reuse_factor')}")

    # Deliberately not calling hls_model.compile() or .build() here.
    #   compile() runs build_lib.sh  -> needs bash + g++, fails under cmd.exe
    #   build()   runs `vitis-run`   -> ships with full Vitis, not Vitis_HLS 2023.2
    # ponytail: run csynth by hand; wire up a tcl driver only if this gets repetitive.
    print(
        f"\nNext: C synthesis\n"
        f'  export PATH="/c/Xilinx/Vitis_HLS/2023.2/bin:$PATH"\n'
        f"  cd {args.out} && vitis_hls -f build_prj.tcl\n"
        f"Report lands in {args.out / (HLS.project_name + '_prj')}/solution1/syn/report/"
    )


if __name__ == "__main__":
    main()
