Self-contained Vitis HLS source set for the JEPASafe CNN+LSTM accelerator.
Copied from hls/jepasafe/ (regenerate with: python -m jepasafe.export_hls).

ADD TO VITIS HLS
----------------
Top Function : jepasafe

Design Files (ONE file):
    firmware\jepasafe.cpp          CFLAGS: -std=c++0x

Testbench Files:
    jepasafe_test.cpp              CFLAGS: -std=c++0x
    firmware\weights               (Add Folders..., not Add Files...)

Solution Configuration:
    Part          xczu7ev-ffvc1156-2-e      (Parts tab; ZCU104 board file not installed)
    Clock period  10 ns
    Uncertainty   27%

DO NOT add the .h files individually.
    jepasafe.cpp  ->  includes jepasafe.h and parameters.h
    parameters.h  ->  includes all 12 weights/*.h
    parameters.h  ->  includes the nnet_utils/*.h layer implementations
Everything resolves by relative path from the source locations, so no
include directories need to be configured either.

The folder layout matters: jepasafe_test.cpp includes "firmware/jepasafe.h",
so it must stay one level above firmware/. Keep this structure intact.

CONTENTS
    firmware/jepasafe.cpp      top-level function (the only design file)
    firmware/jepasafe.h        its prototype
    firmware/defines.h         layer dimensions and fixed-point typedefs
    firmware/parameters.h      per-layer config structs; includes the weights
    firmware/weights/          12 .h weight+bias arrays (and .txt copies)
    firmware/nnet_utils/       56 .h hls4ml layer implementations
    firmware/ap_types/         Xilinx arbitrary-precision type headers
    jepasafe_test.cpp          C testbench

NOTE: run Synthesis, not C Simulation. The tb_data/ vectors hls4ml normally
writes for csim were not generated, so csim will fail on a missing input file.
