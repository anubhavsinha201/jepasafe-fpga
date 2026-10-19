# Package an already-synthesised hls4ml project as a Vivado IP.
#
# This is the step that produces the one artefact Vivado can consume. csynth
# alone leaves RTL in jepasafe_prj/solution1/syn/verilog/, which Vivado cannot
# import directly — it needs an IP-XACT package.
#
# hls4ml's generated build_opt.tcl ships with `export 0`, and it is regenerated
# on every export_hls.py run, so editing it does not stick. Hence this file.
#
#   export PATH="/c/Xilinx/Vitis_HLS/2023.2/bin:$PATH"
#   cd hls/jepasafe && vitis_hls -f ../../scripts/export_ip.tcl
#
# Output: jepasafe_prj/solution1/impl/ip/  plus
#         jepasafe_prj/solution1/impl/export.zip
# Add either path to Vivado: Settings -> IP -> Repository.

open_project jepasafe_prj
open_solution solution1

# -flow is deliberately omitted. With it, Vitis runs Vivado synthesis to report
# post-synthesis resources (more accurate than the csynth estimate, and slower).
# Add `-flow syn` when you want the real numbers rather than the estimate.
export_design -format ip_catalog -rtl verilog \
    -display_name "JEPASafe CNN+LSTM HAR inference" \
    -description "1D CNN + LSTM activity classifier, 128x6 IMU window -> 6 class logits" \
    -vendor "nitt" -version "1.0"

exit
