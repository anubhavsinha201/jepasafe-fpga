# C simulation only — the fixed-point accuracy number, without re-synthesising.
#
# hls4ml's build_prj.tcl runs csim + csynth + cosim + validation in one go, and
# it regenerates build_opt.tcl on every export_hls.py run, so flipping the opt
# array there does not stick. csynth takes ~4 minutes and cosim far longer;
# neither is needed to find out what quantisation costs in accuracy.
#
#   export PATH="/c/Xilinx/Vitis_HLS/2023.2/bin:$PATH"
#   cd hls/jepasafe && vitis_hls -f ../../scripts/csim.tcl
#
# Requires tb_data/tb_input_features.dat to exist, or jepasafe_test.cpp falls
# through to its fill_zero branch and "passes" on a window of zeros:
#   python -m jepasafe.make_tb_data make
#
# Output: tb_data/csim_results.log   (one line of 6 fixed-point logits per window)
# Score it with:
#   python -m jepasafe.make_tb_data compare hls/jepasafe/tb_data/csim_results.log

set tcldir [file dirname [info script]]
source [file join $tcldir .. hls jepasafe project.tcl]

if {![file exists tb_data/tb_input_features.dat]} {
    puts "ERROR: tb_data/tb_input_features.dat is missing."
    puts "ERROR: csim would run on an all-zero window and prove nothing."
    puts "ERROR: generate it with: python -m jepasafe.make_tb_data make"
    exit 1
}

# Reuse the existing solution rather than -reset: that keeps the csynth results
# already sitting in solution1/syn/ intact.
open_project ${project_name}_prj
set_top ${project_name}
add_files firmware/${project_name}.cpp -cflags "-std=c++0x"
add_files -tb ${project_name}_test.cpp -cflags "-std=c++0x"
add_files -tb firmware/weights
add_files -tb tb_data
open_solution "solution1"
set_part $part
create_clock -period $clock_period -name default

set time_start [clock clicks -milliseconds]
csim_design
set time_end [clock clicks -milliseconds]
puts "***** C SIMULATION COMPLETED IN [expr ($time_end - $time_start) / 1000]s *****"

exit
