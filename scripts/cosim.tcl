# C/RTL co-simulation — the RTL-vs-C check, on whatever vectors tb_data holds.
#
# This is the real "does the generated Verilog compute what the C++ computes"
# test. Both sides run the same fixed-point arithmetic, so the two result logs
# must come out *identical*, not merely close.
#
#   export PATH="/c/Xilinx/Vitis_HLS/2023.2/bin:$PATH"
#   cd hls/jepasafe && vitis_hls -f ../../scripts/cosim.tcl
#
# Needs csynth to have run already (solution1/syn must exist) — cosim simulates
# the synthesised RTL, it does not produce it.
#
# Use a SUBSET. cosim simulates ~160k clock cycles per window, versus csim's
# ~0.2 s, so the full 2,947-window test set is not viable here:
#   python -m jepasafe.make_tb_data --out-dir data/processed/tb60 make --num-vectors 60
#   cp data/processed/tb60/*.dat hls/jepasafe/tb_data/
#
# Afterwards, compare the two logs bit-for-bit:
#   python -m jepasafe.make_tb_data --out-dir data/processed/tb60 compare \
#       <solution1>/csim/build/tb_data/csim_results.log \
#       --against <solution1>/sim/wrapc/tb_data/rtl_cosim_results.log
#
# trace_level is deliberately left at its default of none. hls4ml's build_prj.tcl
# passes `-trace_level all`, then patches `log_wave -r /` out of the generated
# run_sim.tcl to undo the cost of it. For a numerical check there is nothing to
# gain; add `-trace_level port` only when you actually want waveforms for a demo.

set tcldir [file dirname [info script]]
source [file join $tcldir .. hls jepasafe project.tcl]

if {![file exists tb_data/tb_input_features.dat]} {
    puts "ERROR: tb_data/tb_input_features.dat is missing."
    puts "ERROR: cosim would run on an all-zero window and prove nothing."
    exit 1
}

set n_vectors [exec wc -l < tb_data/tb_input_features.dat]
puts "***** COSIM over $n_vectors vector(s) *****"

# Vitis copies tb_data/ into its csim and cosim build directories and does NOT
# refresh those copies on a re-run without -reset. Stale vectors there are
# silent and vicious: swap in a 60-window subset, re-run, and csim grinds
# through the previous 2,947 instead, after which the comparison is 2,947 csim
# rows against 60 RTL rows. Purging just the copies gets fresh vectors while
# leaving solution1/syn/ intact — cosim needs those synthesis results.
foreach stale [list \
        ${project_name}_prj/solution1/csim/build/tb_data \
        ${project_name}_prj/solution1/sim/wrapc/tb_data \
        ${project_name}_prj/solution1/sim/wrapc_pc/tb_data] {
    if {[file exists $stale]} {
        puts "purging stale vectors: $stale"
        file delete -force $stale
    }
}

open_project ${project_name}_prj
set_top ${project_name}
add_files firmware/${project_name}.cpp -cflags "-std=c++0x"
add_files -tb ${project_name}_test.cpp -cflags "-std=c++0x"
add_files -tb firmware/weights
add_files -tb tb_data
open_solution "solution1"
set_part $part
create_clock -period $clock_period -name default

# csim first, on the same vectors, so there is something to diff the RTL against.
set time_start [clock clicks -milliseconds]
csim_design
set time_csim [clock clicks -milliseconds]
puts "***** C SIMULATION DONE IN [expr ($time_csim - $time_start) / 1000]s *****"

# Guard: prove csim actually consumed the vectors now in tb_data. The purge
# above can fail silently — on Windows the orphaned csim.exe from a previous
# killed run keeps a handle on the build copy, `file delete -force` does not
# report it, and csim then re-runs the *previous* vector set. That looks like a
# slow but healthy run, and the cosim comparison afterwards is meaningless.
set csim_log ${project_name}_prj/solution1/csim/build/tb_data/csim_results.log
set n_csim [exec wc -l < $csim_log]
if {$n_csim != $n_vectors} {
    puts "ERROR: csim produced $n_csim results but tb_data holds $n_vectors vectors."
    puts "ERROR: the build copy under solution1/csim/ was stale."
    puts "ERROR: kill any leftover csim.exe, delete solution1/csim/, and re-run."
    exit 1
}
puts "***** csim consumed all $n_vectors vectors *****"

# Vitis only defines __RTL_SIMULATION__ for SystemC testbenches, so the C++ one
# needs -DRTL_SIM to switch its output path to rtl_cosim_results.log. Re-adding
# the same file with different cflags is what hls4ml's own build_prj.tcl does.
add_files -tb ${project_name}_test.cpp -cflags "-std=c++0x -DRTL_SIM"
cosim_design -rtl verilog
set time_end [clock clicks -milliseconds]
puts "***** C/RTL SIMULATION DONE IN [expr ($time_end - $time_csim) / 1000]s *****"

puts [read [open ${project_name}_prj/solution1/sim/report/${project_name}_cosim.rpt r]]

exit
