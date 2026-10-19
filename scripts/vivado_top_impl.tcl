# Full standalone-PL implementation: synthesis -> place & route -> bitstream.
#
# This is the step that produces everything a board would have confirmed except
# the act of downloading: real post-route utilization, real timing closure on
# the real part, and a real .bit file.
#
#   export PATH="/c/Xilinx/Vivado/2023.2/bin:$PATH"
#   vivado -mode batch -source scripts/vivado_top_impl.tcl
#
# Unlike vivado_core_impl.tcl, the constraints are read BEFORE synth_design.
# That script's "No constraint files found" meant synthesis ran with no timing
# target at all, which among other things changes how aggressively Vivado
# infers DSP blocks — it reported 486 DSPs against a csynth estimate of 54.
# Constrained synthesis is the flow to trust.
#
# Outputs land in vivado/top_impl/.

set repo   [file normalize [file dirname [info script]]/..]
set rtl    $repo/hls/jepasafe/jepasafe_prj/solution1/syn/verilog
set outdir $repo/vivado/top_impl
set part   xczu7ev-ffvc1156-2-e
set board  xilinx.com:zcu104:part0:1.1

file mkdir $outdir
cd $outdir

foreach d [list $rtl $repo/rtl] {
    if {![file isdirectory $d]} { puts "ERROR: missing $d"; exit 1 }
}

# The core's weight ROMs load with $readmemh("./name.dat", ...), resolved
# against the synthesis working directory.
foreach f [glob -nocomplain $rtl/*.dat] { file copy -force $f $outdir }

create_project -in_memory -part $part
set_property board_part $board [current_project]

# The hand-written RTL plus the generated core. jepasafe_top instantiates the
# core directly rather than through the packaged IP: for a PL-only design there
# is no block diagram, so IP-XACT buys nothing.
read_verilog [glob $repo/rtl/*.v]
read_verilog [glob $rtl/*.v]
read_xdc     $repo/constraints/zcu104.xdc

synth_design -top jepasafe_top -part $part

report_utilization -file $outdir/utilization_synth.rpt
write_checkpoint -force $outdir/post_synth.dcp

opt_design
place_design
phys_opt_design
route_design

write_checkpoint -force $outdir/post_route.dcp
report_utilization        -file $outdir/utilization_placed.rpt
report_timing_summary     -file $outdir/timing_summary.rpt
report_clock_utilization  -file $outdir/clock_utilization.rpt
report_io                 -file $outdir/io.rpt
report_drc                -file $outdir/drc.rpt

set wns [get_property SLACK [get_timing_paths -max_paths 1 -nworst 1 -setup]]
set whs [get_property SLACK [get_timing_paths -max_paths 1 -nworst 1 -hold]]

set n_lut  [llength [get_cells -hier -quiet -filter {REF_NAME =~ LUT*}]]
set n_ff   [llength [get_cells -hier -quiet -filter {REF_NAME =~ FD*}]]
set n_dsp  [llength [get_cells -hier -quiet -filter {REF_NAME =~ DSP*}]]
set n_b36  [llength [get_cells -hier -quiet -filter {REF_NAME =~ RAMB36*}]]
set n_b18  [llength [get_cells -hier -quiet -filter {REF_NAME =~ RAMB18*}]]
set n_uram [llength [get_cells -hier -quiet -filter {REF_NAME =~ URAM*}]]

puts "### RESULT part=$part board=$board"
puts "### LUT=$n_lut  FF=$n_ff  DSP=$n_dsp  RAMB36=$n_b36  RAMB18=$n_b18  URAM=$n_uram"
puts "### BRAM_18K equivalent = [expr {$n_b36 * 2 + $n_b18}]"
puts "### WNS=$wns ns  WHS=$whs ns"

if {$wns >= 0 && $whs >= 0} {
    puts "### TIMING MET — writing bitstream"
    write_bitstream -force $outdir/jepasafe_top.bit
    puts "### BITSTREAM $outdir/jepasafe_top.bit"
} else {
    # A bitstream that does not meet timing will misbehave on hardware in ways
    # that look like a model bug, so do not produce one.
    puts "### TIMING VIOLATED — refusing to write a bitstream"
}

exit
