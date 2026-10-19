# Out-of-context synthesis + implementation of the hls4ml core alone.
#
# Replaces the csynth *estimate* (33,500 LUT / 54 DSP / 159 BRAM_18K) with real
# post-synthesis and post-route numbers from Vivado, and proves the design
# closes timing on the actual part. No board required — the only step a board
# would add is downloading the bitstream.
#
#   export PATH="/c/Xilinx/Vivado/2023.2/bin:$PATH"
#   vivado -mode batch -source scripts/vivado_core_impl.tcl
#
# Out-of-context because the core's ports are AXI4-Stream, not package pins:
# OOC skips I/O buffer insertion so the module can be placed and routed on its
# own. The full standalone-PL top with real pins is a separate script.
#
# Reports land in vivado/core_impl/.

set repo    [file normalize [file dirname [info script]]/..]
set rtl     $repo/hls/jepasafe/jepasafe_prj/solution1/syn/verilog
set outdir  $repo/vivado/core_impl
set part    xczu7ev-ffvc1156-2-e
set period  10.0

file mkdir $outdir
cd $outdir

if {![file isdirectory $rtl]} {
    puts "ERROR: no generated RTL at $rtl — run csynth first."
    exit 1
}

set vfiles [glob -nocomplain $rtl/*.v]
set dfiles [glob -nocomplain $rtl/*.dat]
puts "### [llength $vfiles] Verilog files, [llength $dfiles] ROM .dat files"

# The ROMs load with $readmemh("./name.dat", ...). Vivado resolves that against
# the synthesis working directory, so the .dat files have to sit here. Copying
# them is more reliable than hoping the source path is searched.
foreach f $dfiles { file copy -force $f $outdir }

read_verilog $vfiles

# -mode out_of_context: no I/O buffers, so the AXI-Stream ports do not need pins.
synth_design -top jepasafe -part $part -mode out_of_context

# Constrain the clock the design was characterised at (100 MHz, 0.37 ns slack
# per the Vitis HLS report) so the timing numbers are comparable.
create_clock -period $period -name ap_clk [get_ports ap_clk]

report_utilization -file $outdir/utilization_synth.rpt
write_checkpoint -force $outdir/post_synth.dcp

opt_design
place_design
phys_opt_design
route_design

write_checkpoint -force $outdir/post_route.dcp
report_utilization    -file $outdir/utilization_placed.rpt
report_timing_summary -file $outdir/timing_summary.rpt
report_design_analysis -file $outdir/design_analysis.rpt

# Pull the numbers that matter straight into the log, so the result is readable
# without opening four reports.
set wns [get_property SLACK [get_timing_paths -max_paths 1 -nworst 1 -setup]]
set whs [get_property SLACK [get_timing_paths -max_paths 1 -nworst 1 -hold]]

# Count primitives directly rather than scraping the utilization report.
set n_lut  [llength [get_cells -hier -quiet -filter {REF_NAME =~ LUT*}]]
set n_ff   [llength [get_cells -hier -quiet -filter {REF_NAME =~ FD*}]]
set n_dsp  [llength [get_cells -hier -quiet -filter {REF_NAME =~ DSP*}]]
set n_b36  [llength [get_cells -hier -quiet -filter {REF_NAME =~ RAMB36*}]]
set n_b18  [llength [get_cells -hier -quiet -filter {REF_NAME =~ RAMB18*}]]
set n_uram [llength [get_cells -hier -quiet -filter {REF_NAME =~ URAM*}]]

puts "### RESULT part=$part clock=${period}ns"
puts "### LUT=$n_lut  FF=$n_ff  DSP=$n_dsp  RAMB36=$n_b36  RAMB18=$n_b18  URAM=$n_uram"
puts "### BRAM_18K equivalent = [expr {$n_b36 * 2 + $n_b18}]"
puts "### WNS=$wns ns  WHS=$whs ns"
if {$wns >= 0 && $whs >= 0} {
    puts "### TIMING MET"
} else {
    puts "### TIMING VIOLATED"
}

exit
