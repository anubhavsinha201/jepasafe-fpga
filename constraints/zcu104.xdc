# ZCU104 (XCZU7EV-2FFVC1156E) constraints for jepasafe_top.
#
# Every LOC and IOSTANDARD below was read out of the installed board file
# (xilinx.com:zcu104:part0:1.1) with get_board_part_pins, not taken from a
# datasheet or guessed. Three of them contradict assumptions recorded earlier
# in CLAUDE.md and docs/simulation-plan.md:
#
#   - The user clock is a 300 MHz DIFFERENTIAL pair on DIFF_SSTL12, not a
#     single-ended 125 MHz. jepasafe_top divides it to 100 MHz with an MMCM.
#   - There are FOUR PL user LEDs, not six. The old led[5:0] could never have
#     been constrained.
#   - The PL I2C sits on LVCMOS33, which is exactly what an MPU6050 wants, so
#     the 3.3 V question in Gap 4 resolves favourably with no level shifting.
#     (The PL UART is LVCMOS18, so an HC-08 on that path WOULD need shifting.)
#
# Replaces constraints/tang_nano_9k.cst, which targeted the Gowin part.

# ---------------------------------------------------------------- clock
set_property -dict {PACKAGE_PIN AH18 IOSTANDARD DIFF_SSTL12} [get_ports clk_300_p]
set_property -dict {PACKAGE_PIN AH17 IOSTANDARD DIFF_SSTL12} [get_ports clk_300_n]

create_clock -period 3.333 -name clk_300 [get_ports clk_300_p]

# The design is characterised at 100 MHz; the MMCM output is derived, so Vivado
# infers its period from CLKOUT0_DIVIDE_F. Nothing further to declare here.

# ---------------------------------------------------------------- reset
set_property -dict {PACKAGE_PIN M11 IOSTANDARD LVCMOS33} [get_ports cpu_resetn]

# Asynchronous push button: keep it out of timing analysis.
set_false_path -from [get_ports cpu_resetn]

# ---------------------------------------------------------------- I2C
# PL-side I2C (IIC1), shared with the board's own I2C tree — check the jumper
# settings in UG1267 before attaching an external MPU6050, since other devices
# live on this bus.
set_property -dict {PACKAGE_PIN N12 IOSTANDARD LVCMOS33} [get_ports mpu_scl]
set_property -dict {PACKAGE_PIN P12 IOSTANDARD LVCMOS33} [get_ports mpu_sda]

# 100 kHz open-drain with an external pull-up: nothing here is timing-critical,
# and sda is bidirectional, so constraining it would only create false failures.
set_false_path -to   [get_ports mpu_scl]
set_false_path -from [get_ports mpu_sda]
set_false_path -to   [get_ports mpu_sda]

# ---------------------------------------------------------------- LEDs
set_property -dict {PACKAGE_PIN D5 IOSTANDARD LVCMOS33} [get_ports {led[0]}]
set_property -dict {PACKAGE_PIN D6 IOSTANDARD LVCMOS33} [get_ports {led[1]}]
set_property -dict {PACKAGE_PIN A5 IOSTANDARD LVCMOS33} [get_ports {led[2]}]
set_property -dict {PACKAGE_PIN B5 IOSTANDARD LVCMOS33} [get_ports {led[3]}]

# led[2:0] carries the class index and led[3] the alert/error flag. A human
# reading an LED cannot be a timing path.
set_false_path -to [get_ports {led[*]}]
