// Standalone-PL top: sensor -> scale -> window -> inference -> class -> LEDs.
//
// Everything runs in the programmable logic; the Cortex-A53s are not involved.
// That is a deliberate choice (CLAUDE.md "Gap 0"), and it is also what makes
// the design verifiable without a board, since a PYNQ flow would need the PS
// running Python.
//
// Pin locations and I/O standards come from constraints/zcu104.xdc, which is
// derived from the installed ZCU104 board file (xilinx.com:zcu104:part0:1.1)
// rather than from guesswork.
//
// Clocking: the ZCU104's user clock is a 300 MHz differential pair, not a
// single-ended 125 MHz. It is divided to the 100 MHz the design was
// characterised at by an MMCM. That makes this module Vivado-only — IBUFDS and
// MMCME4_BASE are Xilinx primitives Icarus cannot elaborate. Every block below
// the clocking has its own Icarus testbench in rtl/tb/ for that reason.
//
// On the alert output: the model emits 6 class logits for six ORDINARY
// activities. UCI HAR contains no fall, impact or assault class, so
// ALERT_CLASSES defaults to 6'b000000 — nothing raises an alert. Setting a bit
// here declares "this activity is an emergency", which for walking or sitting
// is false. Making this a genuine safety trigger needs EdgeFall (or a collected
// fall set) and a retrain; it is a dataset gap, not a wiring gap.

`timescale 1ns / 1ps

module jepasafe_top #(
    // Bit n set => class n raises the alert. Class order:
    //   0 WALKING  1 WALKING_UPSTAIRS  2 WALKING_DOWNSTAIRS
    //   3 SITTING  4 STANDING          5 LAYING
    parameter [5:0]   ALERT_CLASSES = 6'b000000,
    parameter integer CLK_HZ        = 100_000_000,
    parameter integer SAMPLE_HZ     = 50,
    parameter integer WINDOW_SIZE   = 128,
    parameter integer STRIDE        = 128
) (
    input  wire       clk_300_p,     // AH18, DIFF_SSTL12
    input  wire       clk_300_n,     // AH17
    input  wire       cpu_resetn,    // M11, active low

    // PL I2C to the MPU6050 (LVCMOS33, so no level shifting needed)
    output wire       mpu_scl,       // N12
    inout  wire       mpu_sda,       // P12

    output wire [3:0] led            // D5 D6 A5 B5 — four, not six
);

    // ------------------------------------------------------------ clocking
    wire clk_300, clk, mmcm_fb, mmcm_locked;

    IBUFDS u_clkbuf (.I(clk_300_p), .IB(clk_300_n), .O(clk_300));

    // 300 MHz / 1 * 4 = 1200 MHz VCO (inside the 800-1600 range for -2),
    // / 12 = 100 MHz.
    MMCME4_BASE #(
        .CLKIN1_PERIOD    (3.333),
        .DIVCLK_DIVIDE    (1),
        .CLKFBOUT_MULT_F  (4.0),
        .CLKOUT0_DIVIDE_F (12.0)
    ) u_mmcm (
        .CLKIN1   (clk_300),
        .CLKFBIN  (mmcm_fb),
        .CLKFBOUT (mmcm_fb),
        .CLKOUT0  (clk),
        .LOCKED   (mmcm_locked),
        .PWRDWN   (1'b0),
        .RST      (~cpu_resetn)
    );

    // Hold the whole design in reset until the clock is actually stable.
    wire rst_n = mmcm_locked & cpu_resetn;

    // -------------------------------------------------------------- sensor
    wire sda_out, sda_oe, sda_in;
    // Open-drain: pull low or release. Never drive a 1 onto an I2C bus.
    assign mpu_sda = sda_oe ? 1'b0 : 1'bz;
    assign sda_in  = mpu_sda;

    wire signed [15:0] ax, ay, az, gx, gy, gz;
    wire sample_valid, i2c_error;

    mpu6050_i2c #(.CLK_HZ(CLK_HZ), .SAMPLE_HZ(SAMPLE_HZ)) u_sensor (
        .clk(clk), .rst_n(rst_n),
        .scl(mpu_scl), .sda_in(sda_in), .sda_out(sda_out), .sda_oe(sda_oe),
        .ax(ax), .ay(ay), .az(az), .gx(gx), .gy(gy), .gz(gz),
        .sample_valid(sample_valid), .error(i2c_error)
    );

    // ------------------------------------------------- counts -> ap_fixed
    wire [95:0] sample_q;
    wire        sample_q_valid;

    sample_scale u_scale (
        .clk(clk), .rst_n(rst_n),
        .ax(ax), .ay(ay), .az(az), .gx(gx), .gy(gy), .gz(gz),
        .sample_valid(sample_valid),
        .sample_q(sample_q), .sample_q_valid(sample_q_valid)
    );

    // -------------------------------------------------------------- window
    wire [95:0] stream_data;
    wire        stream_valid, stream_ready, window_done;

    window_buffer #(
        .WINDOW_SIZE(WINDOW_SIZE), .N_CHANNELS(6), .DATA_WIDTH(16),
        .STRIDE(STRIDE)
    ) u_window (
        .clk(clk), .rst_n(rst_n),
        .sample_in(sample_q), .sample_valid(sample_q_valid),
        .stream_data(stream_data), .stream_valid(stream_valid),
        .stream_ready(stream_ready),
        .window_done(window_done), .stream_busy()
    );

    // ----------------------------------------------------------- inference
    // The hls4ml core. ap_start is tied high because the generated IP has no
    // AXI-Lite register map (ap_ctrl_hs only), so there is no register to start
    // it from — it free-runs and consumes whatever the stream delivers.
    wire [383:0] logits;
    wire         logits_valid;

    jepasafe u_core (
        .imu_window_TDATA  (stream_data),
        .imu_window_TVALID (stream_valid),
        .imu_window_TREADY (stream_ready),
        .layer14_out_TDATA (logits),
        .layer14_out_TVALID(logits_valid),
        .layer14_out_TREADY(1'b1),
        .ap_clk            (clk),
        .ap_rst_n          (rst_n),
        .ap_start          (1'b1),
        .ap_done           (),
        .ap_ready          (),
        .ap_idle           ()
    );

    // -------------------------------------------------------------- argmax
    wire [2:0] class_idx;
    wire       class_valid;

    argmax6 u_argmax (
        .clk(clk), .rst_n(rst_n),
        .logits(logits), .logits_valid(logits_valid),
        .class_idx(class_idx), .class_valid(class_valid)
    );

    // --------------------------------------------------------------- alert
    reg [2:0] last_class;
    reg       alert;

    always @(posedge clk) begin
        if (!rst_n) begin
            last_class <= 3'd0;
            alert      <= 1'b0;
        end else if (class_valid) begin
            last_class <= class_idx;
            alert      <= ALERT_CLASSES[class_idx];
        end
    end

    // Show the classified activity rather than a bare alert bit: with only six
    // classes and no fall label, the class index is the informative output and
    // it is what a waveform or a demo video can actually be checked against.
    // LEDs are active high on this board.
    assign led = {alert | i2c_error, last_class};

endmodule
