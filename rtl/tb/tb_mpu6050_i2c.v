// Verifies mpu6050_i2c against the behavioural slave model.
//
// Checks the three things that break silently on real hardware:
//   - the init sequence writes the ranges sample_scale.v assumes
//   - a burst read reaches the right register with a repeated START
//   - the 14 bytes pair up big-endian, with temperature discarded
//
// I2C_HZ is cranked up so the test finishes quickly; the bit-level protocol is
// unchanged by the divider.

`timescale 1ns / 1ps

module tb_mpu6050_i2c;

    localparam integer CLK_HZ = 100_000_000;
    localparam integer I2C_HZ =   2_000_000;   // fast, for simulation speed
    localparam integer SAMPLE_HZ = 200_000;

    reg clk = 0, rst_n = 0;
    always #5 clk = ~clk;

    wire scl, sda_out, sda_oe;
    wire sda_line;
    wire signed [15:0] ax, ay, az, gx, gy, gz;
    wire sample_valid, error;

    integer errors = 0;
    integer samples = 0;

    mpu6050_i2c #(.CLK_HZ(CLK_HZ), .I2C_HZ(I2C_HZ), .SAMPLE_HZ(SAMPLE_HZ)) dut (
        .clk(clk), .rst_n(rst_n),
        .scl(scl), .sda_in(sda_line), .sda_out(sda_out), .sda_oe(sda_oe),
        .ax(ax), .ay(ay), .az(az), .gx(gx), .gy(gy), .gz(gz),
        .sample_valid(sample_valid), .error(error));

    mpu6050_model slave (
        .scl(scl), .sda_master_oe(sda_oe), .sda_to_master(sda_line));

    // Distinctive values, so a swapped or mis-paired axis is obvious.
    localparam signed [15:0] V_AX =  16'sd4096,  // 0.5 g at +/-4 g
                             V_AY = -16'sd8192,  // -1.0 g
                             V_AZ =  16'sd1234,
                             V_GX =  16'sd655,   // 10 deg/s
                             V_GY = -16'sd2000,
                             V_GZ =  16'sd32000;

    always @(posedge clk) if (sample_valid) begin
        samples = samples + 1;
        if (samples == 1) begin
            $display("  first sample: ax=%0d ay=%0d az=%0d gx=%0d gy=%0d gz=%0d",
                     ax, ay, az, gx, gy, gz);
            if (ax !== V_AX) begin $display("  FAIL ax %0d want %0d", ax, V_AX); errors=errors+1; end
            if (ay !== V_AY) begin $display("  FAIL ay %0d want %0d", ay, V_AY); errors=errors+1; end
            if (az !== V_AZ) begin $display("  FAIL az %0d want %0d", az, V_AZ); errors=errors+1; end
            if (gx !== V_GX) begin $display("  FAIL gx %0d want %0d", gx, V_GX); errors=errors+1; end
            if (gy !== V_GY) begin $display("  FAIL gy %0d want %0d", gy, V_GY); errors=errors+1; end
            if (gz !== V_GZ) begin $display("  FAIL gz %0d want %0d", gz, V_GZ); errors=errors+1; end
        end
    end

    initial begin
        // Load the motion registers and clear temperature so a mis-paired read
        // shows up as a wrong axis rather than a plausible number.
        slave.set_axis(8'h3B, V_AX);
        slave.set_axis(8'h3D, V_AY);
        slave.set_axis(8'h3F, V_AZ);
        slave.set_axis(8'h41, 16'sd0);     // temperature, discarded
        slave.set_axis(8'h43, V_GX);
        slave.set_axis(8'h45, V_GY);
        slave.set_axis(8'h47, V_GZ);

        repeat (10) @(posedge clk);
        rst_n = 1;

        // Long enough for init (3 writes) plus a couple of burst reads.
        repeat (60000) @(posedge clk);

        $display("  init writes: PWR_MGMT_1=0x%02h GYRO_CONFIG=0x%02h ACCEL_CONFIG=0x%02h",
                 slave.wrote_pwr, slave.wrote_gyro, slave.wrote_acc);
        if (slave.wrote_pwr !== 8'h00) begin
            $display("  FAIL device never woken (PWR_MGMT_1 not 0x00)"); errors = errors + 1;
        end
        // 0x08 is what sample_scale.v's scale factors assume: +/-500 deg/s and
        // +/-4 g. A mismatch here silently rescales every model input.
        if (slave.wrote_gyro !== 8'h08) begin
            $display("  FAIL GYRO_CONFIG not 0x08 (+/-500 deg/s)"); errors = errors + 1;
        end
        if (slave.wrote_acc !== 8'h08) begin
            $display("  FAIL ACCEL_CONFIG not 0x08 (+/-4 g)"); errors = errors + 1;
        end
        if (slave.n_bursts == 0) begin
            $display("  FAIL no burst read from 0x3B (repeated START broken?)");
            errors = errors + 1;
        end
        if (samples == 0) begin
            $display("  FAIL no sample_valid pulse"); errors = errors + 1;
        end
        if (error) begin
            $display("  FAIL master reported a NACK"); errors = errors + 1;
        end

        $display("  bursts=%0d samples=%0d", slave.n_bursts, samples);
        if (errors == 0) $display("tb_mpu6050_i2c: PASS");
        else             $display("tb_mpu6050_i2c: FAIL (%0d error(s))", errors);
        $finish;
    end

endmodule
