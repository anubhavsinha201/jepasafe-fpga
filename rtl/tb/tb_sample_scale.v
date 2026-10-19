// Checks the int16 -> ap_fixed<16,6> conversion against hand-computed values,
// and pins the lane order and the no-saturation-needed claim.

`timescale 1ns / 1ps

module tb_sample_scale;

    reg clk = 0, rst_n = 0;
    always #5 clk = ~clk;

    reg signed [15:0] ax, ay, az, gx, gy, gz;
    reg sample_valid = 0;
    wire [95:0] q;
    wire q_valid;

    integer errors = 0;

    sample_scale dut (.clk(clk), .rst_n(rst_n),
        .ax(ax), .ay(ay), .az(az), .gx(gx), .gy(gy), .gz(gz),
        .sample_valid(sample_valid), .sample_q(q), .sample_q_valid(q_valid));

    task drive;
        input signed [15:0] a0, a1, a2, g0, g1, g2;
        begin
            @(negedge clk);
            ax = a0; ay = a1; az = a2; gx = g0; gy = g1; gz = g2;
            sample_valid = 1'b1;
            @(negedge clk);
            sample_valid = 1'b0;
            @(posedge clk);  // q settles the cycle after valid
            #1;
        end
    endtask

    task expect_lane;
        input integer lane;
        input signed [15:0] want;
        reg signed [15:0] got;
        begin
            got = q[lane*16 +: 16];
            if (got !== want) begin
                $display("  FAIL lane %0d: got %0d want %0d", lane, got, want);
                errors = errors + 1;
            end
        end
    endtask

    initial begin
        repeat (4) @(posedge clk);
        rst_n = 1;

        // 1 g on the accelerometer at +/-4 g is 8192 counts -> 8192>>>3 = 1024,
        // which is 1.0 in ap_fixed<16,6>.
        drive(8192, -8192, 0, 0, 0, 0);
        expect_lane(0,  1024);
        expect_lane(1, -1024);
        expect_lane(2,     0);

        // Gyro: 65.5 counts is 1 deg/s = 0.017453 rad/s -> 17.87 stored.
        // (65 * 279) >>> 10 = 18138 >>> 10 = 17.
        drive(0, 0, 0, 65, 0, 0);
        expect_lane(3, 17);

        // Lane order: all six distinct, channel 0 in the LSB.
        drive(8, 16, 24, 0, 0, 0);
        expect_lane(0, 1);
        expect_lane(1, 2);
        expect_lane(2, 3);

        // Full-scale must not overflow ap_fixed<16,6>'s +/-32 range, i.e. the
        // stored magnitude must stay below 32*1024 = 32768. This is the claim
        // that lets sample_scale skip saturation logic entirely.
        drive(32767, -32768, 0, 32767, -32768, 0);
        if ($signed(q[0  +: 16]) !== 4095)  begin $display("  FAIL acc max %0d", $signed(q[0 +: 16])); errors = errors + 1; end
        if ($signed(q[16 +: 16]) !== -4096) begin $display("  FAIL acc min %0d", $signed(q[16 +: 16])); errors = errors + 1; end
        if ($signed(q[48 +: 16]) !== 8927)  begin $display("  FAIL gyro max %0d", $signed(q[48 +: 16])); errors = errors + 1; end
        if ($signed(q[64 +: 16]) !== -8928) begin $display("  FAIL gyro min %0d", $signed(q[64 +: 16])); errors = errors + 1; end

        // Negative values must use an arithmetic shift, not a logical one.
        drive(-8, 0, 0, 0, 0, 0);
        expect_lane(0, -1);

        if (errors == 0) $display("tb_sample_scale: PASS");
        else             $display("tb_sample_scale: FAIL (%0d error(s))", errors);
        $finish;
    end

endmodule
