// MPU6050 int16 counts -> the model's ap_fixed<16,6> input format.
//
// The Keras model trains on raw physical units (g for accelerometer, rad/s for
// gyro) with no normalisation, specifically so that on-device preprocessing is
// this cheap: one arithmetic shift for the accelerometer, one multiply for the
// gyro. A per-channel affine transform would have cost far more.
//
// ap_fixed<16,6> is 16 bits total with 6 integer bits, so 10 fractional bits:
// the stored integer is the real value times 2^10, and the representable range
// is +/-32 with a resolution of 2^-10 ~ 0.001.
//
//   accelerometer   +/-4 g at 8192 LSB/g
//                   stored = (count / 8192) * 1024 = count / 8 = count >>> 3
//   gyro            +/-500 deg/s at 65.5 LSB per deg/s, model wants rad/s
//                   stored = count * (1/65.5) * (pi/180) * 1024
//                          = count * 0.27271 ~ (count * 279) >>> 10
//                   279/1024 = 0.27246, an error of 0.09%
//
// Note CLAUDE.md's "right-shift by 13" was derived for a narrower fixed-point
// format and is wrong here — it would scale the input down by 1024x. The shift
// is log2(LSB_per_unit) - fractional_bits, so it must be recomputed whenever
// the precision changes.
//
// Range check: count = 32767 gives 4095 (4.0 g) on the accelerometer and 8927
// (8.7 rad/s) on the gyro. Both sit inside +/-32, so no saturation logic is
// needed — but see the assertion in the testbench, which pins that.
//
// The scale factors are parameters on purpose. A real MPU6050 has offset and
// sensitivity error, and the calibration knob has to exist even though
// simulation never needs it.

`timescale 1ns / 1ps

module sample_scale #(
    parameter integer ACC_SHIFT  = 3,     // >>> 3  for +/-4 g at 10 frac bits
    parameter integer GYRO_MUL   = 279,   // round(0.27271 * 1024)
    parameter integer GYRO_SHIFT = 10
) (
    input  wire               clk,
    input  wire               rst_n,

    input  wire signed [15:0] ax,
    input  wire signed [15:0] ay,
    input  wire signed [15:0] az,
    input  wire signed [15:0] gx,
    input  wire signed [15:0] gy,
    input  wire signed [15:0] gz,
    input  wire               sample_valid,

    // 6 lanes of ap_fixed<16,6>. Channel 0 occupies the LSB — verified against
    // the hls4ml transaction vectors, not assumed: beat 0 of the test set
    // decodes to total_acc_x = 1.041 g in bits [15:0].
    output reg  [95:0]        sample_q,
    output reg                sample_q_valid
);

    // 16-bit count x 9-bit constant needs 25 bits; 32 is comfortable and costs
    // nothing after the shift truncates back to 16.
    function signed [15:0] scale_gyro;
        input signed [15:0] count;
        reg signed [31:0] wide;
        begin
            wide       = $signed(count) * GYRO_MUL;
            scale_gyro = wide >>> GYRO_SHIFT;
        end
    endfunction

    function signed [15:0] scale_acc;
        input signed [15:0] count;
        begin
            scale_acc = $signed(count) >>> ACC_SHIFT;
        end
    endfunction

    always @(posedge clk) begin
        if (!rst_n) begin
            sample_q       <= 96'd0;
            sample_q_valid <= 1'b0;
        end else begin
            sample_q_valid <= sample_valid;
            if (sample_valid) begin
                sample_q <= { scale_gyro(gz),   // [95:80] channel 5
                              scale_gyro(gy),   // [79:64] channel 4
                              scale_gyro(gx),   // [63:48] channel 3
                              scale_acc (az),   // [47:32] channel 2
                              scale_acc (ay),   // [31:16] channel 1
                              scale_acc (ax) }; // [15:0]  channel 0
            end
        end
    end

endmodule
