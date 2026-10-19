// Argmax over the core's 6 output logits.
//
// Replaces the broken comparator in the old jepasafe_top, which tested a single
// byte against ALERT_THRESHOLD = 8'd205 as though the model emitted an anomaly
// score. It does not: it emits 6 class logits, and picking a class means argmax.
//
// The core's layer14_out_TDATA is 384 bits: 6 lanes of 64, each holding an
// ap_fixed<37,17> value in its low 37 bits. Argmax is monotonic, so comparing
// the raw fixed-point fields as signed integers gives the same answer as
// comparing real values — no rescaling needed. Only a host that wants the
// actual logit magnitudes has to divide by 2^20.
//
// Lane 0 is the LSB. Verified against the hls4ml transaction vectors rather
// than assumed: the last 16 nibbles of transaction 0 decode to -7.22461, which
// is exactly csim's first logit.

`timescale 1ns / 1ps

module argmax6 #(
    parameter integer N_CLASSES = 6,
    parameter integer LANE_W    = 64,   // each logit is padded to 64 bit
    parameter integer FIELD_W   = 37    // ap_fixed<37,17>
) (
    input  wire                       clk,
    input  wire                       rst_n,

    input  wire [N_CLASSES*LANE_W-1:0] logits,
    input  wire                        logits_valid,

    output reg  [2:0]                  class_idx,
    output reg                         class_valid
);

    integer i;
    reg signed [FIELD_W-1:0] best;
    reg signed [FIELD_W-1:0] cand;
    reg        [2:0]         best_i;

    always @(posedge clk) begin
        if (!rst_n) begin
            class_idx   <= 3'd0;
            class_valid <= 1'b0;
        end else begin
            class_valid <= logits_valid;
            if (logits_valid) begin
                best   = logits[0 +: FIELD_W];
                best_i = 3'd0;
                // Strict > keeps the lowest index on a tie, matching numpy's
                // argmax, so RTL and the Python reference agree on ties.
                for (i = 1; i < N_CLASSES; i = i + 1) begin
                    cand = logits[i*LANE_W +: FIELD_W];
                    if (cand > best) begin
                        best   = cand;
                        best_i = i[2:0];
                    end
                end
                class_idx <= best_i;
            end
        end
    end

endmodule
