// Checks argmax6 against the real logit vectors the C model produced, plus the
// sign and tie-breaking edge cases.

`timescale 1ns / 1ps

module tb_argmax6;

    reg clk = 0, rst_n = 0;
    always #5 clk = ~clk;

    reg [383:0] logits = 0;
    reg         logits_valid = 0;
    wire [2:0]  class_idx;
    wire        class_valid;

    integer errors = 0;

    argmax6 dut (.clk(clk), .rst_n(rst_n),
        .logits(logits), .logits_valid(logits_valid),
        .class_idx(class_idx), .class_valid(class_valid));

    task check;
        input [383:0] v;
        input [2:0]   want;
        input [127:0] name;
        begin
            @(negedge clk);
            logits = v;
            logits_valid = 1'b1;
            @(negedge clk);
            logits_valid = 1'b0;
            @(posedge clk);
            #1;
            if (class_idx !== want) begin
                $display("  FAIL %0s: got %0d want %0d", name, class_idx, want);
                errors = errors + 1;
            end
        end
    endtask

    // Build a 384-bit word from six 37-bit signed fields, lane 0 in the LSB.
    function [383:0] pack;
        input signed [36:0] l0, l1, l2, l3, l4, l5;
        begin
            pack = 384'd0;
            pack[0*64 +: 37] = l0;
            pack[1*64 +: 37] = l1;
            pack[2*64 +: 37] = l2;
            pack[3*64 +: 37] = l3;
            pack[4*64 +: 37] = l4;
            pack[5*64 +: 37] = l5;
        end
    endfunction

    initial begin
        repeat (4) @(posedge clk);
        rst_n = 1;

        // Transaction 0 of the real test set, straight from
        // c.jepasafe.autotvout_layer14_out.dat. Logit 4 (6428672/2^20 = 6.13)
        // is the largest, i.e. STANDING — and csim agrees.
        check(pack(-7575552, -3734528, -8174592, 3158016, 6428672, -4187136),
              3'd4, "tx0 STANDING");

        // Transaction 1: logit 4 again (6019072), the other logits all negative
        // or smaller.
        check(pack(-6356992, -4299776, -7348224, 3449856, 6019072, -3331072),
              3'd4, "tx1 STANDING");

        // Transaction 2: logit 5 (12353536) dominates -> LAYING.
        check(pack(-4255744, -7706624, -3404800, -743424, -6035456, 12353536),
              3'd5, "tx2 LAYING");

        // All negative: must still pick the largest (least negative), not fall
        // through to 0. This is the case a naive unsigned compare gets wrong.
        check(pack(-100, -50, -200, -300, -400, -500), 3'd1, "all negative");

        // Winner in lane 0.
        check(pack(99, -1, -2, -3, -4, -5), 3'd0, "lane 0 wins");

        // Winner in the top lane.
        check(pack(-1, -2, -3, -4, -5, 99), 3'd5, "lane 5 wins");

        // Tie: lowest index wins, matching numpy argmax so RTL and the Python
        // reference never disagree on a tie.
        check(pack(42, 42, 0, 0, 0, 0), 3'd0, "tie -> lowest index");

        // Most-negative 37-bit value must lose, not wrap around to a win.
        check(pack(37'sh1000000000, 1, 2, 3, 4, 5), 3'd5, "min negative loses");

        // Only bits 36:0 of each lane may be read. Lane 0 holds 0 while its
        // padding bits 63:37 are all ones; every other lane holds -1. Lane 0
        // must win. If the comparison took the full 64-bit lane, lane 0 would
        // read as a large negative and lose, so this isolates the field width.
        begin : padding_ignored
            reg [383:0] v;
            v = pack(0, -1, -1, -1, -1, -1);
            v[0*64 + 37 +: 27] = {27{1'b1}};
            check(v, 3'd0, "padding bits ignored");
        end

        if (errors == 0) $display("tb_argmax6: PASS");
        else             $display("tb_argmax6: FAIL (%0d error(s))", errors);
        $finish;
    end

endmodule
