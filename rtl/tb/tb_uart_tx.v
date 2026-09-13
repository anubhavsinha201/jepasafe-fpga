`timescale 1ns / 1ps

// Sends one byte and samples the line at mid-bit to recover it.
// Clock/baud are scaled down so DIVISOR stays small and the sim stays quick.

module tb_uart_tx;

    localparam CLK_HZ    = 1_000_000;
    localparam BAUD_RATE = 100_000;
    localparam DIVISOR   = CLK_HZ / BAUD_RATE;
    localparam CLK_PERIOD = 1000;  // ns, = 1 MHz

    reg        clk = 1'b0;
    reg        rst_n = 1'b0;
    reg  [7:0] data = 8'h00;
    reg        valid = 1'b0;
    wire       tx, busy;

    uart_tx #(.CLK_HZ(CLK_HZ), .BAUD_RATE(BAUD_RATE)) dut (
        .clk (clk), .rst_n(rst_n), .data(data), .valid(valid), .tx(tx), .busy(busy)
    );

    always #(CLK_PERIOD/2) clk = ~clk;

    localparam [7:0] TEST_BYTE = 8'hA5;

    integer i, errors = 0;
    reg [7:0] received;

    initial begin
        $dumpfile("sim/tb_uart_tx.vcd");
        $dumpvars(0, tb_uart_tx);

        repeat (4) @(posedge clk);
        rst_n = 1'b1;
        @(posedge clk);

        if (tx !== 1'b1) begin
            $display("FAIL: idle line is %b, expected 1", tx);
            errors = errors + 1;
        end

        data  = TEST_BYTE;
        valid = 1'b1;
        @(posedge clk);
        valid = 1'b0;

        // Wait for the start bit, then step to the middle of it
        wait (tx == 1'b0);
        repeat (DIVISOR/2) @(posedge clk);

        for (i = 0; i < 8; i = i + 1) begin
            repeat (DIVISOR) @(posedge clk);
            received[i] = tx;  // LSB first
        end

        repeat (DIVISOR) @(posedge clk);
        if (tx !== 1'b1) begin
            $display("FAIL: stop bit is %b, expected 1", tx);
            errors = errors + 1;
        end

        if (received !== TEST_BYTE) begin
            $display("FAIL: received %02h, expected %02h", received, TEST_BYTE);
            errors = errors + 1;
        end

        if (errors == 0) $display("PASS: tb_uart_tx");
        else             $display("FAIL: tb_uart_tx (%0d errors)", errors);
        $finish;
    end

endmodule
