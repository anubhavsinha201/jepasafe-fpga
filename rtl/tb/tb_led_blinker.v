`timescale 1ns / 1ps

// Checks the LED pattern rotates. TICKS is shortened so the sim finishes fast;
// the synthesised design keeps the 0.5 s default.

module tb_led_blinker;

    reg        clk = 1'b0;
    reg        rst_n = 1'b0;
    wire [5:0] led;

    localparam TICKS = 24'd9;

    led_blinker #(.TICKS(TICKS)) dut (
        .sys_clk   (clk),
        .sys_rst_n (rst_n),
        .led       (led)
    );

    always #18.5 clk = ~clk;  // ~27 MHz

    integer errors = 0;
    reg [5:0] prev;

    initial begin
        $dumpfile("sim/tb_led_blinker.vcd");
        $dumpvars(0, tb_led_blinker);

        repeat (4) @(posedge clk);
        rst_n = 1'b1;

        if (led !== 6'b111110) begin
            $display("FAIL: reset pattern is %b, expected 111110", led);
            errors = errors + 1;
        end

        // Watch four rotations
        repeat (4) begin
            prev = led;
            repeat (TICKS + 2) @(posedge clk);
            if (led !== {prev[4:0], prev[5]}) begin
                $display("FAIL: %b -> %b, expected %b", prev, led, {prev[4:0], prev[5]});
                errors = errors + 1;
            end
        end

        if (errors == 0) $display("PASS: tb_led_blinker");
        else             $display("FAIL: tb_led_blinker (%0d errors)", errors);
        $finish;
    end

endmodule
