// Week 1 "hello world": rotate the six onboard LEDs every 0.5 s.
// Source: CLAUDE.md "Quick Commands & Snippets".

// TICKS is a parameter purely so the testbench can shorten it — the default
// is the 0.5 s @ 27 MHz value from the doc.

module led_blinker #(
    parameter TICKS = 24'd13_499_999
) (
    input  wire       sys_clk,    // 27 MHz clock input
    input  wire       sys_rst_n,  // Reset (active low)
    output reg  [5:0] led         // 6 LEDs on the board (active low)
);

    reg [23:0] counter;

    always @(posedge sys_clk or negedge sys_rst_n) begin
        if (!sys_rst_n)
            counter <= 24'd0;
        else if (counter < TICKS)
            counter <= counter + 1'b1;
        else
            counter <= 24'd0;
    end

    always @(posedge sys_clk or negedge sys_rst_n) begin
        if (!sys_rst_n)
            led <= 6'b111110;
        else if (counter == TICKS)
            led <= {led[4:0], led[5]};  // rotate left
    end

endmodule
