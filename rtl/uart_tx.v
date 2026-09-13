// 8N1 UART transmitter. Drives the HC-08 BLE module (and the onboard USB
// serial bridge) so alerts and debug traces can leave the board.

module uart_tx #(
    parameter CLK_HZ    = 27_000_000,
    parameter BAUD_RATE = 9600
) (
    input  wire       clk,
    input  wire       rst_n,
    input  wire [7:0] data,
    input  wire       valid,   // pulse high for one cycle to send `data`
    output reg        tx,
    output wire       busy
);

    localparam integer DIVISOR = CLK_HZ / BAUD_RATE;

    localparam IDLE = 2'd0, START = 2'd1, DATA = 2'd2, STOP = 2'd3;

    reg [1:0]  state;
    reg [15:0] tick;
    reg [2:0]  bit_idx;
    reg [7:0]  shifter;

    assign busy = (state != IDLE);

    wire tick_done = (tick == DIVISOR - 1);

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state   <= IDLE;
            tx      <= 1'b1;   // line idles high
            tick    <= 16'd0;
            bit_idx <= 3'd0;
            shifter <= 8'd0;
        end else begin
            case (state)
                IDLE: begin
                    tx   <= 1'b1;
                    tick <= 16'd0;
                    if (valid) begin
                        shifter <= data;
                        state   <= START;
                    end
                end

                START: begin
                    tx   <= 1'b0;  // start bit
                    tick <= tick_done ? 16'd0 : tick + 1'b1;
                    if (tick_done) begin
                        bit_idx <= 3'd0;
                        state   <= DATA;
                    end
                end

                DATA: begin
                    tx   <= shifter[0];
                    tick <= tick_done ? 16'd0 : tick + 1'b1;
                    if (tick_done) begin
                        shifter <= {1'b0, shifter[7:1]};  // LSB first
                        if (bit_idx == 3'd7)
                            state <= STOP;
                        else
                            bit_idx <= bit_idx + 1'b1;
                    end
                end

                STOP: begin
                    tx   <= 1'b1;  // stop bit
                    tick <= tick_done ? 16'd0 : tick + 1'b1;
                    if (tick_done)
                        state <= IDLE;
                end
            endcase
        end
    end

endmodule
