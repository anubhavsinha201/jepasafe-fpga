// TODO (Week 7): sliding-window buffer feeding the inference core.
//
// Holds WINDOW_SIZE x 6 samples in a circular BRAM. Every STRIDE new sample
// sets, it streams the full window out to the hls4ml core.
//
// Keep WINDOW_SIZE in step with src/jepasafe/config.py:ModelConfig.window_size.

module window_buffer #(
    parameter WINDOW_SIZE = 64,
    parameter N_CHANNELS  = 6,
    parameter DATA_WIDTH  = 8,   // ap_fixed<8,4>
    parameter STRIDE      = 32
) (
    input  wire                          clk,
    input  wire                          rst_n,

    input  wire [N_CHANNELS*DATA_WIDTH-1:0] sample_in,
    input  wire                          sample_valid,

    output wire [DATA_WIDTH-1:0]         stream_data,
    output wire                          stream_valid,
    input  wire                          stream_ready,
    output wire                          window_done
);

    assign stream_data  = {DATA_WIDTH{1'b0}};
    assign stream_valid = 1'b0;
    assign window_done  = 1'b0;

endmodule
