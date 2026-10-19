// Sliding-window buffer feeding the hls4ml core's AXI4-Stream input.
//
// Holds WINDOW_SIZE samples of N_CHANNELS x DATA_WIDTH in a circular block RAM
// and, every STRIDE new samples, streams the whole window out oldest-first.
//
// Two things the previous stub got wrong, both of which matter:
//
//   1. The output is ONE BEAT PER TIME STEP, all channels together — 96 bits
//      for 6 x ap_fixed<16,6>, not DATA_WIDTH. The core's imu_window_TDATA is
//      96 bits wide and takes 128 beats per inference. A DATA_WIDTH-wide port
//      would simulate perfectly happily and feed the model nonsense.
//   2. WINDOW_SIZE must be 128 and DATA_WIDTH 16 to match the trained model.
//      The old defaults were 64 and 8, and jepasafe_top instantiated it with no
//      overrides at all.
//
// Timing: at 100 MHz a 50 Hz sample arrives every 2,000,000 cycles, while
// streaming a window takes WINDOW_SIZE (128) cycles and one inference ~160,000.
// So there are three orders of magnitude of slack and a write can never catch
// up with an in-progress read. `stream_busy` is exposed so a testbench can
// assert that rather than leave it as a comment.

`timescale 1ns / 1ps

module window_buffer #(
    parameter integer WINDOW_SIZE = 128,
    parameter integer N_CHANNELS  = 6,
    parameter integer DATA_WIDTH  = 16,  // ap_fixed<16,6>
    // STRIDE == WINDOW_SIZE gives back-to-back windows (a decision every
    // 2.56 s). Halve it for 50% overlap and twice the decision rate; the core
    // has the headroom.
    parameter integer STRIDE      = 128
) (
    input  wire                             clk,
    input  wire                             rst_n,

    input  wire [N_CHANNELS*DATA_WIDTH-1:0] sample_in,
    input  wire                             sample_valid,

    // AXI4-Stream master towards the core
    output wire [N_CHANNELS*DATA_WIDTH-1:0] stream_data,
    output wire                             stream_valid,
    input  wire                             stream_ready,

    output reg                              window_done,  // one pulse per window
    output wire                             stream_busy
);

    localparam integer BEAT_W = N_CHANNELS * DATA_WIDTH;
    localparam integer PTR_W  = $clog2(WINDOW_SIZE);

    reg [BEAT_W-1:0] buf_mem [0:WINDOW_SIZE-1];

    reg [PTR_W-1:0]  wptr;
    reg [PTR_W-1:0]  rptr;
    reg [PTR_W:0]    fill;        // saturates at WINDOW_SIZE
    reg [PTR_W:0]    since_last;  // samples since the last window was emitted
    reg [PTR_W:0]    beats_left;
    reg              streaming;

    assign stream_busy  = streaming;
    assign stream_valid = streaming;
    assign stream_data  = buf_mem[rptr];

    wire primed  = (fill == WINDOW_SIZE);
    wire do_beat = streaming && stream_ready;
    // Start as soon as the buffer is full and STRIDE fresh samples have landed.
    wire start   = !streaming && primed && (since_last >= STRIDE);

    always @(posedge clk) begin
        if (!rst_n) begin
            wptr        <= {PTR_W{1'b0}};
            rptr        <= {PTR_W{1'b0}};
            fill        <= {(PTR_W+1){1'b0}};
            since_last  <= {(PTR_W+1){1'b0}};
            beats_left  <= {(PTR_W+1){1'b0}};
            streaming   <= 1'b0;
            window_done <= 1'b0;
        end else begin
            window_done <= 1'b0;

            if (sample_valid) begin
                buf_mem[wptr] <= sample_in;
                wptr <= (wptr == WINDOW_SIZE-1) ? {PTR_W{1'b0}} : wptr + 1'b1;
                if (!primed)
                    fill <= fill + 1'b1;
                if (!start)
                    since_last <= since_last + 1'b1;
            end

            if (start) begin
                // wptr points at the slot due to be overwritten next, which is
                // the oldest sample — exactly where a chronological read starts.
                rptr       <= wptr;
                beats_left <= WINDOW_SIZE;
                streaming  <= 1'b1;
                since_last <= sample_valid ? {{PTR_W{1'b0}}, 1'b1}
                                           : {(PTR_W+1){1'b0}};
            end else if (do_beat) begin
                rptr <= (rptr == WINDOW_SIZE-1) ? {PTR_W{1'b0}} : rptr + 1'b1;
                if (beats_left == 1) begin
                    streaming   <= 1'b0;
                    window_done <= 1'b1;
                end
                beats_left <= beats_left - 1'b1;
            end
        end
    end

endmodule
