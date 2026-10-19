// Checks that window_buffer emits exactly WINDOW_SIZE beats per window, in
// chronological order, with correct AXI-Stream backpressure handling.
//
// Uses a small WINDOW_SIZE so the test runs instantly; the parameters are what
// change, not the logic.

`timescale 1ns / 1ps

module tb_window_buffer;

    localparam integer W      = 8;    // WINDOW_SIZE
    localparam integer NCH    = 6;
    localparam integer DW     = 16;
    localparam integer STRIDE = 4;    // overlapping, the harder case
    localparam integer BEAT_W = NCH * DW;

    reg clk = 0, rst_n = 0;
    always #5 clk = ~clk;

    reg  [BEAT_W-1:0] sample_in = 0;
    reg               sample_valid = 0;
    wire [BEAT_W-1:0] stream_data;
    wire              stream_valid, window_done, stream_busy;
    reg               stream_ready = 1;

    integer errors = 0;
    integer beats_this_window = 0;
    integer windows = 0;
    integer next_expected = 0;   // chronological value we expect next

    window_buffer #(.WINDOW_SIZE(W), .N_CHANNELS(NCH), .DATA_WIDTH(DW),
                    .STRIDE(STRIDE)) dut (
        .clk(clk), .rst_n(rst_n),
        .sample_in(sample_in), .sample_valid(sample_valid),
        .stream_data(stream_data), .stream_valid(stream_valid),
        .stream_ready(stream_ready),
        .window_done(window_done), .stream_busy(stream_busy));

    // Each sample carries its own sequence number, so order is checkable.
    task push;
        input integer v;
        begin
            @(negedge clk);
            sample_in = v;
            sample_valid = 1'b1;
            @(negedge clk);
            sample_valid = 1'b0;
        end
    endtask

    // Monitor the stream: values must be consecutive and W beats long.
    always @(posedge clk) begin
        if (rst_n && stream_valid && stream_ready) begin
            if (beats_this_window == 0) next_expected = stream_data;
            else if (stream_data !== next_expected) begin
                $display("  FAIL window %0d beat %0d: got %0d want %0d",
                         windows, beats_this_window, stream_data, next_expected);
                errors = errors + 1;
            end
            next_expected     = next_expected + 1;
            beats_this_window = beats_this_window + 1;
        end
        if (rst_n && window_done) begin
            if (beats_this_window !== W) begin
                $display("  FAIL window %0d: %0d beats, expected %0d",
                         windows, beats_this_window, W);
                errors = errors + 1;
            end else begin
                $display("  window %0d OK: %0d beats starting at %0d",
                         windows, beats_this_window, next_expected - W);
            end
            windows           = windows + 1;
            beats_this_window = 0;
        end
    end

    integer i;
    initial begin
        repeat (4) @(posedge clk);
        rst_n = 1;

        // Nothing should stream before the buffer is primed.
        for (i = 0; i < W - 1; i = i + 1) push(i);
        repeat (4) @(posedge clk);
        if (windows != 0) begin
            $display("  FAIL streamed before the buffer was full");
            errors = errors + 1;
        end

        // Priming sample: the first full window should go out now.
        push(W - 1);
        repeat (W + 8) @(posedge clk);

        // STRIDE more samples -> a second, overlapping window.
        for (i = W; i < W + STRIDE; i = i + 1) push(i);
        repeat (W + 8) @(posedge clk);

        // Backpressure: hold ready low mid-window and confirm nothing is lost.
        stream_ready = 0;
        for (i = W + STRIDE; i < W + 2*STRIDE; i = i + 1) push(i);
        repeat (6) @(posedge clk);
        if (windows != 2) begin
            $display("  FAIL window advanced while stream_ready was low");
            errors = errors + 1;
        end
        stream_ready = 1;
        repeat (W + 8) @(posedge clk);

        if (windows != 3) begin
            $display("  FAIL expected 3 windows, saw %0d", windows);
            errors = errors + 1;
        end

        if (errors == 0) $display("tb_window_buffer: PASS (%0d windows)", windows);
        else             $display("tb_window_buffer: FAIL (%0d error(s))", errors);
        $finish;
    end

endmodule
