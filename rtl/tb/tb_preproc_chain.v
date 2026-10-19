// End-to-end check of the preprocessing path: raw sensor counts through
// sample_scale and window_buffer, compared against the exact 96-bit beats the
// hls4ml core expects.
//
// This is the part of Stage 5 that can actually be run here. The full chain
// including the core needs a 2-state simulator (Device Guard blocks xelab, and
// Vitis RTL assumes power-up-zero which Icarus does not model — see
// docs/simulation-plan.md). Everything upstream of the core is hand-written
// RTL with proper resets, so Icarus handles it, and the core itself is verified
// numerically by csim over all 2,947 windows.
//
//   python -m jepasafe.make_tb_data --out-dir sim/jepasafe_rtl counts --window 0
//   python scripts/tv_to_hex.py
//   bash scripts/sim_preproc.sh
//
// Ground truth is stim_imu_window.hex, which came out of the C testbench's own
// transaction capture — the exact payload the core would have been driven with.
//
// Tolerance: counts -> ap_fixed<16,6> rounds twice (physical units to int16 in
// the sensor, then int16 to fixed point here), so a beat can sit one LSB (2^-10)
// off the directly-converted reference. That is sensor quantisation, not an RTL
// fault. Exact matches are counted and reported separately so a real regression
// cannot hide behind the tolerance.

`timescale 1ns / 1ps

module tb_preproc_chain;

    localparam integer BEATS   = 128;
    localparam integer TOL     = 1;      // LSBs of ap_fixed<16,6>

    reg clk = 0, rst_n = 0;
    always #5 clk = ~clk;

    // Raw counts, channel 0 in the low 16 bits.
    reg [95:0] counts   [0:BEATS-1];
    // Reference beats straight from the hls4ml transaction capture. The file
    // holds all 60 transactions; only the first window is compared, but the
    // array is sized for the whole file so $readmemh does not warn.
    reg [95:0] expected [0:60*BEATS-1];

    reg signed [15:0] ax, ay, az, gx, gy, gz;
    reg               sample_valid = 0;

    wire [95:0] sample_q;
    wire        sample_q_valid;

    sample_scale u_scale (.clk(clk), .rst_n(rst_n),
        .ax(ax), .ay(ay), .az(az), .gx(gx), .gy(gy), .gz(gz),
        .sample_valid(sample_valid),
        .sample_q(sample_q), .sample_q_valid(sample_q_valid));

    wire [95:0] stream_data;
    wire        stream_valid, window_done, stream_busy;

    window_buffer #(.WINDOW_SIZE(BEATS), .N_CHANNELS(6), .DATA_WIDTH(16),
                    .STRIDE(BEATS)) u_window (
        .clk(clk), .rst_n(rst_n),
        .sample_in(sample_q), .sample_valid(sample_q_valid),
        .stream_data(stream_data), .stream_valid(stream_valid),
        .stream_ready(1'b1),
        .window_done(window_done), .stream_busy(stream_busy));

    integer errors = 0, exact = 0, within_tol = 0, beat = 0;
    integer i;

    function integer abs_diff;
        input signed [15:0] a, b;
        begin
            abs_diff = (a > b) ? (a - b) : (b - a);
        end
    endfunction

    // Compare each streamed beat lane by lane against the reference.
    always @(posedge clk) begin
        if (rst_n && stream_valid) begin
            if (stream_data === expected[beat]) begin
                exact = exact + 1;
            end else begin
                reg bad;
                bad = 1'b0;
                for (i = 0; i < 6; i = i + 1) begin
                    if (abs_diff($signed(stream_data[i*16 +: 16]),
                                 $signed(expected[beat][i*16 +: 16])) > TOL)
                        bad = 1'b1;
                end
                if (bad) begin
                    if (errors < 5)
                        $display("  FAIL beat %0d: got %h want %h",
                                 beat, stream_data, expected[beat]);
                    errors = errors + 1;
                end else begin
                    within_tol = within_tol + 1;
                end
            end
            beat = beat + 1;
        end
    end

    integer t;
    initial begin
        $readmemh("sim/jepasafe_rtl/counts_window.hex", counts);
        $readmemh("sim/jepasafe_rtl/stim_imu_window.hex", expected);

        if (counts[0] === 96'bx || expected[0] === 96'bx) begin
            $display("FATAL: vectors not loaded. From the repo root run:");
            $display("  python -m jepasafe.make_tb_data --out-dir sim/jepasafe_rtl counts");
            $display("  python scripts/tv_to_hex.py");
            $finish;
        end

        repeat (4) @(posedge clk);
        rst_n = 1;

        // Feed the window one sample at a time. Real spacing is 2,000,000
        // cycles at 50 Hz; a few cycles is enough to prove the data path and
        // keeps the test instant.
        for (t = 0; t < BEATS; t = t + 1) begin
            @(negedge clk);
            ax = counts[t][15:0];
            ay = counts[t][31:16];
            az = counts[t][47:32];
            gx = counts[t][63:48];
            gy = counts[t][79:64];
            gz = counts[t][95:80];
            sample_valid = 1'b1;
            @(negedge clk);
            sample_valid = 1'b0;
            @(negedge clk);
        end

        // Let the window stream out.
        repeat (BEATS * 4 + 50) @(posedge clk);

        $display("  beats streamed   : %0d (expected %0d)", beat, BEATS);
        $display("  bit-exact        : %0d", exact);
        $display("  within %0d LSB     : %0d", TOL, within_tol);
        $display("  out of tolerance : %0d", errors);

        if (beat != BEATS) begin
            $display("  FAIL streamed %0d beats, expected %0d", beat, BEATS);
            errors = errors + 1;
        end

        if (errors == 0)
            $display("tb_preproc_chain: PASS (%0d/%0d bit-exact)", exact, BEATS);
        else
            $display("tb_preproc_chain: FAIL (%0d beat(s) out of tolerance)", errors);
        $finish;
    end

endmodule
