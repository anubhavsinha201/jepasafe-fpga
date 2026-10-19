// RTL-vs-C check for the hls4ml core, under Icarus Verilog.
//
// This is the substitute for Vitis cosim, which cannot run here: Device Guard
// blocks xelab.exe (see docs/simulation-plan.md). It works because the
// generated core is 66 files of plain Verilog-2001 with no vendor primitives,
// so Icarus elaborates it cleanly. hls4ml's own cosim harness does not port —
// it is SystemVerilog and wants Xilinx simulation libraries — but its *data*
// does, via scripts/tv_to_hex.py.
//
//   python scripts/tv_to_hex.py
//   bash scripts/sim_rtl.sh              # all 60 transactions
//   bash scripts/sim_rtl.sh 3            # just the first 3, to prove the flow
//
// Pass criterion: every logit the RTL emits equals the logit the C model
// emitted, exactly. Both run the same fixed-point arithmetic, so "close" is a
// bug, not a tolerance.
//
// The core is driven the way the plan says to drive it on hardware: ap_start
// tied high, free-running, consuming whatever arrives on the stream. There is
// no AXI-Lite control port to do anything else with.

`timescale 1ns / 1ps

module tb_jepasafe_core;

    localparam integer BEATS      = 128;   // 96-bit beats per inference
    localparam integer MAX_TX     = 60;    // transactions in the vector files
    localparam integer LANE_W     = 64;    // each logit padded to 64 bit
    localparam integer FIELD_W    = 37;    // ap_fixed<37,17>
    localparam integer FRAC_BITS  = 20;    // 37 - 17
    localparam integer N_CLASSES  = 6;

    // One inference is ~160k cycles, so 60 of them is ~10M. Give it room, but
    // not so much that a deadlock hangs the run forever.
    localparam integer TIMEOUT_CYCLES = 400_000_000;

    reg clk = 1'b0;
    reg rst_n = 1'b0;

    always #5 clk = ~clk;   // 100 MHz, the clock the design was synthesised for

    // ------------------------------------------------------------- vectors
    reg [95:0]  stim [0:MAX_TX*BEATS-1];
    reg [383:0] expected [0:MAX_TX-1];

    integer n_tx;           // how many transactions to actually run
    integer beats_total;

    // ------------------------------------------------------------- DUT I/O
    wire [95:0]  imu_window_TDATA;
    wire         imu_window_TVALID;
    wire         imu_window_TREADY;

    wire [383:0] layer14_out_TDATA;
    wire         layer14_out_TVALID;
    reg          layer14_out_TREADY;

    wire ap_done, ap_ready, ap_idle;

    jepasafe dut (
        .imu_window_TDATA  (imu_window_TDATA),
        .imu_window_TVALID (imu_window_TVALID),
        .imu_window_TREADY (imu_window_TREADY),
        .layer14_out_TDATA (layer14_out_TDATA),
        .layer14_out_TVALID(layer14_out_TVALID),
        .layer14_out_TREADY(layer14_out_TREADY),
        .ap_clk            (clk),
        .ap_rst_n          (rst_n),
        .ap_start          (1'b1),      // free-running; no AXI-Lite to start it
        .ap_done           (ap_done),
        .ap_ready          (ap_ready),
        .ap_idle           (ap_idle)
    );

    // ------------------------------------------------------- input driver
    integer in_ptr = 0;

    // TDATA/TVALID are combinational functions of in_ptr, which is the only
    // correct shape for an AXI-Stream source. Registering them off the same
    // block that advances in_ptr reads the pre-update pointer under
    // non-blocking assignment, so each beat is held an extra cycle while the
    // handshake consumes one per cycle — the stream silently shifts.
    assign imu_window_TVALID = rst_n && (in_ptr < beats_total);
    assign imu_window_TDATA  = stim[in_ptr];

    always @(posedge clk) begin
        if (!rst_n)
            in_ptr <= 0;
        // Advance only on a completed handshake. The core stalls hard — its
        // interval is ~160k cycles while this driver would push a beat every
        // cycle — so TREADY is not a formality here.
        else if (imu_window_TVALID && imu_window_TREADY)
            in_ptr <= in_ptr + 1;
    end

    // ------------------------------------------------------ output monitor
    integer got_tx   = 0;
    integer mismatch = 0;
    integer i;

    reg signed [FIELD_W-1:0] got_f, exp_f;
    reg [LANE_W-1:0] got_lane, exp_lane;
    reg tx_bad;

    // Sign-extended 37-bit field out of lane `k` of a 384-bit word. Comparing
    // the fields rather than the raw 384 bits keeps the check immune to how
    // each side pads the unused 27 bits of every 64-bit lane.
    function signed [FIELD_W-1:0] field;
        input [383:0] word;
        input integer k;
        begin
            field = word[k*LANE_W +: FIELD_W];
        end
    endfunction

    task real_of;   // print a field as a decimal, for human-readable logs
        input signed [FIELD_W-1:0] f;
        begin
            $write("%0d.%0d", f / (1 << FRAC_BITS), f % (1 << FRAC_BITS));
        end
    endtask

    always @(posedge clk) begin
        if (rst_n && layer14_out_TVALID && layer14_out_TREADY) begin
            tx_bad = 1'b0;
            for (i = 0; i < N_CLASSES; i = i + 1) begin
                got_f = field(layer14_out_TDATA, i);
                exp_f = field(expected[got_tx], i);
                if (got_f !== exp_f) begin
                    tx_bad = 1'b1;
                    $display("  MISMATCH tx %0d logit %0d: rtl %0d  c %0d  (raw %h vs %h)",
                             got_tx, i, got_f, exp_f,
                             layer14_out_TDATA[i*LANE_W +: LANE_W],
                             expected[got_tx][i*LANE_W +: LANE_W]);
                end
            end
            if (tx_bad) mismatch = mismatch + 1;
            else $display("  tx %0d OK   (logit0 raw %h)",
                          got_tx, layer14_out_TDATA[0 +: LANE_W]);
            got_tx = got_tx + 1;
        end
    end

    // --------------------------------------------------------------- main
    integer cycles = 0;
    always @(posedge clk) cycles = cycles + 1;

    initial begin
        if (!$value$plusargs("ntx=%d", n_tx)) n_tx = MAX_TX;
        if (n_tx > MAX_TX) n_tx = MAX_TX;
        beats_total = n_tx * BEATS;

        $readmemh("sim/jepasafe_rtl/stim_imu_window.hex", stim);
        $readmemh("sim/jepasafe_rtl/expected_logits.hex", expected);

        // A missing or misnamed vector file leaves these arrays all-X, and the
        // run would look like a total mismatch rather than a setup error.
        if (stim[0] === 96'bx || expected[0] === 384'bx) begin
            $display("FATAL: vector files not loaded. Run from the repo root, after:");
            $display("       python scripts/tv_to_hex.py");
            $finish;
        end

        $display("=== tb_jepasafe_core: %0d transaction(s), %0d beats ===",
                 n_tx, beats_total);

        layer14_out_TREADY = 1'b1;   // never backpressure the output
        repeat (20) @(posedge clk);
        rst_n = 1'b1;

        wait (got_tx == n_tx);
        repeat (10) @(posedge clk);

        $display("=== %0d/%0d transactions matched the C model, %0d mismatched ===",
                 n_tx - mismatch, n_tx, mismatch);
        $display("=== %0d cycles (%0d per inference) ===", cycles, cycles / n_tx);
        if (mismatch == 0)
            $display("RTL-vs-C: PASS (bit-exact)");
        else
            $display("RTL-vs-C: FAIL");
        $finish;
    end

    initial begin
        wait (cycles > TIMEOUT_CYCLES);
        $display("TIMEOUT after %0d cycles with %0d/%0d transactions out.",
                 cycles, got_tx, n_tx);
        $display("in_ptr=%0d/%0d  ap_idle=%b ap_done=%b", in_ptr, beats_total,
                 ap_idle, ap_done);
        $display("RTL-vs-C: FAIL (timeout)");
        $finish;
    end

endmodule
