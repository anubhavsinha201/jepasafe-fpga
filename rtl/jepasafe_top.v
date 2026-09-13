// Top level: sensor -> window -> inference -> alert.
//
// Wiring is in place; the sensor, window and inference blocks are still stubs.
// The `myproject` instance below is the RTL hls4ml emits into hls/jepasafe/ —
// add that file to the Gowin project and rename the instance to match.

module jepasafe_top (
    input  wire       sys_clk,     // 27 MHz
    input  wire       sys_rst_n,   // active low

    // MPU6050 - I2C
    output wire       mpu_scl,
    inout  wire       mpu_sda,

    // HC-08 - UART
    output wire       ble_tx,
    input  wire       ble_rx,

    // Onboard LEDs (active low)
    output wire [5:0] led
);

    localparam DATA_WIDTH = 8;
    // Score >= 205/255 ~ 0.8 -> CLAUDE.md alert threshold
    localparam [DATA_WIDTH-1:0] ALERT_THRESHOLD = 8'd205;

    // ---------------------------------------------------------------- sensor
    wire sda_in, sda_out, sda_oe;
    wire signed [15:0] ax, ay, az, gx, gy, gz;
    wire sample_valid;

    assign mpu_sda = sda_oe ? sda_out : 1'bz;
    assign sda_in  = mpu_sda;

    mpu6050_i2c u_sensor (
        .clk          (sys_clk),
        .rst_n        (sys_rst_n),
        .scl          (mpu_scl),
        .sda_in       (sda_in),
        .sda_out      (sda_out),
        .sda_oe       (sda_oe),
        .ax           (ax), .ay(ay), .az(az),
        .gx           (gx), .gy(gy), .gz(gz),
        .sample_valid (sample_valid),
        .error        ()
    );

    // ---------------------------------------------------------------- window
    // TODO: scale int16 sensor counts into ap_fixed<8,4> before buffering.
    wire [6*DATA_WIDTH-1:0] sample_q = {6*DATA_WIDTH{1'b0}};

    wire [DATA_WIDTH-1:0] stream_data;
    wire stream_valid, window_done;

    window_buffer u_window (
        .clk          (sys_clk),
        .rst_n        (sys_rst_n),
        .sample_in    (sample_q),
        .sample_valid (sample_valid),
        .stream_data  (stream_data),
        .stream_valid (stream_valid),
        .stream_ready (1'b1),
        .window_done  (window_done)
    );

    // ------------------------------------------------------------- inference
    // TODO: instantiate the hls4ml core from hls/jepasafe/.
    wire [DATA_WIDTH-1:0] anomaly_score = {DATA_WIDTH{1'b0}};
    wire                  score_valid   = 1'b0;

    // ----------------------------------------------------------------- alert
    reg alert;
    always @(posedge sys_clk or negedge sys_rst_n) begin
        if (!sys_rst_n)
            alert <= 1'b0;
        else if (score_valid)
            alert <= (anomaly_score >= ALERT_THRESHOLD);
    end

    assign led = alert ? 6'b000000 : 6'b111111;  // all LEDs on when alerting

    // ------------------------------------------------------------------ BLE
    uart_tx #(.CLK_HZ(27_000_000), .BAUD_RATE(9600)) u_ble (
        .clk   (sys_clk),
        .rst_n (sys_rst_n),
        .data  (anomaly_score),
        .valid (score_valid & alert),
        .tx    (ble_tx),
        .busy  ()
    );

endmodule
