// I2C master + MPU6050 driver.
//
// Wakes the device, sets the ranges the model was trained for, then burst-reads
// the 6 motion axes at SAMPLE_HZ.
//
// The ranges are not arbitrary: sample_scale.v converts counts to the model's
// ap_fixed<16,6> assuming +/-4 g and +/-500 deg/s. Changing them here without
// changing the scale factors there silently rescales every input.
//
//   ACCEL_CONFIG (0x1C) = 0x08  -> AFS_SEL = 1 -> +/-4 g,      8192 LSB/g
//   GYRO_CONFIG  (0x1B) = 0x08  -> FS_SEL  = 1 -> +/-500 deg/s, 65.5 LSB/(deg/s)
//   PWR_MGMT_1   (0x6B) = 0x00  -> wake from sleep, internal 8 MHz oscillator
//
// Burst read of 14 bytes from ACCEL_XOUT_H (0x3B), each axis big-endian:
//   0..5  ax ay az      6..7  temperature (discarded)      8..13  gx gy gz
//
// SDA is open-drain: this module never drives a 1. It asserts sda_oe to pull
// the line low and releases otherwise, so the parent does
//     assign sda = sda_oe ? 1'b0 : 1'bz;
// with an external pull-up. Driving a hard 1 onto an I2C bus is how you cook a
// transceiver when the slave is also driving.

`timescale 1ns / 1ps

module mpu6050_i2c #(
    parameter integer CLK_HZ    = 100_000_000,
    parameter integer I2C_HZ    = 100_000,
    parameter integer SAMPLE_HZ = 50,          // the model's window is 128 @ 50 Hz
    parameter [6:0]   DEV_ADDR  = 7'h68        // AD0 tied low
) (
    input  wire               clk,
    input  wire               rst_n,

    output reg                scl,
    input  wire               sda_in,
    output wire               sda_out,   // always 0; presence is via sda_oe
    output reg                sda_oe,

    output reg signed [15:0]  ax,
    output reg signed [15:0]  ay,
    output reg signed [15:0]  az,
    output reg signed [15:0]  gx,
    output reg signed [15:0]  gy,
    output reg signed [15:0]  gz,
    output reg                sample_valid,
    output reg                error        // sticky: slave did not ACK
);

    assign sda_out = 1'b0;

    // Four phases per SCL bit: SDA changes while SCL is low, the slave samples
    // while SCL is high.
    localparam integer QUARTER = (CLK_HZ / I2C_HZ) / 4;
    localparam integer SAMPLE_DIV = CLK_HZ / SAMPLE_HZ;

    reg [15:0] q_cnt;
    reg [1:0]  phase;
    reg        tick;          // one cycle at each phase boundary

    always @(posedge clk) begin
        if (!rst_n) begin
            q_cnt <= 16'd0;
            phase <= 2'd0;
            tick  <= 1'b0;
        end else if (q_cnt == QUARTER-1) begin
            q_cnt <= 16'd0;
            phase <= phase + 2'd1;
            tick  <= 1'b1;
        end else begin
            q_cnt <= q_cnt + 16'd1;
            tick  <= 1'b0;
        end
    end

    // ------------------------------------------------------------- registers
    localparam [7:0] REG_PWR  = 8'h6B,
                     REG_GYRO = 8'h1B,
                     REG_ACC  = 8'h1C,
                     REG_DATA = 8'h3B;

    // ------------------------------------------------------------ main FSM
    localparam [4:0]
        S_IDLE      = 5'd0,
        S_START     = 5'd1,
        S_SEND      = 5'd2,   // shift out shift_byte
        S_ACK_RD    = 5'd3,   // read the slave's ACK
        S_RECV      = 5'd4,   // shift in a byte
        S_ACK_WR    = 5'd5,   // drive ACK/NACK
        S_STOP      = 5'd6,
        S_WAIT      = 5'd7;

    reg [4:0] state, ret_state;
    reg [7:0] shift_byte;
    reg [2:0] bit_idx;
    reg       ack_fail;

    // Which transaction we are in the middle of.
    localparam [2:0]
        T_WAKE = 3'd0, T_GYRO = 3'd1, T_ACC = 3'd2, T_READ = 3'd3, T_RUN = 3'd4;
    reg [2:0] txn;
    reg [3:0] step;          // position within a transaction
    reg [3:0] byte_idx;      // which of the 14 data bytes
    reg [7:0] data [0:13];

    reg [31:0] sample_timer;
    reg        send_ack;     // value to drive during S_ACK_WR

    always @(posedge clk) begin
        if (!rst_n) begin
            state        <= S_IDLE;
            ret_state    <= S_IDLE;
            scl          <= 1'b1;
            sda_oe       <= 1'b0;
            txn          <= T_WAKE;
            step         <= 4'd0;
            bit_idx      <= 3'd7;
            byte_idx     <= 4'd0;
            sample_valid <= 1'b0;
            error        <= 1'b0;
            ack_fail     <= 1'b0;
            sample_timer <= 32'd0;
            {ax, ay, az, gx, gy, gz} <= 96'd0;
        end else begin
            sample_valid <= 1'b0;

            if (sample_timer != 0) sample_timer <= sample_timer - 32'd1;

            if (tick) begin
                case (state)
                // ------------------------------------------------- idle/pace
                S_IDLE: begin
                    scl    <= 1'b1;
                    sda_oe <= 1'b0;
                    // Leave only at a phase boundary. Every other state is
                    // written assuming it is entered at phase 0, which holds
                    // for a transaction that follows S_STOP but NOT for the
                    // first one after reset, where `phase` is wherever the
                    // divider happens to be. Entering S_START mid-bit makes it
                    // execute its phase-3 case first and skip generating the
                    // START condition entirely, so the slave never sees the
                    // transaction and NACKs the address.
                    if (phase == 2'd3 && (txn != T_RUN || sample_timer == 0)) begin
                        step       <= 4'd0;
                        // Every transaction opens by addressing the device for
                        // a write. Loading it here rather than inside S_START
                        // lets S_START be reused verbatim for the repeated
                        // START before a read, which needs a different byte.
                        shift_byte <= {DEV_ADDR, 1'b0};
                        bit_idx    <= 3'd7;
                        state      <= S_START;
                        if (txn == T_RUN) txn <= T_READ;
                    end
                end

                // START: SDA falls while SCL is high. Phase 0 raises SCL and
                // releases SDA first, so this works both from idle (already
                // high) and as a repeated START straight out of an ACK, where
                // SCL is low.
                S_START: begin
                    case (phase)
                        2'd0: begin scl <= 1'b1; sda_oe <= 1'b0; end
                        2'd1: sda_oe <= 1'b1;          // SDA -> 0, SCL high
                        2'd2: scl    <= 1'b0;
                        2'd3: state  <= S_SEND;
                    endcase
                end

                // ------------------------------------------------- send byte
                S_SEND: begin
                    case (phase)
                        2'd0: begin
                            scl    <= 1'b0;
                            sda_oe <= ~shift_byte[bit_idx];  // drive low for 0
                        end
                        2'd1: scl <= 1'b1;
                        2'd2: ;
                        2'd3: begin
                            scl <= 1'b0;
                            if (bit_idx == 3'd0) begin
                                sda_oe <= 1'b0;              // release for ACK
                                state  <= S_ACK_RD;
                            end else begin
                                bit_idx <= bit_idx - 3'd1;
                            end
                        end
                    endcase
                end

                S_ACK_RD: begin
                    case (phase)
                        2'd0: begin scl <= 1'b0; sda_oe <= 1'b0; end
                        2'd1: scl <= 1'b1;
                        2'd2: if (sda_in) ack_fail <= 1'b1;   // 1 = NACK
                        2'd3: begin
                            scl  <= 1'b0;
                            step <= step + 4'd1;
                            state <= S_WAIT;
                        end
                    endcase
                end

                // ------------------------------------------------- recv byte
                S_RECV: begin
                    case (phase)
                        2'd0: begin scl <= 1'b0; sda_oe <= 1'b0; end
                        2'd1: scl <= 1'b1;
                        2'd2: shift_byte <= {shift_byte[6:0], sda_in};
                        2'd3: begin
                            scl <= 1'b0;
                            if (bit_idx == 3'd0) begin
                                // ACK every byte except the last.
                                send_ack <= (byte_idx != 4'd13);
                                state    <= S_ACK_WR;
                            end else begin
                                bit_idx <= bit_idx - 3'd1;
                            end
                        end
                    endcase
                end

                S_ACK_WR: begin
                    case (phase)
                        2'd0: begin scl <= 1'b0; sda_oe <= send_ack; end
                        2'd1: scl <= 1'b1;
                        2'd2: ;
                        2'd3: begin
                            scl            <= 1'b0;
                            sda_oe         <= 1'b0;
                            data[byte_idx] <= shift_byte;
                            if (byte_idx == 4'd13) begin
                                state <= S_STOP;
                            end else begin
                                byte_idx <= byte_idx + 4'd1;
                                bit_idx  <= 3'd7;
                                state    <= S_RECV;
                            end
                        end
                    endcase
                end

                // ------------------------------------------------------ stop
                S_STOP: begin
                    case (phase)
                        2'd0: begin scl <= 1'b0; sda_oe <= 1'b1; end
                        2'd1: scl <= 1'b1;
                        2'd2: sda_oe <= 1'b0;          // SDA 0->1 with SCL high
                        2'd3: begin
                            error <= error | ack_fail;
                            if (txn == T_READ) begin
                                // Big-endian pairs; bytes 6..7 are temperature.
                                ax <= {data[0],  data[1]};
                                ay <= {data[2],  data[3]};
                                az <= {data[4],  data[5]};
                                gx <= {data[8],  data[9]};
                                gy <= {data[10], data[11]};
                                gz <= {data[12], data[13]};
                                sample_valid <= 1'b1;
                                sample_timer <= SAMPLE_DIV;
                                txn          <= T_RUN;
                            end else begin
                                txn <= txn + 3'd1;
                            end
                            ack_fail <= 1'b0;
                            state    <= S_IDLE;
                        end
                    endcase
                end

                // --------------------------------- transaction sequencing
                // Reached after each ACK: decide what the next byte is.
                S_WAIT: begin
                    if (phase == 2'd3) begin
                        bit_idx <= 3'd7;
                        case (txn)
                        T_WAKE, T_GYRO, T_ACC: begin
                            if (step == 4'd1) begin
                                shift_byte <= (txn == T_WAKE) ? REG_PWR :
                                              (txn == T_GYRO) ? REG_GYRO : REG_ACC;
                                state <= S_SEND;
                            end else if (step == 4'd2) begin
                                shift_byte <= (txn == T_WAKE) ? 8'h00 : 8'h08;
                                state <= S_SEND;
                            end else begin
                                state <= S_STOP;
                            end
                        end
                        T_READ: begin
                            if (step == 4'd1) begin
                                shift_byte <= REG_DATA;
                                state <= S_SEND;
                            end else if (step == 4'd2) begin
                                // Repeated START, then re-address for reading.
                                // S_START no longer clobbers shift_byte, so
                                // this byte survives into S_SEND.
                                shift_byte <= {DEV_ADDR, 1'b1};
                                state <= S_START;
                            end else begin
                                byte_idx <= 4'd0;
                                state    <= S_RECV;
                            end
                        end
                        default: state <= S_STOP;
                        endcase
                    end
                end

                default: state <= S_IDLE;
                endcase
            end
        end
    end

endmodule
