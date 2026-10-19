// Behavioural MPU6050 I2C slave, enough to verify mpu6050_i2c.v.
//
// This tests the master's logic, not the bus: it will not catch electrical
// faults, rise-time problems or 3.3 V level mistakes. It does catch the bugs
// that actually bite in RTL — wrong register sequence, a missing repeated
// START, dropped ACKs, and byte-pairing or endianness errors.
//
// Models: 7-bit addressing, a writable register file, auto-incrementing
// sequential reads from the pointer set by the preceding write, and ACK for its
// own address only. Registers 0x3B.. return the values loaded via the
// `set_axis` task, big-endian, so the testbench can check round-tripping.

`timescale 1ns / 1ps

module mpu6050_model #(
    parameter [6:0]   DEV_ADDR = 7'h68,
    parameter integer VERBOSE  = 0
) (
    input  wire scl,
    input  wire sda_master_oe,   // master pulling the line low
    output wire sda_to_master    // the wired-AND line value
);

    reg slave_pull = 1'b0;
    // Open-drain wired-AND: the line is low if anyone pulls it, else the
    // pull-up wins.
    assign sda_to_master = ~(sda_master_oe | slave_pull);

    wire sda_line = sda_to_master;

    reg [7:0] regfile [0:127];
    reg [7:0] ptr = 8'h00;

    // Observed register writes, so the testbench can assert the init sequence.
    reg [7:0] wrote_pwr  = 8'hFF;
    reg [7:0] wrote_gyro = 8'hFF;
    reg [7:0] wrote_acc  = 8'hFF;
    integer   n_bursts = 0;

    task set_axis;   // load a big-endian 16-bit value at `base`
        input [7:0] base;
        input signed [15:0] v;
        begin
            regfile[base]     = v[15:8];
            regfile[base + 1] = v[7:0];
        end
    endtask

    localparam [2:0] IDLE = 3'd0, ADDR = 3'd1, ACK_A = 3'd2,
                     WR = 3'd3, ACK_W = 3'd4, RD = 3'd5, ACK_R = 3'd6;

    reg [2:0]  st = IDLE;
    reg [7:0]  sh = 8'd0;
    reg [3:0]  nbits = 4'd0;
    reg        is_read = 1'b0;
    reg        first_byte_after_addr = 1'b0;
    reg [7:0]  rd_sh = 8'd0;

    reg scl_d = 1'b1, sda_d = 1'b1;

    // START / STOP are SDA transitions while SCL is high.
    always @(sda_line or scl) begin
        if (scl && scl_d) begin
            if (sda_d && !sda_line) begin          // falling -> START
                st      = ADDR;
                nbits   = 4'd0;
                sh      = 8'd0;
                is_read = 1'b0;
            end else if (!sda_d && sda_line) begin // rising -> STOP
                st = IDLE;
                slave_pull = 1'b0;
            end
        end
        sda_d = sda_line;
    end

    // Shift in on the rising edge of SCL, change our output on the falling edge.
    always @(posedge scl) begin
        scl_d <= 1'b1;
        case (st)
            ADDR, WR: begin
                sh    <= {sh[6:0], sda_line};
                nbits <= nbits + 4'd1;
            end
            default: ;
        endcase
    end

    always @(negedge scl) begin
        scl_d <= 1'b0;
        case (st)
        ADDR: if (nbits == 4'd8) begin
            if (sh[7:1] == DEV_ADDR) begin
                is_read    <= sh[0];
                slave_pull <= 1'b1;             // ACK
                st         <= ACK_A;
            end else begin
                slave_pull <= 1'b0;             // NACK: not us
                st         <= IDLE;
            end
            nbits <= 4'd0;
        end
        ACK_A: begin
            slave_pull <= 1'b0;
            if (is_read) begin
                rd_sh      <= regfile[ptr[6:0]];
                slave_pull <= ~regfile[ptr[6:0]][7];
                nbits      <= 4'd1;
                st         <= RD;
                if (ptr == 8'h3B) n_bursts <= n_bursts + 1;
                ptr        <= ptr + 8'd1;
            end else begin
                first_byte_after_addr <= 1'b1;
                nbits <= 4'd0;
                st    <= WR;
            end
        end
        WR: if (nbits == 4'd8) begin
            if (first_byte_after_addr) begin
                ptr                   <= sh;    // register pointer
                first_byte_after_addr <= 1'b0;
                if (VERBOSE) $display("    [model] ptr <= 0x%02h", sh);
            end else begin
                regfile[ptr[6:0]] <= sh;
                if (VERBOSE) $display("    [model] reg[0x%02h] <= 0x%02h", ptr, sh);
                case (ptr)
                    8'h6B: wrote_pwr  <= sh;
                    8'h1B: wrote_gyro <= sh;
                    8'h1C: wrote_acc  <= sh;
                    default: ;
                endcase
                ptr <= ptr + 8'd1;
            end
            slave_pull <= 1'b1;                 // ACK
            st         <= ACK_W;
            nbits      <= 4'd0;
        end
        ACK_W: begin
            slave_pull <= 1'b0;
            st         <= WR;
            nbits      <= 4'd0;
        end
        RD: begin
            if (nbits == 4'd8) begin
                slave_pull <= 1'b0;             // release for master's ACK
                st         <= ACK_R;
            end else begin
                slave_pull <= ~rd_sh[7 - nbits];
                nbits      <= nbits + 4'd1;
            end
        end
        ACK_R: begin
            // The master NACKs the final byte; either way serve the next one.
            rd_sh      <= regfile[ptr[6:0]];
            slave_pull <= ~regfile[ptr[6:0]][7];
            ptr        <= ptr + 8'd1;
            nbits      <= 4'd1;
            st         <= RD;
        end
        default: slave_pull <= 1'b0;
        endcase
    end

endmodule
