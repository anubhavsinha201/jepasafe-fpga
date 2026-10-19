// AXI4-Stream pass-through that inserts TLAST every BEATS transfers.
//
// The hls4ml core emits layer14_out with TDATA/TVALID/TREADY only — no TLAST,
// because hls4ml does not use ap_axiu side-channel types. AXI DMA's S2MM
// channel uses TLAST to delimit packets, so without it the receive transfer
// never reports completion.
//
// One inference produces 384 bits of logits. After the 48->4 byte width
// converter that is 12 beats of 32 bit, so BEATS=12 marks each inference as
// one packet.
//
// Port names follow the s_axis_*/m_axis_* convention so Vivado's IP Integrator
// infers both AXI4-Stream interfaces automatically when the module is added to
// a block design.
//
// Place between axis_dwidth_converter_1/M_AXIS and axi_dma_0/S_AXIS_S2MM.

`timescale 1ns / 1ps

module axis_tlast_gen #(
    parameter integer DATA_WIDTH = 32,
    parameter integer BEATS      = 12
) (
    input  wire                    aclk,
    input  wire                    aresetn,

    input  wire [DATA_WIDTH-1:0]   s_axis_tdata,
    input  wire                    s_axis_tvalid,
    output wire                    s_axis_tready,

    output wire [DATA_WIDTH-1:0]   m_axis_tdata,
    output wire                    m_axis_tvalid,
    input  wire                    m_axis_tready,
    output wire                    m_axis_tlast
);

    // $clog2(12) = 4, enough to count 0..11
    localparam integer CNT_W = (BEATS <= 1) ? 1 : $clog2(BEATS);

    reg [CNT_W-1:0] beat;

    // Combinational pass-through: this block adds TLAST and nothing else, so
    // it must not break the ready/valid handshake or add a cycle of latency.
    assign m_axis_tdata  = s_axis_tdata;
    assign m_axis_tvalid = s_axis_tvalid;
    assign s_axis_tready = m_axis_tready;
    assign m_axis_tlast  = (beat == BEATS - 1);

    wire handshake = s_axis_tvalid & m_axis_tready;

    always @(posedge aclk) begin
        if (!aresetn)
            beat <= {CNT_W{1'b0}};
        else if (handshake)
            beat <= (beat == BEATS - 1) ? {CNT_W{1'b0}} : beat + 1'b1;
    end

endmodule
