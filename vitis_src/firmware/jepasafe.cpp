#include <iostream>

#include "jepasafe.h"
#include "parameters.h"


void jepasafe(
    hls::stream<input_t> &imu_window,
    hls::stream<result_t> &layer14_out
) {

    // hls-fpga-machine-learning insert IO
    #pragma HLS INTERFACE axis port=imu_window,layer14_out 
    #pragma HLS DATAFLOW

    // hls-fpga-machine-learning insert load weights
#ifndef __SYNTHESIS__
    static bool loaded_weights = false;
    if (!loaded_weights) {
        nnet::load_weights_from_txt<conv1d_weight_t, 480>(w2, "w2.txt");
        nnet::load_weights_from_txt<conv1d_bias_t, 16>(b2, "b2.txt");
        nnet::load_weights_from_txt<conv1d_1_weight_t, 1280>(w5, "w5.txt");
        nnet::load_weights_from_txt<conv1d_1_bias_t, 16>(b5, "b5.txt");
        nnet::load_weights_from_txt<lstm_weight_t, 1024>(w10, "w10.txt");
        nnet::load_weights_from_txt<lstm_recurrent_weight_t, 1024>(wr10, "wr10.txt");
        nnet::load_weights_from_txt<lstm_bias_t, 64>(b10, "b10.txt");
        nnet::load_weights_from_txt<lstm_recurrent_bias_t, 64>(br10, "br10.txt");
        nnet::load_weights_from_txt<dense_weight_t, 256>(w12, "w12.txt");
        nnet::load_weights_from_txt<dense_bias_t, 16>(b12, "b12.txt");
        nnet::load_weights_from_txt<logits_weight_t, 96>(w14, "w14.txt");
        nnet::load_weights_from_txt<logits_bias_t, 6>(b14, "b14.txt");
        loaded_weights = true;    }
#endif
    // ****************************************
    // NETWORK INSTANTIATION
    // ****************************************

    // hls-fpga-machine-learning insert layers

    hls::stream<conv1d_result_t> layer2_out("layer2_out");
    #pragma HLS STREAM variable=layer2_out depth=124

    hls::stream<layer4_t> layer4_out("layer4_out");
    #pragma HLS STREAM variable=layer4_out depth=124

    hls::stream<conv1d_1_result_t> layer5_out("layer5_out");
    #pragma HLS STREAM variable=layer5_out depth=120

    hls::stream<layer7_t> layer7_out("layer7_out");
    #pragma HLS STREAM variable=layer7_out depth=120

    hls::stream<layer8_t> layer8_out("layer8_out");
    #pragma HLS STREAM variable=layer8_out depth=60

    hls::stream<layer10_t> layer10_out("layer10_out");
    #pragma HLS STREAM variable=layer10_out depth=1

    hls::stream<dense_result_t> layer12_out("layer12_out");
    #pragma HLS STREAM variable=layer12_out depth=1

    hls::stream<layer13_t> layer13_out("layer13_out");
    #pragma HLS STREAM variable=layer13_out depth=1

    nnet::conv_1d_cl<input_t, conv1d_result_t, config2>(imu_window, layer2_out, w2, b2); // conv1d

    nnet::relu<conv1d_result_t, layer4_t, relu_config4>(layer2_out, layer4_out); // activation

    nnet::conv_1d_cl<layer4_t, conv1d_1_result_t, config5>(layer4_out, layer5_out, w5, b5); // conv1d_1

    nnet::relu<conv1d_1_result_t, layer7_t, relu_config7>(layer5_out, layer7_out); // activation_1

    nnet::pooling1d_cl<layer7_t, layer8_t, config8>(layer7_out, layer8_out); // max_pooling1d

    nnet::lstm_stack<layer8_t, layer10_t, config10>(layer8_out, layer10_out, w10, wr10, b10, br10); // lstm

    nnet::dense<layer10_t, dense_result_t, config12>(layer10_out, layer12_out, w12, b12); // dense

    nnet::relu<dense_result_t, layer13_t, relu_config13>(layer12_out, layer13_out); // dense_relu

    nnet::dense<layer13_t, result_t, config14>(layer13_out, layer14_out, w14, b14); // logits

}

