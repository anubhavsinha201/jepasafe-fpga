#ifndef DEFINES_H_
#define DEFINES_H_

#include "ap_fixed.h"
#include "ap_int.h"
#include "nnet_utils/nnet_types.h"
#include <array>
#include <cstddef>
#include <cstdio>
#include <tuple>
#include <tuple>


// hls-fpga-machine-learning insert numbers

// hls-fpga-machine-learning insert layer-precision
typedef nnet::array<ap_fixed<16,6>, 6*1> input_t;
typedef ap_fixed<16,6> model_default_t;
typedef nnet::array<ap_fixed<38,18>, 16*1> conv1d_result_t;
typedef ap_fixed<16,6> conv1d_weight_t;
typedef ap_fixed<16,6> conv1d_bias_t;
typedef nnet::array<ap_fixed<16,6>, 16*1> layer4_t;
typedef ap_fixed<18,8> activation_table_t;
typedef nnet::array<ap_fixed<40,20>, 16*1> conv1d_1_result_t;
typedef ap_fixed<16,6> conv1d_1_weight_t;
typedef ap_fixed<16,6> conv1d_1_bias_t;
typedef nnet::array<ap_fixed<16,6>, 16*1> layer7_t;
typedef ap_fixed<18,8> activation_1_table_t;
typedef nnet::array<ap_fixed<16,6>, 16*1> layer8_t;
typedef nnet::array<ap_fixed<16,6>, 16*1> layer10_t;
typedef ap_fixed<16,6> lstm_weight_t;
typedef ap_fixed<16,6> lstm_recurrent_weight_t;
typedef ap_fixed<16,6> lstm_bias_t;
typedef ap_fixed<16,6> lstm_recurrent_bias_t;
typedef ap_fixed<18,8> lstm_table_t;
typedef ap_uint<1> layer10_index;
typedef nnet::array<ap_fixed<37,17>, 16*1> dense_result_t;
typedef ap_fixed<16,6> dense_weight_t;
typedef ap_fixed<16,6> dense_bias_t;
typedef ap_uint<1> layer12_index;
typedef nnet::array<ap_fixed<16,6>, 16*1> layer13_t;
typedef ap_fixed<18,8> dense_relu_table_t;
typedef nnet::array<ap_fixed<37,17>, 6*1> result_t;
typedef ap_fixed<16,6> logits_weight_t;
typedef ap_fixed<16,6> logits_bias_t;
typedef ap_uint<1> layer14_index;

// hls-fpga-machine-learning insert emulator-defines


#endif
