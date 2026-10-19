#ifndef JEPASAFE_H_
#define JEPASAFE_H_

#include "ap_fixed.h"
#include "ap_int.h"
#include "hls_stream.h"

#include "defines.h"


// Prototype of top level function for C-synthesis
void jepasafe(
    hls::stream<input_t> &imu_window,
    hls::stream<result_t> &layer14_out
);

// hls-fpga-machine-learning insert emulator-defines


#endif
