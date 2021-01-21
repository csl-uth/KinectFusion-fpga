/*

 Copyright (c) 2021 Computer Systems Lab - University of Thessaly

 This code is licensed under the MIT License.

*/


#define _HW_

// HW implementation of the Bilateral Filter kernel
// enabled optimizations: BF_Unroll, BF_Pad, BF_Pipe, BF_3x3

#include <kernels.hpp>
#include <string.h>

#define N 322
#define PAD_SZ 2
#define FILTER_SZ 9

inline
float exp1(float x) {
  x = 1.0 + x / 256.0;
  x *= x; x *= x; x *= x; x *= x;
  x *= x; x *= x; x *= x; x *= x;
  return x;
}

extern "C" {


void bilateralFilterKernel(float* out, float* pad_depth, uint size_x,uint size_y,
		const float * gaussian, int r,int start,int end) {


	#pragma HLS INTERFACE s_axilite port=return bundle=control
	#pragma HLS INTERFACE m_axi port=out offset=slave bundle=out max_read_burst_length=256 max_write_burst_length=256
	#pragma HLS INTERFACE s_axilite port=out	bundle=control
	#pragma HLS INTERFACE m_axi port=pad_depth offset=slave bundle=pad_depth max_read_burst_length=256 max_write_burst_length=256
	#pragma HLS INTERFACE s_axilite port=pad_depth	bundle=control
	#pragma HLS INTERFACE m_axi port=gaussian offset=slave bundle=gaussian
	#pragma HLS INTERFACE s_axilite port=gaussian	bundle=control
	#pragma HLS INTERFACE s_axilite port=size_x bundle=control
	#pragma HLS INTERFACE s_axilite port=size_y bundle=control
	#pragma HLS INTERFACE s_axilite port=r bundle=control
	#pragma HLS INTERFACE s_axilite port=start bundle=control
	#pragma HLS INTERFACE s_axilite port=end bundle=control

	uint y = 0;
	int depthsize_x = size_x + PAD_SZ; //322
	float e_d_squared_2 = 0.02;
	
	float pad_depth_array_1[N];
#pragma HLS ARRAY_PARTITION variable=pad_depth_array_1 cyclic factor=2
	float pad_depth_array_2[N];
#pragma HLS ARRAY_PARTITION variable=pad_depth_array_2 cyclic factor=2
	float pad_depth_array_3[N];
#pragma HLS ARRAY_PARTITION variable=pad_depth_array_3 cyclic factor=2

	float gaussian_array[FILTER_SZ];
	memcpy(gaussian_array,gaussian ,FILTER_SZ*sizeof(float));

	memcpy(pad_depth_array_1,pad_depth + (y*depthsize_x), N*sizeof(float));
	memcpy(pad_depth_array_2,pad_depth + ((y+1)*depthsize_x), N*sizeof(float));
	memcpy(pad_depth_array_3,pad_depth + ((y+2)*depthsize_x), N*sizeof(float));

	for (y = start; y < end; y++) {
		uint pos = y * size_x;
		
		for (uint x = 0; x < 320; x++) {
			#pragma HLS PIPELINE II=1

			float sum = gaussian_array[0] + gaussian_array[1] + gaussian_array[2] + gaussian_array[3] \
					+ gaussian_array[4] + gaussian_array[5] + gaussian_array[6] + gaussian_array[7] + gaussian_array[8];
			
			// full unroll
			float t1 = gaussian_array[0] * pad_depth_array_1[x];
			
			float t2 = gaussian_array[1] * pad_depth_array_1[x+1];
			
			float t3 = gaussian_array[2] * pad_depth_array_1[x+2];
			
			float t4 = gaussian_array[3] * pad_depth_array_2[x];
			
			float t5 = gaussian_array[4] * pad_depth_array_2[x + 1];
			
			float t6 = gaussian_array[5] * pad_depth_array_2[x + 2];
			
			float t7 = gaussian_array[6] * pad_depth_array_3[x];
			
			float t8 = gaussian_array[7] * pad_depth_array_3[x + 1];
			
			float t9 = gaussian_array[8] * pad_depth_array_3[x + 2];
			
			out[x + pos] = (t1 + t2 + t3 + t4 + t5 + t6 + t7 + t8 + t9) / sum;
		}

		memcpy(pad_depth_array_1, pad_depth_array_2, N*sizeof(float));
		memcpy(pad_depth_array_2, pad_depth_array_3, N*sizeof(float));
		memcpy(pad_depth_array_3, pad_depth+((y+3)*depthsize_x), N*sizeof(float));
	}

}
}