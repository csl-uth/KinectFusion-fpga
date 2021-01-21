/*

 Copyright (c) 2021 Computer Systems Lab - University of Thessaly

 This code is licensed under the MIT License.

*/
#define _HW_


// HW implementation of the Bilateral Filter kernel
// enabled optimizations: BF_Pipe, BF_Unroll, BF_Pad

#include <kernels.hpp>
#include <string.h>

#define N 324
#define PAD_SZ 4
#define FILTER_SZ 15
#define _HW_


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

	uint y;
	int depthsize_x = size_x + PAD_SZ; //324
	float e_d_squared_2 = 0.02;
	
	float pad_depth_array_1[N];
#pragma HLS ARRAY_PARTITION variable=pad_depth_array_1 cyclic factor=5
	float pad_depth_array_2[N];
#pragma HLS ARRAY_PARTITION variable=pad_depth_array_2 cyclic factor=5
	float pad_depth_array_3[N];
#pragma HLS ARRAY_PARTITION variable=pad_depth_array_3 cyclic factor=5
	float pad_depth_array_4[N];
#pragma HLS ARRAY_PARTITION variable=pad_depth_array_4 cyclic factor=5
	float pad_depth_array_5[N];
#pragma HLS ARRAY_PARTITION variable=pad_depth_array_5 cyclic factor=5

	float gaussian_array[15];
	memcpy(gaussian_array, gaussian, 15*sizeof(float));
	

	memcpy(pad_depth_array_1,pad_depth + (y*depthsize_x), N*sizeof(float));
	memcpy(pad_depth_array_2,pad_depth + ((y+1)*depthsize_x), N*sizeof(float));
	memcpy(pad_depth_array_3,pad_depth + ((y+2)*depthsize_x), N*sizeof(float));
	memcpy(pad_depth_array_4,pad_depth + ((y+3)*depthsize_x), N*sizeof(float));
	memcpy(pad_depth_array_5,pad_depth + ((y+4)*depthsize_x), N*sizeof(float));

	for (y = start; y < end; y++) {
		uint pos = y * size_x;
		
		for (uint x = 0; x < 320; x++) {
			#pragma HLS PIPELINE II=1

			float sum = 0.0f;
			float t = 0.0f;


			const float center = pad_depth_array_3[x+2];

			float curPix = pad_depth_array_1[x];


			float mod = sq(curPix - center);
			float factor = gaussian_array[0]
							*expf(-mod / e_d_squared_2);
			t += factor * curPix;
			sum += factor;

			curPix = pad_depth_array_2[x];


			mod = sq(curPix - center);
			factor = gaussian_array[1]
						*expf(-mod / e_d_squared_2);
			t += factor * curPix;
			sum += factor;

			curPix = pad_depth_array_3[x];


			mod = sq(curPix - center);
			factor = gaussian_array[2]
						*expf(-mod / e_d_squared_2);
			t += factor * curPix;
			sum += factor;


			curPix = pad_depth_array_4[x];

			mod = sq(curPix - center);
			factor = gaussian_array[3]
						*expf(-mod / e_d_squared_2);
			t += factor * curPix;
			sum += factor;

			
			curPix = pad_depth_array_5[x];

			mod = sq(curPix - center);
			factor = gaussian_array[4]
						*expf(-mod / e_d_squared_2);
			t += factor * curPix;
			sum += factor;

			// i = -1
			
			curPix = pad_depth_array_1[x+1];

			mod = sq(curPix - center);
			factor = gaussian_array[1]
						*expf(-mod / e_d_squared_2);
			t += factor * curPix;
			sum += factor;

			
			curPix = pad_depth_array_2[x+1];

			mod = sq(curPix - center);
			factor = gaussian_array[5]
						*expf(-mod / e_d_squared_2);
			t += factor * curPix;
			sum += factor;

			
			curPix = pad_depth_array_3[x+1];

			mod = sq(curPix - center);
			factor = gaussian_array[6]
						*expf(-mod / e_d_squared_2);
			t += factor * curPix;
			sum += factor;

			
			curPix = pad_depth_array_4[x+1];

			mod = sq(curPix - center);
			factor = gaussian_array[7]
						*expf(-mod / e_d_squared_2);
			t += factor * curPix;
			sum += factor;

			
			curPix = pad_depth_array_5[x+1];

			mod = sq(curPix - center);
			factor = gaussian_array[8]
						*expf(-mod / e_d_squared_2);
			t += factor * curPix;
			sum += factor;

			// i = 0
			
			curPix = pad_depth_array_1[x+2];

			mod = sq(curPix - center);
			factor = gaussian_array[2]
						*expf(-mod / e_d_squared_2);
			t += factor * curPix;
			sum += factor;

			
			curPix = pad_depth_array_2[x+2];

			mod = sq(curPix - center);
			factor = gaussian_array[6]
						*expf(-mod / e_d_squared_2);
			t += factor * curPix;
			sum += factor;

			
			curPix = pad_depth_array_3[x+2];

			mod = sq(curPix - center);
			factor = gaussian_array[9]
						*expf(-mod / e_d_squared_2);
			t += factor * curPix;
			sum += factor;

			
			curPix = pad_depth_array_4[x+2];

			mod = sq(curPix - center);
			factor = gaussian_array[10]
						*expf(-mod / e_d_squared_2);
			t += factor * curPix;
			sum += factor;

			
			curPix = pad_depth_array_5[x+2];

			mod = sq(curPix - center);
			factor = gaussian_array[11]
						*expf(-mod / e_d_squared_2);
			t += factor * curPix;
			sum += factor;

			// i = 1
			
			curPix = pad_depth_array_1[x+3];

			mod = sq(curPix - center);
			factor = gaussian_array[3]
						*expf(-mod / e_d_squared_2);
			t += factor * curPix;
			sum += factor;

			
			curPix = pad_depth_array_2[x+3];

			mod = sq(curPix - center);
			factor = gaussian_array[7]
						*expf(-mod / e_d_squared_2);
			t += factor * curPix;
			sum += factor;

			
			curPix = pad_depth_array_3[x+3];

			mod = sq(curPix - center);
			factor = gaussian_array[10]
						*expf(-mod / e_d_squared_2);
			t += factor * curPix;
			sum += factor;

			
			curPix = pad_depth_array_4[x+3];

			mod = sq(curPix - center);
			factor = gaussian_array[12]
						*expf(-mod / e_d_squared_2);
			t += factor * curPix;
			sum += factor;

			
			curPix = pad_depth_array_5[x+3];

			mod = sq(curPix - center);
			factor = gaussian_array[13]
						*expf(-mod / e_d_squared_2);
			t += factor * curPix;
			sum += factor;

			// i = 2
			
			curPix = pad_depth_array_1[x+4];

			mod = sq(curPix - center);
			factor = gaussian_array[4]
						*expf(-mod / e_d_squared_2);
			t += factor * curPix;
			sum += factor;

			
			curPix = pad_depth_array_2[x+4];

			mod = sq(curPix - center);
			factor = gaussian_array[8]
						*expf(-mod / e_d_squared_2);
			t += factor * curPix;
			sum += factor;

			
			curPix = pad_depth_array_3[x+4];

			mod = sq(curPix - center);
			factor = gaussian_array[11]
						*expf(-mod / e_d_squared_2);
			t += factor * curPix;
			sum += factor;

			
			curPix = pad_depth_array_4[x+4];

			mod = sq(curPix - center);
			factor = gaussian_array[13]
						*expf(-mod / e_d_squared_2);
			t += factor * curPix;
			sum += factor;

			
			curPix = pad_depth_array_5[x+4];

			mod = sq(curPix - center);
			factor = gaussian_array[14]
						*expf(-mod / e_d_squared_2);
			t += factor * curPix;
			sum += factor;

			out[x + pos] = t / sum;
		}

		memcpy(pad_depth_array_1, pad_depth_array_2, N*sizeof(float));
		memcpy(pad_depth_array_2, pad_depth_array_3, N*sizeof(float));
		memcpy(pad_depth_array_3, pad_depth_array_4, N*sizeof(float));
		memcpy(pad_depth_array_4, pad_depth_array_5, N*sizeof(float));
		memcpy(pad_depth_array_5, pad_depth+((y+5)*depthsize_x), N*sizeof(float));
	}

}
}