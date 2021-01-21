/*

 Copyright (c) 2021 Computer Systems Lab - University of Thessaly

 This code is licensed under the MIT License.

*/
#define _HW_

// HW implementation of the Bilateral Filter kernel
// enabled precise optimizations: BF_Unroll, BF_Pipe 
// enabled approximate optimizations: BF_Coeff

#include <kernels.hpp>
#include <string.h>

#define N 320
#define PAD_SZ 0

#define FILTER_SZ 9

extern "C" {


void bilateralFilterKernel(float* out, float* in, uint size_x, uint size_y,
		const float * gaussian, float e_d, int r) {


	#pragma HLS INTERFACE s_axilite port=return bundle=control
	#pragma HLS INTERFACE m_axi port=out offset=slave bundle=out max_read_burst_length=256 max_write_burst_length=256
	#pragma HLS INTERFACE s_axilite port=out	bundle=control
	#pragma HLS INTERFACE m_axi port=in offset=slave bundle=pad_depth max_read_burst_length=256 max_write_burst_length=256
	#pragma HLS INTERFACE s_axilite port=in	bundle=control
	#pragma HLS INTERFACE m_axi port=gaussian offset=slave bundle=gaussian
	#pragma HLS INTERFACE s_axilite port=gaussian	bundle=control
	#pragma HLS INTERFACE s_axilite port=size_x bundle=control
	#pragma HLS INTERFACE s_axilite port=size_y bundle=control
	#pragma HLS INTERFACE s_axilite port=e_d bundle=control
	#pragma HLS INTERFACE s_axilite port=r bundle=control


	uint y = 0;
	float e_d_squared_2 = e_d * e_d * 2;
	
	float pad_depth_array_1[N];
#pragma HLS ARRAY_PARTITION variable=pad_depth_array_1 cyclic factor=5
	float pad_depth_array_2[N];
#pragma HLS ARRAY_PARTITION variable=pad_depth_array_2 cyclic factor=5
	float pad_depth_array_3[N];
#pragma HLS ARRAY_PARTITION variable=pad_depth_array_3 cyclic factor=5


	float gaussian_array[FILTER_SZ];
	memcpy(gaussian_array, gaussian, FILTER_SZ*sizeof(float));

	memcpy(pad_depth_array_1, in, (N+PAD_SZ)*sizeof(float));
	memcpy(pad_depth_array_2, pad_depth_array_1, (N+PAD_SZ)*sizeof(float));
	memcpy(pad_depth_array_3, in + y*size_x, (N+PAD_SZ)*sizeof(float));

	for (y = 0; y < size_y; y++) {
		uint pos = y * size_x;
		
		for (uint x = 0; x < size_x; x++) {
			#pragma HLS pipeline II=1
			
			float sum = 0.0f;
			float t = 0.0f;

			const float center = pad_depth_array_2[x];

			
			// i = -1
			int curPosx = clamp(x - 1, 0u, size_x - 1);
			float curPix = pad_depth_array_1[curPosx];

			float mod = sq(curPix - center);
			float factor = gaussian_array[0]
					*expf(-mod / e_d_squared_2);
			t += factor * curPix;
			sum += factor;


			curPix = pad_depth_array_2[curPosx];
			
			mod = sq(curPix - center);
			factor = gaussian_array[3]
					*expf(-mod / e_d_squared_2);
			t += factor * curPix;
			sum += factor;


			curPix = pad_depth_array_3[curPosx];

			mod = sq(curPix - center);
				factor = gaussian_array[6]
					*expf(-mod / e_d_squared_2);
			t += factor * curPix;
			sum += factor;


			// i = 0
			curPosx = clamp(x , 0u, size_x - 1);
			curPix = pad_depth_array_1[curPosx];
			
			mod = sq(curPix - center);
			factor = gaussian_array[1]
					*expf(-mod / e_d_squared_2);
			t += factor * curPix;
			sum += factor;


			curPix = pad_depth_array_2[curPosx];
			
			mod = sq(curPix - center);
			factor = gaussian_array[4]
					*expf(-mod / e_d_squared_2);
			t += factor * curPix;
			sum += factor;


			curPix = pad_depth_array_3[curPosx];
			
			mod = sq(curPix - center);
				factor = gaussian_array[7]
					*expf(-mod / e_d_squared_2);
			t += factor * curPix;
			sum += factor;

			
			// i = 1
			curPosx = clamp(x + 1, 0u, size_x - 1);
			curPix = pad_depth_array_1[curPosx];

			mod = sq(curPix - center);
			factor = gaussian_array[2]
					*expf(-mod / e_d_squared_2);
			t += factor * curPix;
			sum += factor;


			curPix = pad_depth_array_2[curPosx];
			
			mod = sq(curPix - center);
			factor = gaussian_array[5]
					*expf(-mod / e_d_squared_2);
			t += factor * curPix;
			sum += factor;


			curPix = pad_depth_array_3[curPosx];
			
			mod = sq(curPix - center);
				factor = gaussian_array[8]
					*expf(-mod / e_d_squared_2);
			t += factor * curPix;
			sum += factor;

			
			out[x + pos] = t / sum;
		}

		memcpy(pad_depth_array_1, pad_depth_array_2, (N+PAD_SZ)*sizeof(float));
		memcpy(pad_depth_array_2, pad_depth_array_3, (N+PAD_SZ)*sizeof(float));
		if (y < N-2)
			memcpy(pad_depth_array_3, in+((y+2)*size_x), (N+PAD_SZ)*sizeof(float));
	}
}


}
