/*

 Copyright (c) 2021 Computer Systems Lab - University of Thessaly

 This code is licensed under the MIT License.

*/

#define _HW_

#include <kernels.hpp>
#include <string.h>


// unoptimized HW implementation of the Bilateral Filter kernel

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


	uint y;
	float e_d_squared_2 = e_d * e_d * 2;

	for (y = 0; y < size_y; y++) {
		#pragma HLS pipeline off
		for (uint x = 0; x < size_x; x++) {
			#pragma HLS pipeline off

			uint pos = x + y * size_x;
			if (in[pos] == 0) {
				out[pos] = 0;
				continue;
			}

			float sum = 0.0f;
			float t = 0.0f;

			const float center = in[pos];

			for (int i = -r; i <= r; ++i) {
				#pragma HLS pipeline off
				for (int j = -r; j <= r; ++j) {
					#pragma HLS pipeline off
					uint2 curPos = make_uint2(clamp(x + i, 0u, size_x - 1),
							clamp(y + j, 0u, size_y - 1));
					const float curPix = in[curPos.x + curPos.y * size_x];
					if (curPix > 0) {
						const float mod = sq(curPix - center);
						const float factor = gaussian[i + r]
								* gaussian[j + r]
								* expf(-mod / e_d_squared_2);
						t += factor * curPix;
						sum += factor;
					}
				}
			}
			out[pos] = t / sum;
		}
	}
}


}
